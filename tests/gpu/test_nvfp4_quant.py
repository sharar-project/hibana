"""The quantizer's hardware E2M1 rounding against the comparison chain it replaced, for every FP32 bit pattern; and the
prefill launch geometry against the decode one."""

import pytest
import torch
import triton
import triton.language as tl

from sharar.kernels.nvfp4_quant import e2m1_pairs

BLOCK = 4096


@triton.jit
def _chain_pairs(v, N: tl.constexpr):
    """The previous rounding: E2M1 round-to-nearest-even over {0, .5, 1, 1.5, 2, 3, 4, 6} by comparisons."""
    a = tl.minimum(tl.abs(v), 6.0)
    q = tl.where(a <= 0.25, 0.0, tl.where(a < 0.75, 0.5, tl.where(a <= 1.25, 1.0, tl.where(a < 1.75, 1.5,
        tl.where(a <= 2.5, 2.0, tl.where(a < 3.5, 3.0, tl.where(a <= 5.0, 4.0, 6.0)))))))
    code = tl.where(q == 0.0, 0, tl.where(q == 0.5, 1, tl.where(q == 1.0, 2, tl.where(q == 1.5, 3,
           tl.where(q == 2.0, 4, tl.where(q == 3.0, 5, tl.where(q == 4.0, 6, 7))))))).to(tl.int32)
    code = tl.where(v.to(tl.int32, bitcast=True) < 0, code | 8, code)
    pairs = tl.reshape(code, (N // 2, 2))
    return tl.sum(pairs * tl.reshape(tl.where(tl.arange(0, N) % 2 == 0, 1, 16), (N // 2, 2)), axis=1).to(tl.uint8)


@triton.jit
def _mismatches(start, count_ptr, first_ptr, N: tl.constexpr):
    bits = start + tl.program_id(0).to(tl.int64) * N + tl.arange(0, N)
    v = bits.to(tl.uint32).to(tl.int32, bitcast=True).to(tl.float32, bitcast=True)
    wrong = (e2m1_pairs(v, N) != _chain_pairs(v, N)).to(tl.int32)
    n = tl.sum(wrong)
    if n > 0:
        tl.atomic_add(count_ptr, n)
        tl.atomic_min(first_ptr, tl.program_id(0).to(tl.int64) * N + start)


def test_hardware_e2m1_equals_the_comparison_chain_for_every_fp32():
    count = torch.zeros(1, dtype=torch.int32, device="cuda")
    first = torch.full((1,), 2**40, dtype=torch.int64, device="cuda")
    chunk = 2**30
    for start in range(0, 2**32, chunk):
        _mismatches[(chunk // BLOCK,)](start, count, first, N=BLOCK)
    torch.cuda.synchronize()
    assert count.item() == 0, f"{count.item()} mismatched pairs, first block at bit pattern {first.item():#x}"


@pytest.mark.parametrize("rows", [1, 16, 255, 256, 2048, 3000])
@pytest.mark.parametrize("silu", [False, True])
@torch.inference_mode()
def test_prefill_geometry_matches_the_decode_geometry(rows, silu, monkeypatch):
    from sharar.kernels import nvfp4_quant
    torch.manual_seed(rows)
    k = 17408 if silu else 5120
    x = torch.randn(rows, 2 * k if silu else k, device="cuda", dtype=torch.bfloat16) * 3
    scale = torch.tensor([448 * 6 / 12.0], device="cuda")
    quantize = nvfp4_quant.silu_mul_nvfp4_quantize if silu else nvfp4_quant.nvfp4_quantize
    q, sf = quantize(x, scale)
    q, sf = q.clone(), sf.clone()
    monkeypatch.setattr(nvfp4_quant, "PREFILL_ROWS", 10**9 if rows >= 256 else 1)  # the other geometry
    q2, sf2 = quantize(x, scale)
    assert torch.equal(q, q2) and torch.equal(sf.view(torch.uint8), sf2.view(torch.uint8))
