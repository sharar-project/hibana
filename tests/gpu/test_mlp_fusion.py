"""The fused prefill gate/up GEMM (`kernels/mlp_fusion.py`) writes the bytes of the unfused path: the gate/up GEMM on
the concatenated weights, then `silu_mul_nvfp4_quantize`. Random NVFP4 weights of the 27B's MLP shape (one Spark, and
one rank's half on two), at prefill row counts that are and are not multiples of 128; the interleaved weights' unfused
path and its 3-bit copy match too."""
import pytest
import torch

from sharar.kernels import mlp_fusion, ops, w3
from sharar.kernels.nvfp4_quant import nvfp4_quantize, silu_mul_nvfp4_quantize
from sharar.weights.modelopt import Nvfp4Linear, interleave_pairs, pair_order, swizzle_block_scales

H = 5120
pytestmark = pytest.mark.skipif(not mlp_fusion.available(torch.device("cuda")), reason="the fused kernel needs SM121")


def random_linear(n: int, k: int, seed: int) -> Nvfp4Linear:
    g = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=g)
    scales = (torch.rand((n, k // 16), device="cuda", generator=g) * 2 + 0.25).to(torch.float8_e4m3fn)
    return Nvfp4Linear(weight=weight, weight_scale_swizzled=swizzle_block_scales(scales.view(torch.uint8)).view(torch.float8_e4m3fn),
                       alpha=torch.tensor(0.0021, device="cuda"), input_scale_inv=torch.tensor(1 / 0.37, device="cuda"),
                       out_features=n, in_features=k)


def copy(lin: Nvfp4Linear) -> Nvfp4Linear:
    return Nvfp4Linear(**{**lin.__dict__, "weight": lin.weight.clone(), "weight_scale_swizzled": lin.weight_scale_swizzled.clone()})


@pytest.mark.parametrize("inter", [17408, 8704])
@pytest.mark.parametrize("rows", [128, 129, 300, 2048, 2100])
def test_fused_writes_the_unfused_bytes(inter, rows):
    natural = random_linear(2 * inter, H, seed=rows + inter)
    paired = interleave_pairs(copy(natural), mlp_fusion.PAIRS)
    down_scale = torch.tensor(1 / 0.11, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(rows)
    for spread in (1.0, 6.0):
        x = (torch.randn(rows, H, device="cuda", generator=g) * spread).bfloat16()
        pre = nvfp4_quantize(x, natural.input_scale_inv)
        pre = (pre[0], pre[1].clone())
        # the scale buffers are shared per shape: copy each result before the next call
        want_q, want_sf = (t.clone() for t in silu_mul_nvfp4_quantize(ops.nvfp4_linear(None, natural, prequantized=pre), down_scale))
        got_q, got_sf = (t.clone() for t in mlp_fusion.gate_up_silu_quantize(pre, paired, down_scale))
        assert torch.equal(got_q, want_q)
        assert torch.equal(got_sf.view(torch.uint8)[:rows], want_sf.view(torch.uint8)[:rows])
        # the interleaved weights' unfused path (graphs, short calls) reads the same pairs
        q, sf = silu_mul_nvfp4_quantize(ops.nvfp4_linear(None, paired, prequantized=pre), down_scale, paired.pairs)
        assert torch.equal(q, want_q) and torch.equal(sf.view(torch.uint8)[:rows], want_sf.view(torch.uint8)[:rows])


def test_interleaved_rows_and_their_3bit_copy():
    natural = random_linear(2 * 1024, 1024, seed=3)
    paired = interleave_pairs(copy(natural), mlp_fusion.PAIRS)
    order = pair_order(natural.out_features, mlp_fusion.PAIRS).cuda()
    assert torch.equal(paired.dequantize(), natural.dequantize()[order])
    # a calibrated copy is stored concatenated and interleaved at load; a nearest-level one packs the interleaved rows
    assert all(torch.equal(a, b) for a, b in zip(
        (lambda c: (c.planes, c.scales, c.row_scale))(w3.interleave_rows(w3.pack(natural.dequantize()), paired.pairs)),
        (lambda c: (c.planes, c.scales, c.row_scale))(w3.pack(paired.dequantize()))))
    x = torch.randn(8, 1024, device="cuda").bfloat16()
    natural_out = ops.silu_and_mul(x @ natural.dequantize().T.bfloat16())
    assert torch.equal(ops.silu_and_mul(x @ paired.dequantize().T.bfloat16(), paired.pairs), natural_out)
