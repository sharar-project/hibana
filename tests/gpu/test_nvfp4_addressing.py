"""Regression for signed-int32 row offsets in the 27B's large MLP prefill."""
import pytest
import torch


@pytest.mark.parametrize("fused", [False, True])
@torch.inference_mode()
def test_large_strided_rows_match_small_reference(fused):
    from sharar.kernels.nvfp4_quant import nvfp4_quantize, silu_mul_nvfp4_quantize

    m, k = 65536, 17408
    # A preceding owned guard makes the old wrapped loads safe to reproduce.
    # Both paths have stride 34816; plain quantization takes a strided view.
    storage = torch.zeros((2*m, 2*k), device="cuda", dtype=torch.bfloat16)
    x = storage[m:]
    if not fused:
        x = x[:, :k]
    x.fill_(2)
    gs = torch.ones((), device="cuda")
    quantize = silu_mul_nvfp4_quantize if fused else nvfp4_quantize
    rows = torch.tensor([0, 2047, 61439, 61679, 61680, 61681, 63487, 65535], device="cuda")
    c = torch.arange(k//16, device="cuda", dtype=torch.int64)
    offsets = ((rows[:, None]//128)*(128*(k//16)) + (rows[:, None]%32)*16
               + ((rows[:, None]//32)%4)*4 + (c//4)*512 + c%4)
    ref_offsets = (c//4)*512+c%4
    for value in (2., -1.):
        x.fill_(value)
        refq, refsf = quantize(x[:2], gs)
        expected_q = refq[0].clone()
        expected_sf = refsf.view(-1)[ref_offsets].view(torch.uint8).clone()
        q, sf = quantize(x, gs)
        assert torch.equal(q[rows], expected_q.expand(len(rows), -1))
        assert torch.equal(sf.view(-1)[offsets].view(torch.uint8), expected_sf.expand(len(rows), -1))
