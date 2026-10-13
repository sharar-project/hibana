"""The quantizing residual norms equal FlashInfer's norm followed by the separate quantizers, bit for bit."""

import pytest
import torch

ROWS = [1, 2, 3, 127, 128, 2048, 3000, 4096]


def _inputs(rows, seed):
    g = torch.Generator("cuda").manual_seed(seed)
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16, generator=g) * 3
    residual = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16, generator=g) * 20
    weight = torch.randn(5120, device="cuda", dtype=torch.bfloat16, generator=g) * 0.3
    return x, residual, weight


def _reference_norm(x, residual, weight):
    from flashinfer.norm import gemma_fused_add_rmsnorm
    out, res = x.clone(), residual.clone()
    gemma_fused_add_rmsnorm(out, res, weight, 1e-6)
    return out, res


@pytest.mark.parametrize("rows", ROWS)
@torch.inference_mode()
def test_fp8_epilogue(rows):
    from sharar.kernels.cute.residual_norm import residual_norm_fp8
    from sharar.kernels.fp8_quant import quant_fp8
    x, residual, weight = _inputs(rows, rows)
    for scale in (0.004, 0.05, 1.0):  # 0.004 clamps many values at 448
        s = torch.tensor([scale], device="cuda")
        want, want_res = _reference_norm(x, residual, weight)
        want_q = quant_fp8(want, s)
        got, got_res = x.clone(), residual.clone()
        got_q = residual_norm_fp8(got, got_res, weight, 1e-6, s)
        assert torch.equal(got, want) and torch.equal(got_res, want_res), (rows, scale)
        assert torch.equal(got_q.view(torch.uint8), want_q.view(torch.uint8)), (rows, scale)


@pytest.mark.parametrize("rows", ROWS)
@torch.inference_mode()
def test_nvfp4_epilogue(rows):
    from sharar.kernels.cute.residual_norm import residual_norm_nvfp4
    from sharar.kernels.nvfp4_quant import nvfp4_quantize
    x, residual, weight = _inputs(rows, 100 + rows)
    for amax in (2.0, 12.0, 300.0):
        gs = torch.tensor([448 * 6 / amax], device="cuda")
        want, want_res = _reference_norm(x, residual, weight)
        want_q, want_sf = nvfp4_quantize(want, gs)
        want_sf = want_sf.clone()  # the scale buffer is shared per shape
        got_res = residual.clone()
        got_q, got_sf = residual_norm_nvfp4(x.clone(), got_res, weight, 1e-6, gs)
        assert torch.equal(got_res, want_res), (rows, amax)
        assert torch.equal(got_q, want_q), (rows, amax)
        assert torch.equal(got_sf.view(torch.uint8), want_sf.view(torch.uint8)), (rows, amax)
