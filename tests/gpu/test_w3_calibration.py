"""Calibrated codes preserve the W3 kernel and CUDA graph contracts."""

import pytest
import torch

from sharar.kernels import w3
from sharar.weights.w3_calibration import quantize



def test_captured_calibration_matches_eager():
    torch.manual_seed(17)
    x = torch.randn(1024, 256, device='cuda')
    weight = torch.randn(256, 256, device='cuda') * .02
    covariance = x.T @ x / len(x)
    eager, _ = quantize(weight, covariance, capture=False, rows_per_chunk=128)
    captured, _ = quantize(weight, covariance, capture=True, rows_per_chunk=128)
    assert torch.equal(eager.planes, captured.planes)
    assert torch.equal(eager.scales, captured.scales)
    assert torch.equal(eager.row_scale, captured.row_scale)


@pytest.mark.parametrize('n,k', [(128, 256), (10240, 5120), (5120, 17408)])
def test_explicit_calibrated_codes_execute_in_serving_kernel(n, k):
    torch.manual_seed(29)
    codes = torch.randint(1, 7, (n, k), device='cuda', dtype=torch.int32)
    scales = (torch.rand(n, k // 32, device='cuda') * 100).to(torch.float8_e4m3fn)
    row_scale = torch.rand(n, device='cuda') * .001 + .0001
    packed = w3.pack_codes(codes, scales, row_scale)
    reference = w3.dequantize(packed)
    del codes
    for rows in (1, 16, 32, 64):
        x = torch.randn(rows, k, device='cuda', dtype=torch.bfloat16)
        x[:, ::128] *= 100  # outliers exercise activation group scaling
        xq, xs = w3.quant_groups(x)
        expected = (xq.float() * xs.repeat_interleave(32, dim=1)) @ reference.T
        actual = w3.w3_linear(x, packed)
        error = (actual.float() - expected).norm() / expected.norm()
        assert float(error) < .004
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = w3.w3_linear(x, packed)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(actual, captured)


@pytest.mark.parametrize('per_row_scale', [False, True])
@torch.inference_mode()
def test_qkv_static_scales_match_reference_after_graph_input_changes(per_row_scale):
    """Exercise both halves of the 64-row QKV projection with distinct rows."""
    torch.manual_seed(31)
    n, k = 14336, 5120
    codes = torch.randint(0, 8, (n, k), device='cuda', dtype=torch.uint8)
    scales = (torch.rand(n, k // 32, device='cuda') * 100).to(torch.float8_e4m3fn)
    row_scale = torch.rand(n, device='cuda') * .001 + .0001
    packed = w3.pack_codes(codes, scales, row_scale)
    reference = w3.dequantize(packed)
    del codes
    # A contiguous view with a storage offset catches accidental base-pointer
    # assumptions; sharply different halves expose dropped or duplicated rows.
    storage = torch.randn(65, k, device='cuda')
    x = storage[1:]
    x[:32] *= .125
    x[32:] *= 16
    x8 = storage.to(torch.float8_e4m3fn)[1:]
    scale = (torch.linspace(.25, 2, 64, device='cuda') if per_row_scale
             else torch.tensor([.75], device='cuda'))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = w3.w3_linear(x8, packed, scale)
    for shift in (0, 17):
        x8.copy_(x.roll(shift, dims=0).to(torch.float8_e4m3fn))
        expected = (x8.float() * scale[:, None]) @ reference.T
        eager = w3.w3_linear(x8, packed, scale)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(eager, captured)
        assert float((captured.float() - expected).norm() / expected.norm()) < .004
