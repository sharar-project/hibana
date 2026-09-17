"""Calibrated codes preserve the W3 kernel and CUDA graph contracts."""
import os

import pytest
import torch

from sharar.kernels import w3
from sharar.weights.w3_calibration import quantize

pytestmark = pytest.mark.skipif(os.environ.get('SHARAR_RUN_GPU') != '1', reason='opt-in GB10 tests')


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
