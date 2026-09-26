"""Check both residual rounding modes against independent equations.

These checks cover the two experimental Triton modes. Serving uses the
native CUDA norm with the FP32 sum and its reference reduction order.
"""

import pytest
import torch



@pytest.mark.parametrize('rows', [1, 16, 257])
@pytest.mark.parametrize('quantized', [False, True])
@pytest.mark.parametrize('fp32_sum', [False, True])
def test_gemma_norm_residual_rounding_and_quantization(rows, quantized, fp32_sum):
    from sharar.kernels.fused_norm import gemma_rmsnorm
    torch.manual_seed(915)
    x = torch.randn(rows, 5120, device='cuda', dtype=torch.bfloat16)
    residual = torch.randn_like(x) * .01
    weight = torch.randn(5120, device='cuda', dtype=torch.bfloat16) * .1
    summed = x.float() + residual.float()
    numerator = summed if fp32_sum else summed.bfloat16().float()
    expected = (numerator * torch.rsqrt(numerator.square().mean(-1, keepdim=True) + 1e-6)
                * (1 + weight.float())).bfloat16()
    scale = torch.tensor(.03, device='cuda') if quantized else None
    out, packed = gemma_rmsnorm(x, weight, 1e-6, residual=residual, fp8_scale=scale,
                               fp32_residual_sum=fp32_sum)
    assert torch.equal(residual, summed.bfloat16())
    # Reduction order can affect a value exactly at a BF16 midpoint. The two
    # residual equations differ in around 20% of elements on this fixture.
    assert (out == expected).float().mean() > .999
    torch.testing.assert_close(out.float(), expected.float(), rtol=.008, atol=1e-6)
    if quantized:
        reference = (out.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        assert torch.equal(packed.view(torch.uint8), reference.view(torch.uint8))


@pytest.mark.parametrize('fp32_sum', [False, True])
def test_residual_norm_graph_replay_reads_new_operands(fp32_sum):
    from sharar.kernels.fused_norm import gemma_rmsnorm
    x = torch.ones(16, 5120, device='cuda', dtype=torch.bfloat16)
    residual = torch.empty_like(x)
    weight = torch.linspace(-.1, .1, 5120, device='cuda').bfloat16()
    for _ in range(2):
        residual.zero_()
        gemma_rmsnorm(x, weight, 1e-6, residual=residual, fp32_residual_sum=fp32_sum)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, _ = gemma_rmsnorm(x, weight, 1e-6, residual=residual, fp32_residual_sum=fp32_sum)
    for seed in (71, 72):
        torch.manual_seed(seed)
        x.normal_()
        residual.normal_(std=.01)
        summed = x.float() + residual.float()
        numerator = summed if fp32_sum else summed.bfloat16().float()
        expected = (numerator * torch.rsqrt(numerator.square().mean(-1, keepdim=True)+1e-6)
                    * (1+weight.float())).bfloat16()
        graph.replay()
        assert torch.equal(residual, summed.bfloat16())
        assert (out == expected).float().mean() > .999
