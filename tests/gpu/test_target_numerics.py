"""Target precision boundaries, fractional FP4 scales and graph mutation."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('SHARAR_RUN_GPU') != '1', reason='opt-in GB10 tests')


@pytest.mark.parametrize('rows,k', [(1,128), (16,5120), (37,17408), (128,17408)])
@pytest.mark.parametrize('scale', [0.5, 286.7200012207031, 764.586669921875, 111.41968536376953])
@torch.inference_mode()
def test_silu_and_fractional_quantization_match_cuda(rows, k, scale):
    from sharar.kernels.nvfp4_quant import nvfp4_quantize, silu_mul_nvfp4_quantize
    from sharar.kernels.ops import fp4_quantize
    from sharar.weights.modelopt import swizzle_block_scales

    torch.manual_seed(412)
    x = torch.randn(rows, 2*k, device='cuda', dtype=torch.bfloat16) * 3
    gs = torch.tensor(scale, device='cuda')
    mask = swizzle_block_scales(torch.ones(rows, k//16, device='cuda', dtype=torch.uint8)).bool()
    for seed in (412, 413):
        torch.manual_seed(seed)
        x.normal_()
        g, u = x.float().chunk(2, -1)
        product = (torch.nn.functional.silu(g) * u).bfloat16()
        expected, scales = fp4_quantize(product, gs)
        for quantize in (lambda: nvfp4_quantize(product, gs), lambda: silu_mul_nvfp4_quantize(x, gs)):
            actual, sf = quantize()
            assert torch.equal(actual, expected)
            assert torch.equal(sf.view(torch.uint8)[mask], scales.view(torch.uint8)[mask])
        if seed == 412:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_actual, graph_sf = silu_mul_nvfp4_quantize(x, gs)
        graph.replay()
        assert torch.equal(graph_actual, expected)
        assert torch.equal(graph_sf.view(torch.uint8)[mask], scales.view(torch.uint8)[mask])


@pytest.mark.parametrize('rows', [1, 16, 257])
@pytest.mark.parametrize('inplace', [False, True])
@torch.inference_mode()
def test_target_norm_retains_fp32_sum_and_residual_ownership(rows, inplace):
    from sharar.model.norm import GemmaRMSNorm

    torch.manual_seed(915)
    x = torch.randn(rows, 5120, device='cuda', dtype=torch.bfloat16)
    residual = torch.randn_like(x) * .01
    before, x_before = residual.clone(), x.clone()
    weight = torch.randn(5120, device='cuda', dtype=torch.bfloat16) * .1
    summed = x.float() + residual.float()
    expected = (summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
                * (1 + weight.float())).bfloat16()
    out, retained = GemmaRMSNorm(weight, 1e-6).fused_add(x, residual, inplace=inplace)
    assert torch.equal(retained, summed.bfloat16())
    assert (retained.data_ptr() == residual.data_ptr()) == inplace
    if not inplace:
        assert torch.equal(residual, before)
    assert torch.equal(x, x_before)
    assert (out == expected).float().mean() > .999
    torch.testing.assert_close(out.float(), expected.float(), rtol=.008, atol=1e-6)


@pytest.mark.parametrize('offset', [0, 9999, 32760])
@torch.inference_mode()
def test_target_rotary_keeps_fp32_cache_and_products(offset):
    from sharar.model.rope import PartialRotary
    from sharar.kernels.qk_norm_rope import qk_norm_rope

    torch.manual_seed(411)
    t, hq, hk, d, r = 7, 24, 4, 256, 64
    qkv = torch.randn(t, 2*(hq+hk)*d, device='cuda', dtype=torch.bfloat16)
    w = torch.zeros(d, device='cuda', dtype=torch.bfloat16)
    rope = PartialRotary(d, r, 1e7, 32768, torch.device('cuda'))
    assert rope.cos_sin.dtype == torch.float32
    pos = torch.arange(offset, offset+t, device='cuda')
    q, k = qk_norm_rope(qkv, pos, rope.cos_sin, w, w, hq, hk, d, r, 1e-6)
    sources = [qkv[:, :2*hq*d].view(t,hq,2*d)[...,:d], qkv[:, 2*hq*d:2*hq*d+hk*d].view(t,hk,d)]
    for actual, x in zip((q,k),sources):
        xf=x.float()
        normalized=(xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+1e-6)).bfloat16()
        expected=rope(pos,normalized)
        assert (actual==expected).float().mean() > .999
        torch.testing.assert_close(actual.float(),expected.float(),rtol=.008,atol=1e-5)
