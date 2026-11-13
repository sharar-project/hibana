"""Compiled target normalization precision, ownership and graph mutation."""

import pytest
import torch



@pytest.mark.parametrize('rows', [1, 16, 64])
@torch.inference_mode()
def test_initial_verify_norm_graph_reads_new_embeddings(rows):
    from hibana.model.norm import GemmaRMSNorm
    from hibana.kernels.fp8_quant import quant_fp8

    torch.manual_seed(105)
    x = torch.randn(rows, 5120, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(5120, device='cuda', dtype=torch.bfloat16) * .1
    scale = torch.tensor(.0237, device='cuda')
    norm = GemmaRMSNorm(weight, 1e-6)
    for _ in range(2):
        norm.fused_add_quant(x, None, scale, verify=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, packed, retained = norm.fused_add_quant(x, None, scale, verify=True)
    for seed in [106, 107]:
        torch.manual_seed(seed)
        x.normal_()
        original = x.clone()
        xf = x.float()
        expected = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
                    * (1 + weight.float())).bfloat16()
        graph.replay()
        assert torch.equal(x, original) and torch.equal(retained, original)
        assert (out == expected).float().mean() > .9999
        torch.testing.assert_close(out.float(), expected.float(), rtol=.008, atol=1e-6)
        assert torch.equal(packed.view(torch.uint8), quant_fp8(out, scale).view(torch.uint8))


@pytest.mark.parametrize('rows', [1, 16, 64])
@pytest.mark.parametrize('verify', [False, True])
@pytest.mark.parametrize('inplace', [False, True])
@torch.inference_mode()
def test_phase_specific_norm_uses_unrounded_variance(rows, verify, inplace):
    from hibana.model.norm import GemmaRMSNorm
    from hibana.kernels.fp8_quant import quant_fp8

    torch.manual_seed(841)
    x = torch.randn(rows, 5120, device='cuda', dtype=torch.bfloat16)
    residual = torch.randn_like(x) * .01
    original, x_original = residual.clone(), x.clone()
    w = torch.randn(5120, device='cuda', dtype=torch.bfloat16) * .1
    scale = torch.tensor(.0237, device='cuda')
    summed = x.float() + residual.float()
    numerator = summed.bfloat16().float() if verify else summed
    expected = (numerator * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
                * (1 + w.float())).bfloat16()
    out, packed, retained = GemmaRMSNorm(w, 1e-6).fused_add_quant(
        x, residual, scale, inplace=inplace, verify=verify)
    assert torch.equal(retained, summed.bfloat16())
    assert (retained.data_ptr() == residual.data_ptr()) == inplace
    assert torch.equal(x, x_original)
    if not inplace:
        assert torch.equal(residual, original)
    # Independent torch reduction can differ at BF16 midpoints; using the wrong
    # numerator or rounded variance changes thousands of elements here.
    assert (out == expected).float().mean() > .9999
    torch.testing.assert_close(out.float(), expected.float(), rtol=.008, atol=1e-6)
    assert torch.equal(packed.view(torch.uint8), quant_fp8(out, scale).view(torch.uint8))


@torch.inference_mode()
def test_verify_norm_graph_reads_new_inputs_and_preserves_ownership():
    from hibana.model.norm import GemmaRMSNorm
    torch.manual_seed(22)
    x = torch.randn(16, 5120, device='cuda', dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(5120, device='cuda', dtype=torch.bfloat16) * .1
    norm = GemmaRMSNorm(weight, 1e-6)
    for _ in range(2):
        norm.fused_add(x, residual, verify=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, retained = norm.fused_add(x, residual, verify=True)
    for seed in [27, 28]:
        torch.manual_seed(seed)
        x.normal_(); residual.normal_()
        before = residual.clone()
        expected, summed = norm.fused_add(x, residual, verify=True)
        graph.replay()
        assert torch.equal(out, expected) and torch.equal(retained, summed)
        assert torch.equal(residual, before)


@pytest.mark.parametrize('tokens', [2, 16, 64])
@torch.inference_mode()
def test_compiled_gdn_norm_scalar_precision_and_graph(tokens):
    from hibana.kernels.verify_norm import verify_gated_norm
    torch.manual_seed(123)
    x = torch.randn(tokens, 48, 128, device='cuda', dtype=torch.bfloat16)
    z = torch.randn_like(x)
    weight = torch.randn(128, device='cuda', dtype=torch.bfloat16)
    def reference():
        xf, zf = x.float(), z.float()
        variance = xf.square().mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance.double() + 1e-6)
        gate = zf * torch.sigmoid(zf)
        return (xf * rstd * weight.float() * gate).bfloat16()
    for _ in range(2):
        verify_gated_norm(x, z, weight, 1e-6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = verify_gated_norm(x, z, weight, 1e-6)
    for seed in [15, 16]:
        torch.manual_seed(seed); x.normal_(); z.normal_()
        expected = reference()
        graph.replay()
        assert (out == expected).float().mean() > .9999
        torch.testing.assert_close(out.float(), expected.float(), rtol=.008, atol=1e-6)
