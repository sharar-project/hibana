import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GB10")



def _close_fp8(a, b):
    a = a.float()
    b = b.float()
    exact = (a == b).float().mean().item()
    assert exact > 0.995, exact
    assert torch.allclose(a, b, rtol=0.13, atol=0.5), (a - b).abs().max()


@cuda
def test_quant_fp8_and_gate():
    from sharar.kernels.fp8_quant import gated_quant_fp8, quant_fp8

    torch.manual_seed(0)
    x = torch.randn(5, 24 * 256, device="cuda", dtype=torch.bfloat16)
    qkv = torch.randn(5, 24, 512, device="cuda", dtype=torch.bfloat16)
    gate = qkv[..., 256:]
    scale = torch.tensor(0.02, device="cuda")
    ref = (x.float() / 0.02).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(quant_fp8(x, scale).float(), ref.float())
    gated = (x.view(5, 24, 256).float() * torch.sigmoid(gate.float())).to(torch.bfloat16).reshape(5, -1)
    ref_g = (gated.float() / 0.02).clamp(-448, 448).to(torch.float8_e4m3fn)
    _close_fp8(gated_quant_fp8(x.view(5, 24, 256), gate, scale), ref_g)



@cuda
def test_qk_norm_rope_matches_torch():
    from sharar.kernels.qk_norm_rope import qk_norm_rope
    from sharar.model.norm import GemmaRMSNorm
    from sharar.model.rope import PartialRotary

    torch.manual_seed(0)
    t, hq, hk, d, r = 7, 24, 4, 256, 64
    qkv = torch.randn(t, hq * 2 * d + 2 * hk * d, device="cuda", dtype=torch.bfloat16)
    qw = torch.randn(d, device="cuda", dtype=torch.bfloat16) * 0.3
    kw = torch.randn(d, device="cuda", dtype=torch.bfloat16) * 0.3
    rope = PartialRotary(d, r, 1e7, 4096, torch.device("cuda"))
    pos = torch.arange(100, 100 + t, device="cuda")
    q, k = qk_norm_rope(qkv, pos, rope.cos_sin, qw, kw, hq, hk, d, r, 1e-6)
    qn, kn = GemmaRMSNorm(qw, 1e-6), GemmaRMSNorm(kw, 1e-6)
    q_gate, k_ref, _ = qkv.split([hq * 2 * d, hk * d, hk * d], dim=-1)
    q_ref = q_gate.view(t, hq, 2 * d)[..., :d]
    q_ref = rope(pos, qn.torch_forward(q_ref.reshape(-1, d)).view(t, hq, d))
    k_ref = rope(pos, kn.torch_forward(k_ref.reshape(-1, d)).view(t, hk, d))
    assert torch.allclose(q.float(), q_ref.float(), rtol=2e-2, atol=2e-2), (q.float() - q_ref.float()).abs().max()
    assert torch.allclose(k.float(), k_ref.float(), rtol=2e-2, atol=2e-2), (k.float() - k_ref.float()).abs().max()


@cuda
def test_nvfp4_quantize_matches_flashinfer():
    from sharar.kernels import ops
    from sharar.kernels.nvfp4_quant import nvfp4_quantize
    from sharar.weights.modelopt import swizzle_block_scales

    torch.manual_seed(0)
    for m, k in ((2048, 5120), (37, 17408), (1, 5120)):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.05
        gs = torch.tensor(700.0, device="cuda")
        q_ref, sf_ref = ops.fp4_quantize(x, gs)
        q, sf = nvfp4_quantize(x, gs)
        assert sf.shape == sf_ref.shape, (sf.shape, sf_ref.shape)
        # Cached padding can contain earlier rows; GEMM never consumes it.
        valid = swizzle_block_scales(torch.ones(m, k//16, dtype=torch.uint8, device=x.device)).bool()
        assert torch.equal(sf.view(torch.uint8)[valid], sf_ref.view(torch.uint8)[valid])
        assert torch.equal(q, q_ref)


@cuda
def test_silu_mul_nvfp4_quantize_matches_two_step():
    from sharar.kernels.nvfp4_quant import nvfp4_quantize, silu_mul_nvfp4_quantize

    torch.manual_seed(0)
    x = torch.randn(64, 2 * 17408, device="cuda", dtype=torch.bfloat16)
    gs = torch.tensor(400.0, device="cuda")
    g, u = x[:, :17408].float(), x[:, 17408:].float()
    h = (torch.nn.functional.silu(g) * u).to(torch.bfloat16)
    q_ref, sf_ref = nvfp4_quantize(h, gs)
    # The quantizer reuses one scale buffer per shape. Freeze the reference
    # before the fused call overwrites that buffer.
    sf_ref = sf_ref.clone()
    q, sf = silu_mul_nvfp4_quantize(x, gs)
    assert torch.equal(sf.view(torch.uint8), sf_ref.view(torch.uint8))
    assert torch.equal(q, q_ref)
