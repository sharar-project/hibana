"""Fused Flash-Next kernels against their torch definitions (sharar/model/flashnext math)."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
from sharar.kernels import flashnext_fused as fk  # noqa: E402

H, HC = 2560, 4


def _close(a, b, tol=2e-2):
    a, b = a.float(), b.float()
    assert a.shape == b.shape
    err = (a - b).abs().max().item()
    ref = b.abs().max().item() + 1e-6
    assert err <= tol * ref + 1e-3, (err, ref)


@pytest.mark.parametrize("t", [1, 4, 37])
def test_grouped_norm(t):
    x = torch.randn(t, HC * H, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(HC * H, device="cuda", dtype=torch.bfloat16) * 0.5
    xf = x.float().reshape(t, HC, H)
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).flatten(-2) * (1 + w.float())
    _close(fk.grouped_gemma_rms(x, w, 1e-6, H), ref.to(torch.bfloat16))


@pytest.mark.parametrize("t", [1, 4])
def test_gate_mean_and_inject(t):
    u = torch.randn(t, HC * H, device="cuda", dtype=torch.bfloat16) * 3
    n = torch.randn(t, HC * H, device="cuda", dtype=torch.bfloat16)
    ref = (torch.sigmoid(u.float()).view(t, HC, H) * n.float().view(t, HC, H)).mean(-2)
    _close(fk.gate_mean(u, n, HC, H), ref.to(torch.bfloat16))
    hyper = torch.randn(t, HC * H, device="cuda", dtype=torch.bfloat16)
    block = torch.randn(t, H, device="cuda", dtype=torch.bfloat16)
    inj = torch.randn(t, HC, device="cuda", dtype=torch.bfloat16)
    ref = (hyper.float().view(t, HC, H) + block.float().unsqueeze(1) * 2 * torch.sigmoid(inj.float()).unsqueeze(-1)).flatten(-2)
    _close(fk.inject_combine(hyper, block, inj, HC, H), ref.to(torch.bfloat16))


@pytest.mark.parametrize("shape", [(1, 2560), (4, 6144), (3, 640)])
def test_quant_small_matches_draft_quant(shape):
    from sharar.kernels.draft_fp8 import quant_dynamic
    x = (torch.randn(shape, device="cuda") * 4).to(torch.bfloat16).contiguous()
    wscale = torch.tensor([[0.0123]], device="cuda")
    q, scale, alpha = fk.quant_fp8_small(x, wscale)
    q2, scale2 = quant_dynamic(x)
    assert torch.equal(q.view(torch.uint8), q2.view(torch.uint8)) and torch.allclose(scale, scale2)
    assert torch.allclose(alpha, (scale2 * wscale).reshape(1))


@pytest.mark.parametrize("m", [1, 4, 16])
def test_route_topk(m):
    logits = (torch.randn(m, 512, device="cuda") * 2).to(torch.bfloat16)
    ids, w = fk.route_topk(logits, 10)
    probs = torch.softmax(logits.float(), -1)
    rw, ri = probs.topk(10, -1)
    # ties among BF16 logits may be ordered differently: compare as sets and the weight mass
    assert torch.equal(ids.long().sort(-1).values, ri.sort(-1).values)
    _close(w.sort(-1, descending=True).values, (rw / rw.sum(-1, keepdim=True)).sort(-1, descending=True).values, 1e-4)


def test_moe_combine_silu_mul_sigmoid_gate():
    t = 4
    routed = torch.randn(t, H, device="cuda", dtype=torch.bfloat16)
    shared = torch.randn(t, H, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(t, H, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(H, device="cuda") * 0.01
    ref = routed.float() + torch.sigmoid(x.float() @ g) [:, None] * shared.float()
    _close(fk.moe_combine(routed, shared, x, g), ref.to(torch.bfloat16))
    gu = torch.randn(t, 1280, device="cuda", dtype=torch.bfloat16)
    _close(fk.silu_mul(gu), (F.silu(gu[:, :640].float()) * gu[:, 640:].float()).to(torch.bfloat16))
    qkv = torch.randn(t, 24 * 512 + 2 * 512, device="cuda", dtype=torch.bfloat16)
    gate = qkv[:, : 24 * 512].view(t, 24, 512)[..., 256:]
    attn = torch.randn(t, 24, 256, device="cuda", dtype=torch.bfloat16)
    ref = (attn.float() * torch.sigmoid(gate.float())).reshape(t, -1)
    _close(fk.sigmoid_gate(attn, gate), ref.to(torch.bfloat16))


@pytest.mark.parametrize("r", [1, 10, 40])
def test_expert_gemv(r):
    e, n, k = 16, 1280, 2560
    w = (torch.randn(e, n, k, device="cuda") * 0.02)
    scale = (w.abs().amax(dim=(1, 2)) / 448).clamp_min(1e-12)
    w8 = (w / scale[:, None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    x = torch.randn(r, k, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, e, (r,), device="cuda", dtype=torch.int32)
    ref = torch.stack([(w8[i].float() * scale[i]) @ x[j].float() for j, i in enumerate(idx.tolist())])
    _close(fk.expert_gemv(w8, scale, x, idx), ref.to(torch.bfloat16), 1e-2)


@pytest.mark.parametrize("shape", [(1, 320, 10240), (4, 10240, 320), (2, 2560, 640)])
def test_fp8_gemv_any(shape):
    m, n, k = shape
    w = (torch.randn(n, k, device="cuda") * 0.02)
    ws = (w.abs().amax() / 448).reshape(1, 1)
    w8 = (w / ws).to(torch.float8_e4m3fn)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    q, scale, alpha = fk.quant_fp8_small(x.contiguous(), ws)
    ref = (q.float() @ w8.float().T) * alpha
    _close(fk.fp8_gemv_any(q, w8, alpha), ref.to(torch.bfloat16), 1e-2)
