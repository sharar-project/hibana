"""Fused Flash-Next kernels against their torch definitions (sharar/model/flashnext math)."""
import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

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


# ---- draft-step fusions: bitwise against the unfused chains they replace ----

def _act_input(m, k, seed, outliers=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=g) * 3
    if outliers:
        x[:, :: max(1, k // 7)] *= 40  # a few loud channels set the per-tensor scale
    return x.to(torch.bfloat16).contiguous()


def _fp8_weight(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", generator=g) * 0.02
    ws = (w.abs().amax() / 448).reshape(1, 1)
    return (w / ws).to(torch.float8_e4m3fn), ws


def _bits(t):
    t = t.contiguous()
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


@pytest.mark.parametrize("m", [1, 2, 3, 4])
@pytest.mark.parametrize("n,k", [(13312, 2560), (2560, 6144), (2560, 2560), (1280, 2560), (2560, 8192)])
def test_fp8_gemv_quant_cuda_bitwise(m, n, k):
    from sharar.kernels.fp8_gemv_cuda import fp8_gemv_cuda, fp8_gemv_quant_cuda
    w8, ws = _fp8_weight(n, k, 1)
    for seed in range(3):
        x = _act_input(m, k, seed, outliers=seed != 0)
        q, scale, alpha = fk.quant_fp8_small(x, ws)
        ref = fp8_gemv_cuda(q, w8, alpha)
        got_scale = torch.empty(1, device="cuda")
        got = fp8_gemv_quant_cuda(x, w8, ws, got_scale)
        assert torch.equal(_bits(got), _bits(ref)) and torch.equal(got_scale, scale.reshape(1))


def test_fp8_gemv_quant_cuda_zero_and_tiny():
    from sharar.kernels.fp8_gemv_cuda import fp8_gemv_cuda, fp8_gemv_quant_cuda
    w8, ws = _fp8_weight(2560, 2560, 2)
    for x in (torch.zeros(2, 2560, device="cuda", dtype=torch.bfloat16), (torch.randn(2, 2560, device="cuda") * 1e-30).to(torch.bfloat16)):
        q, _, alpha = fk.quant_fp8_small(x, ws)
        assert torch.equal(_bits(fp8_gemv_quant_cuda(x, w8, ws)), _bits(fp8_gemv_cuda(q, w8, alpha)))


@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("n,k,act", [(320, 10240, fk.ACT_NONE), (10240, 320, fk.ACT_NONE), (10240, 320, fk.ACT_SILU), (2560, 640, fk.ACT_SILU_MUL),
                                     (2560, 640, fk.ACT_NONE)])
def test_fp8_gemv_quant_any_bitwise(m, n, k, act):
    w8, ws = _fp8_weight(n, k, 3)
    for seed in range(3):
        x = _act_input(m, 2 * k if act == fk.ACT_SILU_MUL else k, seed, outliers=seed != 0)
        a = F.silu(x) if act == fk.ACT_SILU else fk.silu_mul(x) if act == fk.ACT_SILU_MUL else x
        q, scale, alpha = fk.quant_fp8_small(a.contiguous(), ws)
        ref = fk.fp8_gemv_any(q, w8, alpha)
        got_scale = torch.empty(1, device="cuda")
        got = fk.fp8_gemv_quant_any(x, w8, ws, act, scale_out=got_scale)
        assert torch.equal(_bits(got), _bits(ref)) and torch.equal(got_scale, scale.reshape(1))


@triton.jit
def _silu_probe(X, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, mask=i < N, other=0.0).to(tl.float32)
    tl.store(Y + i, fk._silu_torch(x).to(tl.bfloat16), mask=i < N)


def test_silu_matches_torch_on_every_bf16():
    """The fused kernels' SiLU (ACT_SILU) equals F.silu on every BF16 value."""
    x = torch.arange(-32768, 32768, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16)
    y = torch.empty_like(x)
    _silu_probe[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), BLOCK=1024)
    ref = F.silu(x)
    finite = torch.isfinite(ref)
    assert torch.equal(_bits(y[finite]), _bits(ref[finite]))


@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("act", [fk.ACT_NONE, fk.ACT_SILU_MUL])
def test_expert_gemv_fused_bitwise(m, act):
    e, topk, h, inter = 32, 10, 2560, 640
    g = torch.Generator(device="cuda").manual_seed(5)
    n, k = (h, inter) if act == fk.ACT_SILU_MUL else (2 * inter, h)
    w = torch.randn(e, n, k, device="cuda", generator=g) * 0.02
    scale = (w.abs().amax(dim=(1, 2)) / 448).clamp_min(1e-12)
    w8 = (w / scale[:, None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    idx = torch.randint(0, e, (m * topk,), device="cuda", generator=g, dtype=torch.int32)
    if act == fk.ACT_SILU_MUL:
        x = _act_input(m * topk, 2 * inter, 6)
        ref = fk.expert_gemv(w8, scale, fk.silu_mul(x), idx)
        got = fk.expert_gemv_fused(w8, scale, x, idx, act=act)
    else:
        x = _act_input(m, h, 6)
        ref = fk.expert_gemv(w8, scale, x.repeat_interleave(topk, 0), idx)
        got = fk.expert_gemv_fused(w8, scale, x, idx, x_div=topk)
    assert torch.equal(_bits(got), _bits(ref))


@pytest.mark.parametrize("t", [1, 4])
def test_qkv_epilogue_bitwise(t):
    from sharar.kernels.qk_norm_rope import qk_norm_rope
    from sharar.model.rope import PartialRotary
    hq, hkv, d, rot = 24, 2, 256, 64
    rope = PartialRotary(d, rot, 10_000_000.0, 4096, torch.device("cuda"))
    qkv = _act_input(t, hq * 2 * d + 2 * hkv * d, 7)
    pos = torch.tensor([3, 700, 1500, 4000][:t], device="cuda")
    qw = torch.randn(d, device="cuda").to(torch.bfloat16) * 0.1
    kw = torch.randn(d, device="cuda").to(torch.bfloat16) * 0.1
    pages = torch.tensor([9, 2, 31, 17][:t], device="cuda", dtype=torch.int32)
    k_ref = torch.zeros(40, 1, hkv, d, device="cuda", dtype=torch.bfloat16)
    v_ref, k_got, v_got = torch.zeros_like(k_ref), torch.zeros_like(k_ref), torch.zeros_like(k_ref)
    q, k = qk_norm_rope(qkv, pos, rope.cos_sin, qw, kw, hq, hkv, d, rot, 1e-6)
    v = qkv[:, hq * 2 * d + hkv * d:].view(t, hkv, d).contiguous()
    k_ref[pages, 0] = k
    v_ref[pages, 0] = v
    q_got = fk.qkv_epilogue(qkv, pos, rope.cos_sin, qw, kw, k_got, v_got, pages, hq, hkv, d, rot, 1e-6)
    assert torch.equal(_bits(q_got), _bits(q)) and torch.equal(_bits(k_got), _bits(k_ref)) and torch.equal(_bits(v_got), _bits(v_ref))


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_moe_combine_routed_bitwise(m):
    """Bits over 4M+ outputs: a contracted product (an FMA) flips the BF16 rounding of only a few in a million."""
    topk, h = 10, 2560
    g = torch.Generator(device="cuda").manual_seed(8)
    for _ in range(max(1, 1600 // m)):
        y = (torch.randn(m * topk, h, device="cuda", generator=g) * 3).to(torch.bfloat16)
        w = torch.rand(m, topk, device="cuda", generator=g)
        w = w / w.sum(-1, keepdim=True)
        shared = torch.randn(m, h, device="cuda", generator=g).to(torch.bfloat16)
        x = torch.randn(m, h, device="cuda", generator=g).to(torch.bfloat16)
        gate = torch.randn(h, device="cuda", generator=g) * 0.01
        routed = (y.float() * w.reshape(-1)[:, None]).view(m, topk, -1).sum(1).to(torch.bfloat16)
        ref = fk.moe_combine(routed, shared, x, gate)
        assert torch.equal(_bits(fk.moe_combine_routed(y, w, shared, x, gate)), _bits(ref))


@pytest.mark.parametrize("t", [1, 4, 5, 20])
def test_grouped_norm_amax_and_parallel_quant_bitwise(t):
    x = _act_input(t, HC * H, 9)
    w = (torch.randn(HC * H, device="cuda") * 0.5).to(torch.bfloat16)
    ref = fk.grouped_gemma_rms(x, w, 1e-6, H)
    y, parts = fk.grouped_gemma_rms(x, w, 1e-6, H, amax=True)
    assert torch.equal(_bits(y), _bits(ref)) and parts.shape == (t * HC,)
    assert torch.equal(parts.max(), y.float().abs().max())
    ws = torch.tensor([[0.0123]], device="cuda")
    q, scale, _ = fk.quant_fp8_small(y, ws)
    q2, scale2 = fk.quant_fp8_parts(y, parts)
    assert torch.equal(q.view(torch.uint8), q2.view(torch.uint8)) and torch.equal(scale, scale2)


@pytest.mark.parametrize("t", [1, 5, 300, 4096])
@pytest.mark.parametrize("scale", [0.05, 1.0, 400.0])
def test_inject_combine_norm_bitwise(t, scale):
    """The combine and the next hyper-connection's norm in one pass: the two kernels' bits, maxima included."""
    hyper = (torch.randn(t, HC * H, device="cuda") * scale).to(torch.bfloat16)
    block = (torch.randn(t, H, device="cuda") * scale).to(torch.bfloat16)
    inj = torch.randn(t, HC, device="cuda").to(torch.bfloat16)
    w = (torch.randn(HC * H, device="cuda") * 0.5).to(torch.bfloat16)
    y_ref = fk.inject_combine(hyper, block, inj, HC, H)
    n_ref, parts_ref = fk.grouped_gemma_rms(y_ref, w, 1e-6, H, amax=True)
    y, n, parts = fk.inject_combine_norm(hyper, block, inj, HC, H, w, 1e-6)
    assert torch.equal(_bits(y), _bits(y_ref)) and torch.equal(_bits(n), _bits(n_ref)) and torch.equal(parts, parts_ref)


@pytest.mark.parametrize("rows", [1, 3, 4, 5, 16])
@pytest.mark.parametrize("shape,act", [((320, 10240), fk.ACT_NONE), ((10240, 320), fk.ACT_SILU), ((2560, 640), fk.ACT_SILU_MUL), ((13312, 2560), fk.ACT_NONE),
                                       ((2560, 2560), fk.ACT_SILU)])
def test_trunk_linear_fused_paths_bitwise(rows, shape, act):
    """TrunkLinear's fused FP8 calls equal the unfused chain they replaced: act, quantize, then GEMV or cuBLAS."""
    from sharar.kernels.fp8_gemv_cuda import fp8_gemv_cuda
    from sharar.model.flashnext.linear import TrunkLinear
    n, k = shape
    lin = TrunkLinear.from_bf16((torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16), "fp8")
    x = _act_input(rows, 2 * k if act == fk.ACT_SILU_MUL else k, 10)
    a = F.silu(x) if act == fk.ACT_SILU else fk.silu_mul(x) if act == fk.ACT_SILU_MUL else x
    a = a.contiguous()
    if a.numel() <= 65536:
        q, scale, alpha = fk.quant_fp8_small(a, lin.scale)
    else:
        from sharar.kernels.draft_fp8 import quant_dynamic
        q, scale = quant_dynamic(a)
        alpha = (scale * lin.scale).reshape(1)
    if rows <= 4 and k % 512 == 0 and k <= 8192:
        ref = fp8_gemv_cuda(q, lin.weight, alpha)
    elif rows <= 4:
        ref = fk.fp8_gemv_any(q, lin.weight, alpha)
    else:
        ref = torch._scaled_mm(q, lin.weight.T, scale_a=scale, scale_b=lin.scale, out_dtype=torch.bfloat16)
    assert torch.equal(_bits(lin(x, act=act)), _bits(ref))
    if act == fk.ACT_NONE and k == HC * H:  # with the producer's block maxima
        parts = a.float().abs().view(rows * HC, H).amax(-1)
        assert torch.equal(_bits(lin(x, amax_parts=parts)), _bits(ref))


@pytest.mark.parametrize("m", [1, 5, 20, 64])
@pytest.mark.parametrize("inter", [640, 320])
def test_silu_mul_quant_groups_bitwise(m, inter):
    from sharar.kernels import w3
    for seed in range(4):
        gu = _act_input(m, 2 * inter, seed, outliers=seed % 2 == 1)
        if seed == 3:
            gu[:, :64] = 0  # all-zero groups take the 1e-12 scale floor
            gu[:, 64:96] = (torch.randn(m, 32, device="cuda") * 1e-20).to(torch.bfloat16)
        q, s = w3.quant_groups(fk.silu_mul(gu))
        q2, s2 = fk.silu_mul_quant_groups(gu)
        assert torch.equal(q.view(torch.uint8), q2.view(torch.uint8)) and torch.equal(s, s2)
