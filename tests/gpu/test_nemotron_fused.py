"""Nemotron's fused decode/verify kernels equal the chains they replace, bit for bit: the residual norm (FlashInfer's
fused add + RMS norm on clones, then quant_fp8), the K/V page write (contiguous copies, casts, index_put), and
FlashInfer's attention reading q as a strided view of the q|k|v rows."""

import pytest
import torch

H = 4096
ROWS = [1, 2, 4, 5, 16, 20, 64, 300]


def _inputs(rows, seed):
    g = torch.Generator("cuda").manual_seed(seed)
    x = torch.randn(rows, H, device="cuda", dtype=torch.bfloat16, generator=g) * 3
    residual = torch.randn(rows, H, device="cuda", dtype=torch.bfloat16, generator=g) * 20
    weight = torch.randn(H, device="cuda", dtype=torch.bfloat16, generator=g) * 0.3
    return x, residual, weight


@pytest.mark.parametrize("rows", ROWS)
@torch.inference_mode()
def test_residual_norm(rows):
    from flashinfer.norm import fused_add_rmsnorm
    from sharar.kernels.cute.nemotron_norm import residual_norm
    from sharar.kernels.fp8_quant import quant_fp8
    x, residual, weight = _inputs(rows, rows)
    want, want_res = x.clone(), residual.clone()
    fused_add_rmsnorm(want, want_res, weight, 1e-5)
    for scale in (None, 0.004, 0.05, 1.0):  # 0.004 clamps many values at 448
        s = None if scale is None else torch.tensor([scale], device="cuda")
        # separate outputs: the inputs stay as they were
        x0, r0 = x.clone(), residual.clone()
        out, res = torch.empty_like(x), torch.empty_like(residual)
        q = residual_norm(x0, r0, weight, 1e-5, out, res, s)
        assert torch.equal(x0, x) and torch.equal(r0, residual), (rows, scale)
        assert torch.equal(out, want) and torch.equal(res, want_res), (rows, scale)
        if s is not None:
            assert torch.equal(q.view(torch.uint8), quant_fp8(want, s).view(torch.uint8)), (rows, scale)
        else:
            assert q is None
        # in place
        x1, r1 = x.clone(), residual.clone()
        residual_norm(x1, r1, weight, 1e-5, x1, r1, s)
        assert torch.equal(x1, want) and torch.equal(r1, want_res), (rows, scale)


@pytest.mark.parametrize("inplace,consume", [(False, False), (True, False), (True, True)])
@torch.inference_mode()
def test_rmsnorm_module(inplace, consume, monkeypatch):
    """`RMSNorm.fused_add(_quant)` on the fused kernel against the FlashInfer chain (the path of the shapes the fused
    kernel does not take)."""
    from sharar.kernels.cute import nemotron_norm
    from sharar.model.nemotron import norm
    x, residual, weight = _inputs(5, 7)
    n = norm.RMSNorm(weight, 1e-5)
    s = torch.tensor([0.05], device="cuda")
    results = {}
    for fused in (False, True):
        monkeypatch.setattr(norm, "_fused_ok", nemotron_norm.supported if fused else lambda x, residual: False)
        xa, ra = x.clone(), residual.clone()
        out, q, res = n.fused_add_quant(xa, ra, s, inplace=inplace, consume_input=consume)
        xb, rb = x.clone(), residual.clone()
        out2, res2 = n.fused_add(xb, rb, inplace=inplace, consume_input=consume)
        results[fused] = (out.clone(), q.view(torch.uint8).clone(), res.clone(), out2.clone(), res2.clone(), xa, ra, xb, rb)
    for a, b in zip(results[False], results[True]):
        assert torch.equal(a, b)


@pytest.mark.parametrize("rows", [1, 4, 16, 20])
@torch.inference_mode()
def test_write_kv(rows):
    from sharar.cache.kv_pool import KvPool
    from sharar.kernels.nemotron_fused import write_kv
    from sharar.target import NEMOTRON
    q_size, kv = NEMOTRON.num_attention_heads * NEMOTRON.head_dim, NEMOTRON.num_kv_heads * NEMOTRON.head_dim
    g = torch.Generator("cuda").manual_seed(rows)
    qkv = torch.randn(rows, q_size + 2 * kv, device="cuda", dtype=torch.bfloat16, generator=g) * 40
    qkv[0, q_size:q_size + 8] = torch.tensor([447.0, 449.0, 455.0, 460.0, -447.0, 1e-6, -3e-3, 0.0], dtype=torch.bfloat16)
    for dtype in (torch.float8_e4m3fn, torch.bfloat16):
        a = KvPool(64, torch.device("cuda"), layer_ids=[3], dtype=dtype, target=NEMOTRON)
        b = KvPool(64, torch.device("cuda"), layer_ids=[3], dtype=dtype, target=NEMOTRON)
        pages = torch.randperm(64, device="cuda", generator=g)[:rows].to(torch.int32)
        t = rows
        k = qkv[:, q_size:q_size + kv].reshape(t, NEMOTRON.num_kv_heads, NEMOTRON.head_dim).contiguous()
        v = qkv[:, q_size + kv:].reshape(t, NEMOTRON.num_kv_heads, NEMOTRON.head_dim).contiguous()
        a.write(3, pages, k, v)
        write_kv(qkv, pages, *b.cache(3), q_size, kv)
        for x, y in zip(a.cache(3), b.cache(3)):
            assert torch.equal(x.view(torch.uint8), y.view(torch.uint8)), (rows, dtype)


@pytest.mark.parametrize("batch,steps", [(1, 1), (4, 1), (1, 4), (4, 4)])
@torch.inference_mode()
def test_strided_q_attention(batch, steps):
    """FlashInfer's decode (one row) and paged prefill (verify rows) read q by its strides: a view of the q|k|v rows
    gives the contiguous copy's bits."""
    import flashinfer
    from sharar.kernels.attention_plan import AttentionGeometry, AttentionPlanner
    from sharar.target import NEMOTRON
    heads, kvh, d = NEMOTRON.num_attention_heads, NEMOTRON.num_kv_heads, NEMOTRON.head_dim
    g = torch.Generator("cuda").manual_seed(batch * 10 + steps)
    pages_n = 2048
    kc = (torch.randn(pages_n, 1, kvh, d, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    vc = (torch.randn(pages_n, 1, kvh, d, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    lens = [300 + 97 * i for i in range(batch)]
    indices = [torch.randperm(pages_n, device="cuda", generator=g)[:n].to(torch.int32) for n in lens]
    indptr = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32)
    rows = batch * steps
    qkv = torch.randn(rows, (heads + 2 * kvh) * d, device="cuda", dtype=torch.bfloat16, generator=g)
    q_view = qkv[:, :heads * d].view(rows, heads, d)
    planner = AttentionPlanner(AttentionGeometry.of_target(NEMOTRON, 1, torch.float8_e4m3fn))
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    if steps == 1:
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", use_tensor_cores=True)
        planner.decode(w, indptr, torch.cat(indices), torch.ones(batch, dtype=torch.int32))
    else:
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
        planner.paged_prefill(w, torch.arange(0, rows + 1, steps, dtype=torch.int32), indptr, torch.cat(indices),
                              torch.ones(batch, dtype=torch.int32), causal=True, page_size=1)
    want = w.run(q_view.contiguous(), (kc, vc)).clone()
    got = w.run(q_view, (kc, vc))
    assert torch.equal(got, want)

