"""kernels/csrc/verify_attn.cu on the 27B's shapes, against exact attention over the same NVFP4 KV values: sequences of
their own lengths and rows (more than 8 a sequence, the kernel's use), at FlashInfer's precision or better."""
import pytest
import torch

QH, KVH, D = 24, 4, 256


def _exact(q, k, v, kv, rows):
    """FP32 attention a sequence, its rows causal over the last pages; k, v [sum(kv), KVH, D] in sequence order."""
    out, at = [], 0
    for s, n in enumerate(kv):
        kk, vv = k[at:at + n].float(), v[at:at + n].float()
        at += n
        qq = q[s * rows:(s + 1) * rows].float().view(rows, KVH, QH // KVH, D)
        scores = torch.einsum("rkgd,nkd->rkgn", qq, kk) * D ** -0.5
        mask = torch.arange(n, device=q.device)[None, :] > (n - rows + torch.arange(rows, device=q.device))[:, None]
        scores.masked_fill_(mask[:, None, None, :], float("-inf"))
        out.append(torch.einsum("rkgn,nkd->rkgd", scores.softmax(-1), vv).reshape(rows, QH, D))
    return torch.cat(out)


@pytest.mark.parametrize("sharp", [1.0, 4.0])
@pytest.mark.parametrize("rows,ctx", [(16, [3000]), (9, [700, 5000, 64]), (16, [2048, 9000, 300, 17])])
def test_matches_exact(rows, ctx, sharp):
    import flashinfer
    from flashinfer.page import nvfp4_quantize_append_paged_kv_cache_with_slot_mapping as write_fp4
    from sharar.kernels.kv_fp4 import nvfp4_round
    from sharar.kernels.verify_attn_cuda import verify_attention
    torch.manual_seed(len(ctx) * 100 + rows)
    dev = "cuda"
    b = len(ctx)
    kv = [c + rows for c in ctx]
    pages = sum(kv) + 8
    kb = torch.randn(pages, KVH, D, device=dev, dtype=torch.bfloat16)
    vb = torch.randn_like(kb)
    k4 = torch.zeros(pages, 1, KVH, D // 2, dtype=torch.uint8, device=dev)
    v4 = torch.zeros_like(k4)
    ks = torch.zeros(pages, 1, KVH, D // 16, dtype=torch.float8_e4m3fn, device=dev)
    vs = torch.zeros_like(ks)
    one = torch.ones((), device=dev)
    write_fp4(kb, vb, torch.arange(pages, dtype=torch.int32, device=dev), (k4, v4), (ks, vs), one, one)
    qo = torch.arange(0, (b + 1) * rows, rows, dtype=torch.int32)
    ind = torch.tensor([0] + list(torch.tensor(kv).cumsum(0)), dtype=torch.int32)
    idx = (torch.randperm(pages - 1, device=dev)[:sum(kv)] + 1).int()
    q = (torch.randn(b * rows, QH, D, device=dev) * sharp).bfloat16()  # sharp: more peaked attention
    ref = _exact(q, nvfp4_round(kb)[idx.long()], nvfp4_round(vb)[idx.long()], kv, rows)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(torch.empty(128 << 20, dtype=torch.uint8, device=dev), "NHD")
    w.plan(qo, ind, idx, torch.ones(b, dtype=torch.int32), QH, KVH, D, 1, causal=True, q_data_type=torch.bfloat16,
           kv_data_type=torch.uint8, sm_scale=D ** -0.5)
    fa2 = (w.run(q, (k4, v4), kv_cache_sf=(ks, vs)).float() - ref).abs()
    for splits in (1, 5, 12):
        err = (verify_attention(q, (k4, v4), (ks, vs), ind.to(dev), idx, qo.to(dev), D ** -0.5, splits=splits).float() - ref).abs()
        assert err.mean() <= max(fa2.mean().item(), 2e-3 * ref.abs().mean().item()), (splits, err.mean().item(), fa2.mean().item())
        assert err.max() <= max(fa2.max().item(), 1e-2 * ref.abs().max().item()), (splits, err.max().item(), fa2.max().item())
