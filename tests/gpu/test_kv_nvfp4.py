"""The NVFP4 KV pool (SHARAR_KV_FORMAT=nvfp4): FlashInfer's writer is the screen's rounding, and fa2 attention over
the pool's codes and scales is attention over the rounded values."""
import pytest
import torch


def _pools(pages, heads=4, dim=256):
    from hibana.cache.kv_pool import KvPool
    from hibana.target import DEFAULT_TARGET
    layers = [3]
    fp4 = KvPool(pages, torch.device("cuda"), layer_ids=layers, target=DEFAULT_TARGET, padding_rows=4, nvfp4=True)
    ref = KvPool(pages, torch.device("cuda"), layer_ids=layers, target=DEFAULT_TARGET, padding_rows=4, dtype=torch.bfloat16)
    return fp4, ref


def _dequant(codes, scales):
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=codes.device)
    vals = torch.stack([lut[(codes & 15).long()], lut[(codes >> 4).long()]], -1).reshape(*codes.shape[:-1], -1)
    return (vals.view(*vals.shape[:-1], -1, 16) * scales.float()[..., None]).reshape(vals.shape)


def test_writes_are_the_screens_rounding():
    from hibana.kernels.kv_fp4 import nvfp4_round
    torch.manual_seed(1)
    fp4, _ = _pools(64)
    k = torch.randn(40, 4, 256, device="cuda", dtype=torch.bfloat16) * 3
    v = torch.randn(40, 4, 256, device="cuda", dtype=torch.bfloat16)
    pages = torch.randperm(60, device="cuda")[:40].int() + 1
    fp4.write(3, pages, k, v)
    ks, vs = fp4.scales(3)
    assert torch.equal(_dequant(fp4.k[3][pages.long(), 0], ks[pages.long(), 0]).bfloat16(), nvfp4_round(k))
    assert torch.equal(_dequant(fp4.v[3][pages.long(), 0], vs[pages.long(), 0]).bfloat16(), nvfp4_round(v))
    assert fp4.row_bytes == 2 * (4 * 128 + 4 * 16)  # one layer: K and V codes and scales, 4.5 bits a value


@pytest.mark.parametrize("graph", [False, True])
def test_attention_over_the_pool_is_attention_over_the_rounded_values(graph):
    import flashinfer
    from hibana.kernels.kv_fp4 import nvfp4_round
    torch.manual_seed(2)
    ctx, rows = [900, 300], [16, 5]
    total = sum(c + r for c, r in zip(ctx, rows))
    fp4, ref = _pools(total + 8)
    k = torch.randn(total, 4, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(total, 4, 256, device="cuda", dtype=torch.bfloat16)
    pages = torch.arange(1, total + 1, dtype=torch.int32, device="cuda")
    fp4.write(3, pages, k, v)
    ref.write(3, pages, nvfp4_round(k), nvfp4_round(v))
    qo = torch.tensor([0, rows[0], sum(rows)], dtype=torch.int32)
    ind = torch.tensor([0, ctx[0] + rows[0], total], dtype=torch.int32)
    last = torch.ones(2, dtype=torch.int32)
    q = torch.randn(sum(rows), 24, 256, device="cuda", dtype=torch.bfloat16)
    outs = []
    for pool in (fp4, ref):
        ws = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
        kw = {}
        if graph:
            kw = dict(use_cuda_graph=True, qo_indptr_buf=qo.cuda(), paged_kv_indptr_buf=ind.cuda(),
                      paged_kv_indices_buf=torch.zeros(total + 8, dtype=torch.int32, device="cuda"), paged_kv_last_page_len_buf=last.cuda())
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", **kw)
        w.plan(qo, ind, pages, last, 24, 4, 256, 1, causal=True, q_data_type=torch.bfloat16, kv_data_type=pool.dtype, sm_scale=256 ** -0.5)
        out = torch.empty_like(q)
        run = lambda: w.run(q, pool.cache(3), kv_cache_sf=pool.scales(3), out=out)  # noqa: E731
        if graph:
            run()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run()
            out.zero_()
            g.replay()
        else:
            run()
        outs.append(out.float())
    torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=8e-3)
