"""Flash-Next sparse attention on the GPU: the vendored kernel against a torch
computation, and the engine indexer (chunked prefill, then decode, on scrambled
pages) against the reference selection `qsa_token_mask`."""
import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("SHARAR_RUN_GPU") != "1" or not torch.cuda.is_available(), reason="GB10 only")


def test_sparse_attention_matches_torch():
    from sharar.kernels.qsa import sparse_attention
    g = torch.Generator(device="cuda").manual_seed(0)
    pages, heads, kv_heads, dim, rows, width = 3000, 24, 2, 256, 40, 2051
    k_pool = torch.randn(pages, kv_heads, dim, generator=g, device="cuda").to(torch.float8_e4m3fn)
    v_pool = torch.randn(pages, kv_heads, dim, generator=g, device="cuda").to(torch.float8_e4m3fn)
    q = torch.randn(rows, heads, dim, generator=g, device="cuda").to(torch.bfloat16)
    counts = torch.randint(1, width + 1, (rows,), generator=g, device="cuda").to(torch.int32)
    sel = torch.full((rows, width), -1, dtype=torch.int32, device="cuda")
    for r in range(rows):
        sel[r, : counts[r]] = torch.randperm(pages, generator=g, device="cuda")[: counts[r]].to(torch.int32)
    out = sparse_attention(q, k_pool, v_pool, sel, counts, dim ** -0.5)
    for r in range(rows):
        idx = sel[r, : counts[r]].long()
        k = k_pool[idx].float().repeat_interleave(heads // kv_heads, dim=1)  # [n, heads, dim]
        v = v_pool[idx].float().repeat_interleave(heads // kv_heads, dim=1)
        p = torch.softmax(torch.einsum("hd,nhd->hn", q[r].float(), k) * dim ** -0.5, dim=-1)
        want = torch.einsum("hn,nhd->hd", p, v)
        torch.testing.assert_close(out[r].float(), want, atol=2e-2, rtol=2e-2)


def _indexer(hidden, budget, ratio, device):
    from sharar.model.flashnext.indexer import FlashNextIndexer
    g = torch.Generator(device=device).manual_seed(1)
    ix = FlashNextIndexer.__new__(FlashNextIndexer)
    ix.name, ix.heads, ix.dim, ix.ratio, ix.budget = "qsa.0", 4, 128, ratio, budget
    ix.width, ix.eps, ix.rotary_dim, ix.theta = budget + ratio - 1, 1e-6, 64, 1e7
    ix.qk = (torch.randn(5 * 128, hidden, generator=g, device=device) / hidden ** 0.5).to(torch.bfloat16)
    ix.q_norm = torch.randn(128, generator=g, device=device) * 0.1
    ix.k_norm = torch.randn(128, generator=g, device=device) * 0.1
    return ix


@pytest.mark.parametrize("budget", [64, 2048])
def test_indexer_selection_matches_the_reference(budget):
    from sharar.memory.kv_pool import KvPool
    from sharar.model.flashnext.reference import gemma_rms, qsa_token_mask, rope_neox
    from sharar.target import FLASHNEXT
    dev, hidden, ratio = torch.device("cuda"), 256, 4
    total = 3 * budget + 37  # sparse rows exist; a chunk boundary inside a group
    ix = _indexer(hidden, budget, ratio, dev)
    pool = KvPool(total + 64, dev, target=FLASHNEXT)
    ix.register(pool)
    seq_pages = (torch.randperm(total + 63, device=dev) + 1)[:total].to(torch.int32)  # scrambled pages, never page 0
    x = (torch.randn(total, hidden, device=dev)).to(torch.bfloat16)
    selections = {}
    projected = []  # the engine's own index projections: bf16 GEMMs change with the row count

    def forward(first, n):
        projected.append(torch.nn.functional.linear(x[first:first + n], ix.qk))
        ctx = SimpleNamespace(kv_pool=pool, positions=torch.arange(first, first + n, device=dev),
                              qsa_seqs=[(first, n, seq_pages[: first + n])])
        out = ix.run(x[first:first + n], ctx, seq_pages[first:first + n])
        if out is not None:
            pages, counts = out
            for i in range(n):
                selections[first + i] = set(pages[i, : counts[i]].tolist())
                assert (pages[i, counts[i]:] == -1).all()

    split = 2 * budget + 5  # prefill in two chunks, then one-token decode steps
    forward(0, split // 2)
    forward(split // 2, split - split // 2)
    for p in range(split, total):
        forward(p, 1)
    # the reference selection on the whole sequence, from the same projected rows
    qk = torch.cat(projected)
    q = rope_neox(gemma_rms(qk[:, :512].reshape(total, 4, 128), ix.q_norm, ix.eps), torch.arange(total, device=dev), 64, ix.theta)
    groups = total // ratio
    keys = qk[: groups * ratio, 512:].reshape(groups, ratio, 128).float().mean(1).to(torch.bfloat16)
    keys = rope_neox(gemma_rms(keys.unsqueeze(1), ix.k_norm, ix.eps), torch.arange(0, groups * ratio, ratio, device=dev), 64, ix.theta)[:, 0]
    mask = qsa_token_mask(q, keys, budget, ratio)
    assert selections, "no row past the budget"
    lookup = seq_pages.tolist()
    for p, chosen in selections.items():
        want = {lookup[j] for j in mask[p].nonzero().flatten().tolist()}
        assert chosen == want, f"row {p}: {len(chosen ^ want)} pages differ"


def test_graph_safe_maintenance_matches_the_eager_keys_under_replay():
    """`maintain` (captured once, replayed with fresh per-step rows) writes the same raw and
    compressed keys as the eager path; rows that complete no group only touch page 0."""
    from sharar.memory.kv_pool import KvPool
    from sharar.model.flashnext.indexer import FlashNextIndexer
    from sharar.target import FLASHNEXT
    dev, hidden, ratio, total = torch.device("cuda"), 256, 4, 300
    ix = _indexer(hidden, 2048, ratio, dev)
    eager_pool, graph_pool = KvPool(total + 64, dev, target=FLASHNEXT), KvPool(total + 64, dev, target=FLASHNEXT)
    ix.register(eager_pool)
    ix.register(graph_pool)
    seq_pages = (torch.randperm(total + 63, device=dev) + 1)[:total].to(torch.int32)
    x = torch.randn(total, hidden, device=dev).to(torch.bfloat16)
    prefill = 100
    ctx = SimpleNamespace(kv_pool=eager_pool, positions=torch.arange(prefill, device=dev), qsa_seqs=[(0, prefill, seq_pages[:prefill])])
    assert ix.run(x[:prefill], ctx, seq_pages[:prefill]) is None
    graph_pool.page_aux[ix.name].copy_(eager_pool.page_aux[ix.name])  # the same prefix in both pools
    # graph: one decode row captured, replayed per token with new static rows
    static = {"x": torch.zeros(1, hidden, dtype=torch.bfloat16, device=dev), "pos": torch.zeros(1, dtype=torch.int64, device=dev),
              "page": torch.zeros(1, dtype=torch.int32, device=dev), "members": torch.zeros(1, ratio, dtype=torch.int32, device=dev),
              "writes": torch.zeros(1, dtype=torch.int32, device=dev)}
    gctx = SimpleNamespace(kv_pool=graph_pool, positions=static["pos"], qsa_members=static["members"], qsa_writes=static["writes"])
    ix.maintain(static["x"], gctx, static["page"])  # warm up
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ix.maintain(static["x"], gctx, static["page"])
    for p in range(prefill, total):
        seq = [(p, 1, seq_pages[: p + 1])]
        ix.run(x[p:p + 1], SimpleNamespace(kv_pool=eager_pool, positions=torch.arange(p, p + 1, device=dev), qsa_seqs=seq), seq_pages[p:p + 1])
        members, writes = FlashNextIndexer.static_rows(seq, ratio, dev)
        static["x"].copy_(x[p:p + 1]); static["pos"].fill_(p); static["page"].copy_(seq_pages[p:p + 1])
        static["members"].copy_(members); static["writes"].copy_(writes)
        graph.replay()
    used = seq_pages.long()
    eager, replayed = eager_pool.page_aux[ix.name][used], graph_pool.page_aux[ix.name][used]
    assert torch.equal(eager[:, 0], replayed[:, 0])  # raw keys
    complete = torch.arange(total, device=dev) % ratio == ratio - 1
    assert torch.equal(eager[complete, 1], replayed[complete, 1])  # compressed keys at every group end
