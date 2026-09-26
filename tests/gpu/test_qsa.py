"""Flash-Next sparse attention on the GPU: the vendored kernel against a torch
computation, and the engine indexer (chunked prefill, then decode, on scrambled
pages) against the reference selection `qsa_token_mask`."""
from types import SimpleNamespace

import pytest
import torch



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
    from sharar.cache.kv_pool import KvPool
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
    token_of = {page: j for j, page in enumerate(lookup)}
    ties = 0
    for p, chosen in selections.items():
        want = {lookup[j] for j in mask[p].nonzero().flatten().tolist()}
        if chosen == want:
            continue
        # Only a tie at the top-k boundary may differ: the kernel sums in another order than the
        # reference, so groups whose scores agree to rounding can swap places.
        visible = (p + 1) // ratio
        scores = torch.relu(torch.einsum("hd,gd->gh", q[p].float(), keys[:visible].float())).sum(-1) / 128 ** 0.5
        kth = scores.topk(budget // ratio).values[-1]
        swapped = {token_of[page] // ratio for page in chosen ^ want}
        assert all(abs(scores[g] - kth) <= 1e-5 * max(1.0, abs(kth.item())) for g in swapped), f"row {p}: not a boundary tie"
        ties += 1
    assert ties <= max(1, len(selections) // 100), f"{ties} of {len(selections)} rows differ at the boundary"


def test_graph_safe_maintenance_matches_the_eager_keys_under_replay():
    """`maintain` (captured once, replayed with fresh per-step rows) writes the same raw and
    compressed keys as the eager path; rows that complete no group only touch page 0."""
    from sharar.cache.kv_pool import KvPool
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


def _torch_select(q, aux, first, n, seq_pages, ratio, budget):
    """The indexer's former torch selection (scores every visible group, top budget // ratio,
    the pending tail, valid entries first): the oracle for `kernels/qsa_select.py`."""
    r, d, device = ratio, q.shape[-1], q.device
    width = budget + r - 1
    positions = torch.arange(first, first + n, device=device)
    visible = (positions + 1) // r
    groups = (first + n) // r
    top = min(budget // r, groups)
    rows = torch.full((n, width), -1, dtype=torch.int64, device=device)
    if top:
        keys = aux[seq_pages[r - 1: groups * r: r].long(), 1].float()
        logits = sum(torch.relu(q[:, h].float() @ keys.T) for h in range(q.shape[1])) / d ** 0.5
        logits.masked_fill_(torch.arange(groups, device=device)[None, :] >= visible[:, None], -float("inf"))
        values, idx = logits.topk(top, dim=-1)
        chosen = torch.where(torch.isfinite(values), idx, torch.full_like(idx, -1))
        members = chosen.unsqueeze(-1) * r + torch.arange(r, device=device)
        rows[:, : top * r] = torch.where(chosen.unsqueeze(-1) >= 0, members, torch.full_like(members, -1)).reshape(n, top * r)
    tail = (visible * r).unsqueeze(1) + torch.arange(r - 1, device=device)
    rows[:, width - (r - 1):] = torch.where(tail <= positions.unsqueeze(1), tail, torch.full_like(tail, -1))
    rows = rows.gather(1, torch.argsort((rows < 0).to(torch.int8), dim=1, stable=True))
    return torch.where(rows >= 0, seq_pages.long()[rows.clamp_min(0)], torch.full_like(rows, -1)).to(torch.int32)


@pytest.mark.parametrize("budget,length,rows", [(2048, 3000, 1), (2048, 70000, 1), (2048, 70000, 4), (2048, 9000, 256), (64, 600, 40), (2048, 1500, 3)])
def test_selection_kernel_matches_the_torch_selection(budget, length, rows):
    from sharar.kernels.qsa_select import select_pages
    g = torch.Generator(device="cuda").manual_seed(2)
    ratio, dev = 4, torch.device("cuda")
    pages = length + 500
    aux = torch.randn(pages, 2, 128, generator=g, device=dev).to(torch.bfloat16)
    seq_pages = torch.randperm(pages, generator=g, device=dev)[:length].to(torch.int32)
    table = torch.cat([torch.zeros(7, dtype=torch.int32, device=dev), seq_pages])  # a nonzero base offset
    q = torch.randn(rows, 4, 128, generator=g, device=dev).to(torch.bfloat16)
    first = length - rows
    want = _torch_select(q, aux, first, rows, seq_pages, ratio, budget)
    base = torch.full((rows,), 7, dtype=torch.int32, device=dev)
    gcap = length // ratio + 1000  # wider than the sequence: padding groups must never be chosen
    got, counts = select_pages(q, aux, table, base, torch.arange(first, length, device=dev), ratio, budget, gcap, 128 ** -0.5)
    assert torch.equal(counts, (want >= 0).sum(1, dtype=torch.int32))
    for i in range(rows):
        assert torch.equal(got[i].sort().values, want[i].sort().values), f"row {i}"


def test_selecting_graph_matches_the_eager_selection():
    """`maintain` + `select` captured once for a batch of two decode rows past the budget and
    replayed per step with fresh static inputs selects what the eager path selects."""
    from sharar.cache.kv_pool import KvPool
    from sharar.model.flashnext.indexer import FlashNextIndexer
    from sharar.target import FLASHNEXT
    dev, hidden, ratio, budget = torch.device("cuda"), 256, 4, 64
    lengths, steps, cap = (400, 700), 40, 1024
    ix = _indexer(hidden, budget, ratio, dev)
    eager_pool, graph_pool = KvPool(4096, dev, target=FLASHNEXT), KvPool(4096, dev, target=FLASHNEXT)
    ix.register(eager_pool)
    ix.register(graph_pool)
    perm = (torch.randperm(4000, device=dev) + 1).to(torch.int32)
    seqs_pages = [perm[:cap], perm[cap:2 * cap]]
    xs = [torch.randn(cap, hidden, device=dev).to(torch.bfloat16) for _ in lengths]
    for pool in (eager_pool, graph_pool):  # the same prefill in both pools
        for pages, x, n in zip(seqs_pages, xs, lengths):
            ctx = SimpleNamespace(kv_pool=pool, positions=torch.arange(n, device=dev), qsa_seqs=[(0, n, pages[:n])])
            ix.run(x[:n], ctx, pages[:n])
    b = len(lengths)
    st = {"x": torch.zeros(b, hidden, dtype=torch.bfloat16, device=dev), "pos": torch.zeros(b, dtype=torch.int64, device=dev),
          "page": torch.zeros(b, dtype=torch.int32, device=dev), "members": torch.zeros(b, ratio, dtype=torch.int32, device=dev),
          "writes": torch.zeros(b, dtype=torch.int32, device=dev), "table": torch.zeros(b * cap, dtype=torch.int32, device=dev),
          "indptr": torch.arange(0, (b + 1) * cap, cap, dtype=torch.int32, device=dev), "row_seq": torch.arange(b, device=dev)}
    gctx = SimpleNamespace(kv_pool=graph_pool, positions=st["pos"], qsa_members=st["members"], qsa_writes=st["writes"])
    aux = graph_pool.page_aux[ix.name]

    def body():
        qk = ix.maintain(st["x"], gctx, st["page"])
        return ix.select(qk, st["pos"], aux, st["table"], st["indptr"][st["row_seq"]], cap // ratio)

    body()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_pages, out_counts = body()
    for step in range(steps):
        seqs = [(n + step, 1, pages[: n + step + 1]) for n, pages in zip(lengths, seqs_pages)]
        x = torch.cat([xs[i][n + step: n + step + 1] for i, n in enumerate(lengths)])
        want_pages, want_counts = ix.run(x, SimpleNamespace(kv_pool=eager_pool, positions=torch.tensor([s[0] for s in seqs], device=dev),
                                                            qsa_seqs=seqs), torch.cat([p[s[0]:s[0] + 1] for s, p in zip(seqs, seqs_pages)]))
        members, writes = FlashNextIndexer.static_rows(seqs, ratio, dev)
        st["x"].copy_(x); st["pos"].copy_(torch.tensor([s[0] for s in seqs])); st["members"].copy_(members); st["writes"].copy_(writes)
        st["page"].copy_(torch.cat([p[s[0]:s[0] + 1] for s, p in zip(seqs, seqs_pages)]))
        st["table"].copy_(torch.cat(seqs_pages))
        graph.replay()
        assert torch.equal(out_counts, want_counts), f"step {step}"
        for i in range(b):
            assert torch.equal(out_pages[i].sort().values, want_pages[i].sort().values), f"step {step} row {i}"
