"""The eager SM121 kernels of the `original` QSA selection (`kernels/qsa/score_exact.py`) against `selection.py`:
scores with the same bits, then the same pages and counts, through long contexts, several sequences, ties and the
top-k's fallbacks; and the same attention output through the indexer."""
import pytest
import torch

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (12, 1), reason="SM121 kernels")
SCALE = 128 ** -0.5


def _inputs(contexts, rows, seed=0, key_scale=1.0):
    """Rows ending each sequence of `contexts`, on scrambled pages of one pool."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(contexts)
    aux = (torch.randn(total + 8, 2, 128, generator=g, device="cuda") * key_scale).to(torch.bfloat16)
    table = (torch.randperm(total + 8, generator=g, device="cuda")[:total]).to(torch.int32)
    offsets = torch.tensor([0] + list(contexts), device="cuda").cumsum(0)[:-1]
    positions = torch.cat([torch.arange(c - r, c, device="cuda") for c, r in zip(contexts, rows)])
    base = torch.repeat_interleave(offsets, torch.tensor(rows, device="cuda")).to(torch.int32)
    q = torch.randn(sum(rows), 4, 128, generator=g, device="cuda").to(torch.bfloat16)
    return q, aux, table, base, positions


@pytest.mark.parametrize("contexts,rows", [([32768 + 4099], [300]), ([65536, 40001], [200, 129]),
                                           ([131072], [256]), ([3000, 9001, 5], [64, 70, 5])])
def test_scores_have_the_same_bits(contexts, rows):
    from sharar.kernels.qsa.score_exact import _tiles, load_exact_scorer
    from sharar.kernels.qsa.selection import score_groups
    for key_scale in (1.0, 30.0):
        q, aux, table, base, positions = _inputs(contexts, rows, key_scale=key_scale)
        gcap = max(contexts) // 4
        want = score_groups(q, aux, table, base, positions, 4, gcap, SCALE)
        got = torch.full_like(want, float("nan"))
        tiles = torch.tensor(_tiles(rows, 0, sum(rows)), dtype=torch.int32, device="cuda").view(-1, 2)
        for per_block in (1, 8):
            load_exact_scorer().score(q, aux, table, base, positions, tiles, got, SCALE, per_block)
            assert torch.equal(got.view(torch.int32), want.view(torch.int32))


@pytest.mark.parametrize("contexts,rows", [([32768 + 4099], [300]), ([65536, 40001], [200, 129]),
                                           ([131072], [256]), ([3000, 9001, 5], [64, 70, 5])])
def test_selection_is_the_same(contexts, rows):
    from sharar.kernels.qsa.score_exact import select_exact
    from sharar.kernels.qsa.selection import select_pages
    q, aux, table, base, positions = _inputs(contexts, rows, seed=1)
    gcap = max(contexts) // 4
    want = select_pages(q, aux, table, base, positions, 4, 2048, gcap, SCALE, chunk=256)
    for chunk in (256, 100):
        got = select_exact(q, aux, table, base, positions, gcap, SCALE, rows, chunk)
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def _scores(kind, rows, cols, g):
    if kind == "random":
        return torch.randn(rows, cols, generator=g, device="cuda").relu()
    if kind == "ties":  # few distinct values: the 512th is tied, so only the first ties by group fit
        return torch.randint(0, 40, (rows, cols), generator=g, device="cuda").float() / 8
    if kind == "zeros":  # fewer than 512 positive scores: ties at 0
        s = torch.zeros(rows, cols, device="cuda")
        s[:, torch.randperm(cols, generator=g, device="cuda")[:300]] = 1.0 + torch.rand(rows, 300, generator=g, device="cuda")
        return s
    if kind == "crowded":  # every key in one 11-bit bin: resolved over the row instead of shared memory
        return 1.0 + torch.rand(rows, cols, generator=g, device="cuda") * 0.2
    if kind == "crowded_ties":
        return 1.0 + torch.randint(0, 3, (rows, cols), generator=g, device="cuda").float() / 64
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["random", "ties", "zeros", "crowded", "crowded_ties"])
@pytest.mark.parametrize("cols", [700, 8192, 32768])
def test_topk_matches_torch(kind, cols, monkeypatch):
    """The page assembly from the long-row top-512 against Torch's top-k and `_assemble_kernel`, on given scores (some
    rows partly invisible: -inf)."""
    from sharar.kernels.qsa import selection
    from sharar.kernels.qsa.score_exact import load_exact_scorer
    g = torch.Generator(device="cuda").manual_seed(cols)
    rows = 64
    scores = _scores(kind, rows, cols, g)
    visible = torch.randint(cols // 2, cols + 1, (rows,), generator=g, device="cuda")
    visible[:4] = torch.tensor([cols, 1, 511, 513], device="cuda")
    scores[torch.arange(cols, device="cuda")[None, :] >= visible[:, None]] = float("-inf")
    positions = visible * 4 + torch.randint(0, 4, (rows,), generator=g, device="cuda") - 1
    positions = positions.clamp_min(0)
    table = torch.randperm(cols * 4 + 8, generator=g, device="cuda").to(torch.int32)
    base = torch.zeros(rows, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(selection, "score_groups", lambda *args: scores)
    q = torch.empty(rows, 4, 128, device="cuda", dtype=torch.bfloat16)
    want = selection.select_pages(q, None, table, base, positions, 4, 2048, cols, SCALE)
    pages = torch.empty(rows, 2051, dtype=torch.int32, device="cuda")
    counts = torch.empty(rows, dtype=torch.int32, device="cuda")
    load_exact_scorer().topk_pages_long(scores, table, base.long(), positions.long(), pages, counts)
    assert torch.equal(counts, want[1]) and torch.equal(pages, want[0])


@pytest.mark.parametrize("prefix,length", [(40960, 1024), (2048, 2048)])
@torch.inference_mode()
def test_indexer_attention_is_the_same(prefix, length, monkeypatch):
    """`FlashNextIndexer.run` (the `original` backend) with and without the exact kernels: the same attention output."""
    from sharar.kernels.qsa import score_exact
    from tests.qsa_fixture import QsaFixture
    f = QsaFixture(1, length, prefix, hidden=256)
    aux = f.aux.clone()
    calls = []
    select_exact = score_exact.select_exact
    monkeypatch.setattr(score_exact, "select_exact", lambda *a: calls.append(1) or select_exact(*a))
    got = f.native().clone()
    assert calls
    f.aux.copy_(aux)
    monkeypatch.setattr(score_exact, "supported", lambda *a: False)
    want = f.native()
    assert len(calls) == 1 and torch.equal(got, want)
