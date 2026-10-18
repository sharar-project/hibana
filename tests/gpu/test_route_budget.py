"""Budgeted routing (`flashnext_fused.route_budget`) against the rule written out in torch: the call's set is every real
row's top `keep` plus the experts with the most router probability inside the real rows' top k, up to the budget; each
real row takes its candidates in the set in rank order, the first k. The scores are fp32 sums of the kernel's own
softmax, so the set is checked against the reference up to boundary near-ties; given the set, the ids are checked
bitwise."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
from sharar.kernels import flashnext_fused as fk  # noqa: E402
from tests.unit.test_moe_batch_aware import E, K, candidates  # noqa: E402


def logits_of(rows, seed, spread=1.2, shared=0.0):
    """Router-like bf16 logits; `shared` adds a common component so rows agree on experts, as in real verify calls."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(rows, E, generator=g) * spread + torch.randn(1, E, generator=g) * shared
    return (z - z.max(-1, keepdim=True).values + 3.0).to(torch.bfloat16)


def reference_set(logits, k, keep, budget, real):
    p = torch.softmax(logits.float().cpu(), -1)
    cand = candidates(logits)
    core = {e for c, r in zip(cand, real) if r for e in c[:keep]}
    score = torch.zeros(E)
    for r, c in enumerate(cand):
        if real[r]:
            for e in c[:k]:
                score[e] += p[r, e]
    order = sorted((e for e in range(E) if e not in core and score[e] > 0), key=lambda e: (-float(score[e]), e))
    return core, set(order[:max(0, budget - len(core))]), score


def check(logits, k, keep, budget, slots=None, steps=1):
    ids, w = fk.route_budget(logits.cuda(), k, keep, budget, None if slots is None else slots.cuda(), steps)
    ids, w = ids.cpu(), w.cpu()
    r = logits.shape[0]
    real = [slots is None or int(slots[i // steps]) != 0 for i in range(r)]
    core, extra, score = reference_set(logits, k, keep, budget, real)
    got = {int(e) for e in ids.reshape(-1).tolist()} - {E}
    assert core <= got and len(got) <= max(budget, len(core))
    # the set agrees with the reference except where the last places tie to fp32 rounding
    diff = (got - core) ^ extra
    if diff:
        boundary = sorted(float(score[e]) for e in (got - core) | extra)
        assert all(abs(float(score[e]) - boundary[0]) <= 1e-5 * max(1.0, boundary[-1]) for e in diff), diff
    cand = candidates(logits)
    for i, c in enumerate(cand):
        row = [e for e in c if e in got][:k] if real[i] else []
        want = row + [E] * (k - len(row))
        assert ids[i].tolist() == want, (i, ids[i].tolist(), want)
        if real[i]:
            assert ids[i, :keep].tolist() == c[:keep]  # the guaranteed top keep, first
            assert torch.isclose(w[i].sum(), torch.tensor(1.0), atol=1e-5)
    return got


@pytest.mark.parametrize("rows", [1, 5, 20, 64, 256])
@pytest.mark.parametrize("keep,budget", [(1, 8), (3, 16), (3, 24), (3, 64), (5, 20)])
def test_budget_matches_the_rule(rows, keep, budget):
    for seed, shared in [(rows, 0.0), (rows + 1, 1.0)]:
        got = check(logits_of(rows, seed, shared=shared), K, keep, budget)
        assert len(got) <= max(budget, keep * rows)


def test_padded_requests_take_no_expert_and_score_nothing():
    lg = logits_of(20, 7, shared=1.0)
    slots = torch.tensor([3, 0, 5, 0], dtype=torch.int32)  # 4 requests x 5 rows, two padded
    check(lg, K, 3, 16, slots, 5)
    ids, _ = fk.route_budget(lg.cuda(), K, 3, 16, slots.cuda(), 5)
    assert (ids.cpu()[5:10] == E).all() and (ids.cpu()[15:20] == E).all()


def test_graph_replay_gives_the_same_ids():
    lg = logits_of(40, 3, shared=1.0).cuda()
    eager = fk.route_budget(lg, K, 3, 24)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fk.route_budget(lg, K, 3, 24)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(eager[0], captured[0]) and torch.equal(eager[1], captured[1])


def test_per_request_cap_counts_only_real_requests():
    """budget P:B: the cap is min(B, P x real requests); a graph's padded requests (slot 0) do not raise it."""
    lg = logits_of(40, 11, shared=0.5)  # 8 requests x 5 rows
    slots = torch.tensor([3, 4, 0, 0, 0, 0, 0, 0], dtype=torch.int32)  # two real requests
    ids, _ = fk.route_budget(lg.cuda(), K, 1, 64, slots.cuda(), 5, per_request=6)
    got = {int(e) for e in ids.cpu().reshape(-1).tolist()} - {E}
    core = {c[0] for i, c in enumerate(candidates(lg)) if i < 10}
    assert core <= got and len(got) <= max(12, len(core))  # min(64, 6 x 2) = 12
    full, _ = fk.route_budget(lg.cuda(), K, 1, 64, torch.arange(1, 9, dtype=torch.int32).cuda(), 5, per_request=6)
    assert len({int(e) for e in full.cpu().reshape(-1).tolist()} - {E}) > 12  # eight real requests: min(64, 48)
    check(lg, K, 1, 12, slots, 5)  # the same set as a fixed cap of 12
