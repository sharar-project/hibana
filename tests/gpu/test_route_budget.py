"""Budgeted expert selection (`flashnext_fused.route_budget`) against the rule in tests/unit/test_moe_selection.py.
The set's scores are fp32 sums of the kernel's own softmax, so the set is checked against the reference up to fp32
near-ties at its boundary; given the set, ids are bitwise and weights to rtol 1e-5. Router-like logits, exact ties at
the guarantee, rank-10 and rank-20 boundaries, both signed zeros, padded requests anywhere, a per-request cap, the
NVFP4 experts' repeated slots on real weights, and inside a replayed CUDA graph."""
import pytest
import torch

from hibana.kernels import flashnext_fused as fk  # noqa: E402
from tests.unit.test_moe_selection import C, E, K, rule, selected  # noqa: E402


def router_logits(rows: int, seed: int, top: float = 3.0, shared: float = 0.0) -> torch.Tensor:
    """Gaussian bf16 logits (sigma 1.2, about the router's spread) whose row maximum sits near `top` (the bf16 spacing
    there sets how often exact ties occur); `shared` adds a component common to the rows, as in real calls."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(rows, E, generator=g) * 1.2 + torch.randn(1, E, generator=g) * shared
    return (z - z.max(-1, keepdim=True).values + top).to(torch.bfloat16)


def tied(lg: torch.Tensor, at: int) -> torch.Tensor:
    """Each row's logit at rank `at` set to rank at-1's (odd rows rank at+1's too): exact ties across that boundary."""
    out = lg.clone()
    for r, row in enumerate(lg.float().tolist()):
        order = sorted(range(E), key=lambda e: (-row[e], e))
        for d in range(1, 2 + r % 2):
            out[r, order[at - 1 + d]] = lg[r, order[at - 1]]
    return out


def coarse(rows: int, seed: int) -> torch.Tensor:
    """Logits on a 1/8 grid in [-3, 0], zeros of both signs: each row's top 20 is mostly tied zeros."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randint(-24, 1, (rows, E), generator=g) * 0.125
    return torch.where(torch.rand(rows, E, generator=g) < 0.5, v, v.copysign(torch.tensor(-1.0))).to(torch.bfloat16)


def request_slots(n: int, seed: int) -> list[torch.Tensor]:
    """Slots for n requests: trailing padding (graphs), padding anywhere (request 0 included), all padded."""
    g = torch.Generator().manual_seed(seed)
    real = torch.randint(1, 1000, (n,), generator=g, dtype=torch.int32)
    trailing = real.clone()
    trailing[n - max(1, n // 3):] = 0
    anywhere = torch.where(torch.arange(n) % 3 == 0, 0, real).to(torch.int32)
    return [trailing, anywhere, torch.zeros(n, dtype=torch.int32)]


def check(logits, keep, cap, slots=None, steps=1, per_request=0, repeat=False, min_cap=0):
    ids, w = fk.route_budget(logits.cuda(), K, keep, cap, None if slots is None else slots.cuda(), steps, repeat=repeat,
                             per_request=per_request, min_cap=min_cap)
    ids, w = ids.cpu(), w.cpu()
    real = [slots is None or int(slots[i // steps]) != 0 for i in range(logits.shape[0])]
    if per_request:
        cap = min(cap, max(min_cap, per_request * sum(1 for i in range(0, len(real), steps) if real[i])))
    want, core, score = selected(logits, K, keep, cap, real)
    # the set as the real rows name it (with `repeat` a hole repeats the row's top 1, which is in it)
    got = {int(e) for i, row in enumerate(ids.tolist()) if real[i] for e in row} - {E}
    assert core <= got and len(got) <= max(cap, len(core))
    diff = (got - core) ^ (want - core)
    if diff:  # the set agrees except where its last places tie to fp32 rounding
        edge = [float(score[e]) for e in (got | want) - core]
        assert all(abs(float(score[e]) - min(edge)) <= 1e-5 * max(1.0, max(edge)) for e in diff), diff
    rids, rw = rule(logits, K, keep, got, slots, steps, repeat)
    assert torch.equal(ids, rids)
    torch.testing.assert_close(w, rw, rtol=1e-5, atol=0)
    return ids, got


@pytest.mark.parametrize("rows", [1, 5, 20, 64, 256])
@pytest.mark.parametrize("keep,cap", [(1, 8), (2, 16), (3, 24), (4, 64), (4, 512)])
def test_matches_the_rule(rows, keep, cap):
    seed = rows * 100 + keep * 10
    base = router_logits(rows, seed, shared=1.0)
    for lg in (base, router_logits(rows, seed + 1, top=8.0), tied(base, keep), tied(base, K), tied(base, C), coarse(rows, seed)):
        check(lg, keep, cap)


@pytest.mark.parametrize("steps", [1, 5])
@pytest.mark.parametrize("rows", [5, 20, 65, 320])
def test_padded_requests_score_nothing_and_name_no_expert(rows, steps):
    if rows % steps:
        pytest.skip("rows are whole requests")
    for i, slots in enumerate(request_slots(rows // steps, rows + steps)):
        for lg in (tied(router_logits(rows, i, shared=1.0), 3), coarse(rows, i)):
            ids, _ = check(lg, 3, 32, slots, steps)
            pad = ~(slots != 0).repeat_interleave(steps)
            assert (ids[pad] == E).all()


def test_per_request_cap_counts_only_real_requests():
    lg = router_logits(40, 11, shared=0.5)  # 8 requests x 5 rows
    two = torch.tensor([3, 4, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    _, got = check(lg, 1, 64, two, 5, per_request=6)  # min(64, 6 x 2) = 12
    assert len(got) <= 12
    _, got = check(lg, 1, 64, torch.arange(1, 9, dtype=torch.int32), 5, per_request=6)  # min(64, 48)
    assert 12 < len(got) <= 48


def test_a_request_alone_takes_the_minimum_cap():
    lg = router_logits(9, 12, shared=0.5)  # one request, a depth-8 draft
    _, got = check(lg, 1, 64, torch.tensor([5], dtype=torch.int32), 9, per_request=12, min_cap=16)  # max(16, 12)
    assert 12 < len(got) <= 16
    lg = router_logits(36, 13, shared=0.5)  # four requests: 48, above the floor
    _, got = check(lg, 1, 64, torch.arange(1, 5, dtype=torch.int32), 9, per_request=12, min_cap=16)
    assert 16 < len(got) <= 48


@pytest.mark.parametrize("rows,steps", [(5, 5), (20, 1), (320, 5)])
def test_graph_replay(rows, steps):
    from cuda.bindings import runtime
    n = rows // steps
    feeds = [(router_logits(rows, 1, shared=1.0), torch.arange(1, n + 1, dtype=torch.int32))]
    feeds += [(tied(router_logits(rows, 2 + i), 4), s) for i, s in enumerate(request_slots(n, rows))]
    lg, slots = feeds[0][0].cuda(), feeds[0][1].cuda()
    fk.route_budget(lg, K, 4, 64, slots, steps, per_request=12)  # compile outside the capture
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        ids, w = fk.route_budget(lg, K, 4, 64, slots, steps, per_request=12)
    err, _, nodes = runtime.cudaGraphGetNodes(graph.raw_cuda_graph(), 0)
    assert err == runtime.cudaError_t.cudaSuccess and nodes == 4  # the four launches: no memset, no host work
    for logits, sl in feeds:
        lg.copy_(logits)
        slots.copy_(sl)
        graph.replay()
        eager = fk.route_budget(logits.cuda(), K, 4, 64, sl.cuda(), steps, per_request=12)
        assert torch.equal(ids, eager[0]) and torch.equal(w.view(torch.int32), eager[1].view(torch.int32))


def test_nvfp4_experts_take_the_selection_through_repeated_slots():
    """FlashInfer's NVFP4 MoE needs an expert in every slot: a slot repeating the row's top 1 at weight 0 gives the
    bits of leaving the slot out, so the NVFP4 experts serve the selection unchanged and read only the set."""
    from hibana import target as targets
    from hibana.model.flashnext.linear import TrunkLinear
    from hibana.model.flashnext.moe import MoeLayer
    from hibana.weights.safetensors import SnapshotWeights
    target = targets.get("flashnext")
    try:
        snap = target.snapshot_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    weights = SnapshotWeights(snap)
    moe = MoeLayer(weights, "model.language_model.layers.3.mlp", torch.device("cuda"), target, TrunkLinear, "fp8")
    weights.close()
    g = torch.Generator(device="cuda").manual_seed(0)
    for rows in (1, 5, 16, 64):
        lg = router_logits(rows, rows).cuda()
        x = (torch.randn(rows, moe.hidden, device="cuda", generator=g) * 0.5).bfloat16()
        ids, w = fk.route_budget(lg, K, 4, 16, repeat=True)
        used = (fk.route_budget(lg, K, 4, 16)[0] < E).sum(1)
        for n in used.unique().tolist():  # the filled slots come first: compare with only them
            sel = used == n
            full = moe.routed(x[sel].contiguous(), ids[sel].contiguous(), w[sel].contiguous())
            kept = moe.routed(x[sel].contiguous(), ids[sel][:, :n].contiguous(), w[sel][:, :n].contiguous())
            assert torch.equal(full, kept), (rows, n)
