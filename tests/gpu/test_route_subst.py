"""Batch-aware routing (`flashnext_fused.route_subst`) against the rule's one reference, `rule` in
tests/unit/test_moe_batch_aware.py (the rule written out one row at a time): ids bitwise, weights to rtol 1e-5, on
router-like bf16 logits, exact ties at the rank-keep, rank-10 and rank-20 boundaries, logits of a few values with both
signed zeros, padded requests (dummy slot 0) at 1 and 5 rows per request, and inside a replayed CUDA graph. The same
checks run on CPU under the Triton interpreter (no graph test there):
`TRITON_INTERPRET=1 python -m pytest --noconftest tests/gpu/test_route_subst.py`."""
import os

import pytest
import torch

DEV = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
pytestmark = pytest.mark.skipif(DEV == "cuda" and not torch.cuda.is_available(), reason="GPU")
from sharar.kernels import flashnext_fused as fk  # noqa: E402
from tests.unit.test_moe_batch_aware import C, E, K, candidates, rule  # noqa: E402

ROWS = (1, 5, 20, 64, 65, 320, 640)
KEEPS = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def router_logits(rows: int, seed: int, top: float = 3.0) -> torch.Tensor:
    """Gaussian bf16 logits (sigma 1.2, about the router's spread) whose row maximum sits near `top`: the bf16 spacing
    there sets how often exact ties occur."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(rows, E, generator=g) * 1.2
    return (z - z.max(-1, keepdim=True).values + top).to(torch.bfloat16)


def tied(lg: torch.Tensor, at: int) -> torch.Tensor:
    """Each row's logit at rank `at` (0-based) set to rank at-1's, odd rows rank at+1's too: exact ties across the
    boundary between ranks at-1 and at."""
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


def check(logits: torch.Tensor, keep: int, slots: torch.Tensor | None = None, steps: int = 1, k: int = K,
          fallback: bool = True, repeat: bool = False):
    lg = logits.to(DEV).contiguous()
    sl = None if slots is None else slots.to(DEV)
    ids, w = fk.route_subst(lg, k, keep, sl, steps, fallback=fallback, repeat=repeat)
    assert ids.dtype == torch.int32 and w.dtype == torch.float32 and ids.is_contiguous() and w.is_contiguous()
    rids, rw = rule(logits, k, keep, slots, steps, fallback, repeat)
    assert torch.equal(ids.cpu(), rids)
    torch.testing.assert_close(w.cpu(), rw, rtol=1e-5, atol=0)
    return ids


def request_slots(n: int, seed: int) -> list[torch.Tensor]:
    """Slots for n requests: the graphs' trailing padding, padding anywhere (request 0 included), all padded."""
    g = torch.Generator().manual_seed(seed)
    real = torch.randint(1, 1000, (n,), generator=g, dtype=torch.int32)
    trailing = real.clone()
    trailing[n - max(1, n // 3):] = 0
    anywhere = torch.where(torch.arange(n) % 3 == 0, 0, real).to(torch.int32)
    return [trailing, anywhere, torch.zeros(n, dtype=torch.int32)]


@pytest.mark.parametrize("keep", KEEPS)
@pytest.mark.parametrize("rows", ROWS)
def test_matches_rule(rows, keep):
    seed = rows * 100 + keep
    base = router_logits(rows, seed)
    loud = router_logits(rows, seed + 1, top=8.0)  # bf16 steps of 1/32 just below 8: natural ties
    for lg in (base, loud, tied(base, keep), tied(base, K), tied(base, C), coarse(rows, seed)):
        check(lg, keep)


@pytest.mark.parametrize("steps", [1, 5])
@pytest.mark.parametrize("keep", KEEPS)
@pytest.mark.parametrize("rows", ROWS)
def test_padded_requests(rows, keep, steps):
    if rows % steps:
        pytest.skip("rows are whole requests")
    seed = rows * 1000 + keep * 10 + steps
    for i, slots in enumerate(request_slots(rows // steps, seed)):
        for lg in (tied(router_logits(rows, seed + i), keep), coarse(rows, seed + i)):
            ids = check(lg, keep, slots, steps)
            if slots.any():  # padded rows add nothing to the core: the same as their real rows routed alone
                real = (slots != 0).repeat_interleave(steps)
                alone, _ = fk.route_subst(lg[real].to(DEV).contiguous(), K, keep)
                assert torch.equal(ids[real.to(DEV)], alone)


@pytest.mark.parametrize("fallback,repeat", [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize("keep,k", [(2, 10), (5, 10), (5, 7), (8, 9)])
@pytest.mark.parametrize("rows", [1, 5, 20, 64, 65, 320])
def test_reuse_width_and_fallback(rows, keep, k, fallback, repeat):
    """k - keep reuse slots (SHARAR_MOE_KEEP / _REUSE), with and without the fallback, unfilled slots empty or
    repeating the row's top 1: ids bitwise, weights to rtol 1e-5, padded requests included; without the fallback the
    call reads only its core."""
    seed = rows * 100 + keep * 10 + k
    steps = 5 if rows % 5 == 0 else 1
    for lg in (router_logits(rows, seed), tied(router_logits(rows, seed + 1), keep), coarse(rows, seed)):
        ids = check(lg, keep, k=k, fallback=fallback, repeat=repeat)
        if not fallback:
            core = {e for c in candidates(lg) for e in c[:keep]}
            assert set(ids.reshape(-1).tolist()) <= (core if repeat else core | {E})
        for slots in request_slots(rows // steps, seed):
            check(lg, keep, slots, steps, k=k, fallback=fallback, repeat=repeat)


@pytest.mark.skipif(DEV != "cuda", reason="CUDA graphs")
def test_graph_replay_without_fallback():
    rows, steps, keep, k = 64, 4, 5, 10
    lg = router_logits(rows, 3).cuda()
    slots = torch.arange(1, rows // steps + 1, dtype=torch.int32, device="cuda")
    fk.route_subst(lg, k, keep, slots, steps, fallback=False)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        ids, w = fk.route_subst(lg, k, keep, slots, steps, fallback=False)
    assert graph_nodes(graph) == 2
    for seed in (4, 5):
        logits = router_logits(rows, seed)
        lg.copy_(logits)
        graph.replay()
        rids, rw = rule(logits, k, keep, slots.cpu(), steps, False)
        assert torch.equal(ids.cpu(), rids)
        torch.testing.assert_close(w.cpu(), rw, rtol=1e-5, atol=0)


@pytest.mark.parametrize("rows", [5, 64, 320])
def test_padded_rows_take_route_topk(rows):
    """A padded row's experts are what the served route gives it (route_topk orders by fp32 probability, so two
    different logits of equal probability could order differently; these logits have none)."""
    lg = tied(router_logits(rows, 7, top=8.0), K).to(DEV)
    ids, w = fk.route_subst(lg, K, 4, torch.zeros(rows, dtype=torch.int32, device=DEV))
    sids, sw = fk.route_topk(lg, K)
    assert torch.equal(ids, sids)
    torch.testing.assert_close(w, sw, rtol=1e-5, atol=0)


def test_moe_layer_route():
    """MoeLayer.route gives route_subst a decode forward's slots and a verify forward's request slots and steps;
    prefill takes route_topk."""
    from sharar.engine.context import ForwardContext
    from sharar.model.flashnext.moe import MoeLayer

    g = torch.Generator().manual_seed(4)
    layer = MoeLayer.__new__(MoeLayer)  # the routing state alone, no experts
    layer.router = torch.randn(E, 64, generator=g).bfloat16().to(DEV)
    layer.topk, layer.keep_rows, layer.keep, layer.reuse, layer.fallback = K, 64, 5, K - 5, True  # keep 5 + reuse 5, the fallback
    layer.w3 = object()  # the 3-bit experts: unfilled slots name no expert
    x = torch.randn(20, 64, generator=g).bfloat16().to(DEV)
    logits = x @ layer.router.T
    pos = torch.zeros(20, dtype=torch.int64, device=DEV)
    seq_slots = torch.tensor([3, 0, 8, 0], dtype=torch.int32)  # 5 rows per request, the second and fourth padded
    slots = torch.tensor([2] * 13 + [0] * 7, dtype=torch.int32)
    verify = ForwardContext(mode="verify", kv_pool=None, state_pool=None, positions=pos, seq_slots=seq_slots.to(DEV),
                            verify_steps=5)
    decode = ForwardContext(mode="decode", kv_pool=None, state_pool=None, positions=pos, slots=slots.to(DEV))
    prefill = ForwardContext(mode="prefill", kv_pool=None, state_pool=None, positions=pos)
    assert torch.equal(layer.route(x, verify)[0].cpu(), rule(logits, K, 5, seq_slots, 5)[0])
    assert torch.equal(layer.route(x, decode)[0].cpu(), rule(logits, K, 5, slots)[0])
    assert torch.equal(layer.route(x, prefill)[0], fk.route_topk(logits, K)[0])


def graph_nodes(graph: torch.cuda.CUDAGraph) -> int:
    from cuda.bindings import runtime
    err, _, n = runtime.cudaGraphGetNodes(graph.raw_cuda_graph(), 0)
    assert err == runtime.cudaError_t.cudaSuccess
    return n


@pytest.mark.skipif(DEV != "cuda", reason="CUDA graphs")
@pytest.mark.parametrize("rows,steps", [(5, 5), (20, 1), (64, 1), (320, 5), (640, 5)])
def test_graph_replay(rows, steps):
    keep, n = 5, rows // steps
    feeds = [(router_logits(rows, 1), torch.arange(1, n + 1, dtype=torch.int32))]
    feeds += [(tied(router_logits(rows, 2 + i), keep), s) for i, s in enumerate(request_slots(n, rows))]
    feeds += [(coarse(rows, 9), feeds[0][1])]
    lg = feeds[0][0].cuda()
    slots = feeds[0][1].cuda()
    fk.route_subst(lg, K, keep, slots, steps)  # compile outside the capture
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        ids, w = fk.route_subst(lg, K, keep, slots, steps)
    assert graph_nodes(graph) == 2  # the two launches: no memset, no host work
    for logits, sl in feeds:
        lg.copy_(logits)
        slots.copy_(sl)
        graph.replay()
        rids, rw = rule(logits, K, keep, sl, steps)
        assert torch.equal(ids.cpu(), rids)
        torch.testing.assert_close(w.cpu(), rw, rtol=1e-5, atol=0)
        first = ids.clone(), w.clone()
        graph.replay()
        assert torch.equal(ids, first[0]) and torch.equal(w.view(torch.int32), first[1].view(torch.int32))


@pytest.mark.skipif(DEV != "cuda", reason="FlashInfer's NVFP4 MoE")
def test_nvfp4_experts_take_the_selection_through_repeated_slots():
    """FlashInfer's NVFP4 MoE needs an expert in every slot: a slot repeating the row's top 1 at weight 0 gives the
    bits of leaving the slot out, so the NVFP4 experts serve the selection unchanged and read only the core."""
    from sharar import target as targets
    from sharar.model.flashnext.linear import TrunkLinear
    from sharar.model.flashnext.moe import MoeLayer
    from sharar.weights.safetensors import SnapshotWeights
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
        ids, w = fk.route_subst(lg, K, 5, fallback=False, repeat=True)
        used = (fk.route_subst(lg, K, 5, fallback=False)[0] < E).sum(1)
        for n in used.unique().tolist():  # the filled slots come first: compare with only them
            sel = used == n
            full = moe.routed(x[sel].contiguous(), ids[sel].contiguous(), w[sel].contiguous())
            kept = moe.routed(x[sel].contiguous(), ids[sel][:, :n].contiguous(), w[sel][:, :n].contiguous())
            assert torch.equal(full, kept), (rows, n)
