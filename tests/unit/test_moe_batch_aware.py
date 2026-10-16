"""Batch-aware routing (SHARAR_MOE_KEEP / SHARAR_MOE_REUSE): `rule`, the rule's one reference
(tests/gpu/test_route_subst.py checks the fused kernel `flashnext_fused.route_subst` against it), and which MoE calls
take the kernel (decode and verify forwards of at most SHARAR_MOE_KEEP_ROWS rows, with their slots)."""
import sys
from types import SimpleNamespace

import pytest
import torch

from sharar.engine import config
from sharar.engine.context import ForwardContext
from sharar.model.flashnext.moe import MoeLayer

K, E, C = 10, 512, 20  # experts per token, experts, the rule's candidates per row (flashnext_fused.SUBST_CANDIDATES)


def candidates(logits):
    """Each row's C candidates: its largest logits first, an exact tie to the lower expert (-0.0 equals +0.0)."""
    return [sorted(range(len(row)), key=lambda e: (-row[e], e))[:C] for row in logits.float().cpu().tolist()]


def rule(logits, k, keep, slots=None, steps=1, fallback=True, repeat=False):
    """The rule, literally, one row at a time: the core is the union of every real row's top `keep` candidates; a real
    row takes its top `keep`, then its ranks keep+1..C in the core, then (`fallback`) its other ranks, each in rank
    order, the first k; a padded row (slots[r // steps] == 0: a graph's dummy request) its top k. Without `fallback`
    the slots left over, and all of a padded row's, hold id E (no expert) at weight 0, or with `repeat` the row's top 1
    (a padded row: row 0's) at weight 0. Weights: the fp32 softmax over
    all E at the chosen experts, renormalized over them. bf16 logits [R, E] -> (int32 ids [R, k], fp32 weights [R, k])
    on CPU."""
    cand = candidates(logits)
    real = [slots is None or int(slots[r // steps]) != 0 for r in range(len(cand))]
    core = {e for c, r in zip(cand, real) if r for e in c[:keep]}
    rows = []
    for c, r in zip(cand, real):
        if not r:
            row = c[:k] if fallback else []
        else:
            row = c[:keep] + [e for e in c[keep:] if e in core] + ([e for e in c[keep:] if e not in core] if fallback else [])
        row = row[:k]
        rows.append(row + [E] * (k - len(row)))
    ids = torch.tensor(rows)
    hole = ids == E
    p = torch.softmax(logits.float().cpu(), dim=-1).gather(1, ids.clamp(max=E - 1)) * ~hole
    total = p.sum(-1, keepdim=True)
    if repeat:
        fill = torch.tensor([c[0] if r else cand[0][0] for c, r in zip(cand, real)])
        ids = torch.where(hole, fill[:, None], ids)
    return ids.to(torch.int32), p / torch.where(total > 0, total, 1.0)


def _logits(rows=12, seed=0, scale=1.2):
    """bf16 router logits [rows, E]; at this scale exact ties inside a row's top 20 are common."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, E, generator=g) * scale).bfloat16()


def test_rule_invariants():
    """Every row keeps its top `keep` in order and picks within its top 20; the call reads no expert outside its rows'
    served top-k sets; weights are a distribution."""
    for seed, keep in [(0, 2), (1, 4), (2, 6), (3, 8)]:
        logits = _logits(16, seed)
        ids, w = rule(logits, K, keep)
        cands = candidates(logits)
        assert ids.dtype == torch.int32 and w.dtype == torch.float32 and ids.shape == w.shape == (16, K)
        for row, c in zip(ids.tolist(), cands):
            assert row[:keep] == c[:keep] and len(set(row)) == K and set(row) <= set(c)
        assert set(ids.reshape(-1).tolist()) <= {e for c in cands for e in c[:K]}
        torch.testing.assert_close(w.sum(-1), torch.ones(16))
        assert (w > 0).all()


def test_without_fallback_the_call_reads_only_the_core():
    """Compensation from the core alone: every row's experts lie in the core; slots no core candidate fills hold id E
    (no expert) at weight 0; with as many slots as the fallback rule fills, the in-core picks are the same; a padded
    row names no expert."""
    for seed, keep, k in [(0, 2, 10), (1, 5, 10), (2, 5, 7), (3, 8, 9)]:
        logits = _logits(16, seed)
        ids, w = rule(logits, k, keep, fallback=False)
        cands = candidates(logits)
        core = {e for c in cands for e in c[:keep]}
        assert set(ids.reshape(-1).tolist()) <= core | {E} and ids.shape == w.shape == (16, k)
        torch.testing.assert_close(w.sum(-1), torch.ones(16))
        with_fb, _ = rule(logits, k, keep)
        for row, wr, fb, c in zip(ids.tolist(), w.tolist(), with_fb.tolist(), cands):
            n = sum(x > 0 for x in wr)
            assert row[:keep] == c[:keep] and row[n:] == [E] * (k - n) and row[:n] == fb[:n]
            assert all(x > 0 for x in wr[:n]) and all(x == 0 for x in wr[n:])
    ids, w = rule(_logits(6, 9), 10, 5, torch.tensor([3, 0], dtype=torch.int32), steps=3, fallback=False)
    assert ids[3:].tolist() == [[E] * 10] * 3 and not w[3:].any()


def test_prefers_experts_the_call_already_reads():
    logits = _logits(2).float()
    shared = candidates(logits)[1][7]  # row 1's 8th choice made row 0's top: row 1 takes it in a reuse slot
    logits[0, shared] = logits[0].max() + 5.0
    ids, _ = rule(logits.bfloat16(), K, 6)
    assert shared in ids[1].tolist() and shared not in candidates(logits)[1][:6]


def test_exact_ties_go_to_the_lower_index():
    """Row a's ties straddle ranks 6/7 and 10/11 and hold -0.0 against +0.0; row b's top 6 pull a's ranks 11-14 in."""
    a = torch.full((E,), -4.0)
    a[[400, 20, 250]] = 3.0        # ranks 0-2: 20, 250, 400
    a[[511, 0, 77, 300, 5]] = 1.0  # ranks 3-7: 0, 5, 77, 300, 511
    a[[90, 30, 60, 10]] = 0.5      # ranks 8-11: 10, 30, 60, 90
    a[150], a[200] = -0.0, 0.0     # ranks 12-13: 150, 200 (-0.0 ties +0.0)
    b = torch.full((E,), -4.0)
    b[[150, 200, 90, 1, 2, 3]] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.5, 1.25])
    alone, _ = rule(a[None].bfloat16(), K, 6)
    assert alone.tolist() == [[20, 250, 400, 0, 5, 77, 300, 511, 10, 30]]  # the served top-10: 10 and 30 over 60 and 90
    ids, _ = rule(torch.stack([a, b]).bfloat16(), K, 6)
    assert ids.tolist() == [[20, 250, 400, 0, 5, 77, 90, 150, 200, 1],  # in-core ranks 11-14, -0.0 (150) before +0.0 (200)
                            [150, 200, 90, 1, 2, 3, 0, 5, 4, 6]]      # in-core ranks 6 and 8, then ranks 7 and 9


def test_padded_requests_change_nothing_and_take_their_top_k():
    """Requests of 3 rows each, the second and fourth padded (dummy slot 0)."""
    logits = _logits(12, 0)
    slots = torch.tensor([5, 0, 9, 0], dtype=torch.int32)
    real = torch.tensor([True] * 3 + [False] * 3 + [True] * 3 + [False] * 3)
    ids, w = rule(logits, K, 4, slots, steps=3)
    alone = rule(logits[real], K, 4)
    assert torch.equal(ids[real], alone[0]) and torch.equal(w[real], alone[1])
    assert ids[~real].tolist() == [c[:K] for c in candidates(logits[~real])]
    assert not torch.equal(rule(logits, K, 4)[0][real], alone[0])  # unmasked, the padding moves real rows' picks


# ------------------------------------------------------------------ which calls substitute


def _layer(keep=6, reuse=4, fallback=True, limit=64):
    """The routing state alone, no experts: by default keep 6, reuse 4 with the fallback."""
    layer = MoeLayer.__new__(MoeLayer)
    layer.router = torch.randn(E, 32, generator=torch.Generator().manual_seed(1)).bfloat16()
    layer.topk, layer.keep_rows = K, limit
    layer.keep, layer.reuse, layer.fallback, layer.w3 = keep, reuse, fallback, SimpleNamespace()
    return layer


def _ctx(mode, rows, slots=None, steps=0):
    pos = torch.zeros(rows, dtype=torch.int64)
    if mode == "verify":
        return ForwardContext(mode=mode, kv_pool=None, state_pool=None, positions=pos, seq_slots=slots, verify_steps=steps)
    return ForwardContext(mode=mode, kv_pool=None, state_pool=None, positions=pos, slots=slots)


@pytest.fixture
def kernels(monkeypatch):
    """flashnext_fused stand-in (the module needs Triton): route_subst notes its arguments, route_topk is the served
    route at K and "top<k>" below it."""
    calls = []

    def route_subst(logits, k, keep, slots=None, steps=1, fallback=True, repeat=False):
        assert logits.shape[1] == E
        calls.append(SimpleNamespace(k=k, keep=keep, slots=slots, steps=steps, fallback=fallback, repeat=repeat))
        return "subst"

    stub = SimpleNamespace(route_subst=route_subst, route_topk=lambda logits, k: "served" if k == K else f"top{k}")
    monkeypatch.setitem(sys.modules, "sharar.kernels.flashnext_fused", stub)
    return calls


def _built(w3=SimpleNamespace(base=0, experts=E), **changes):
    w = SimpleNamespace(get=lambda name: torch.zeros(E, 4, dtype=torch.bfloat16))
    target = SimpleNamespace(num_experts=E, experts_per_token=K, hidden_size=4, moe_intermediate_size=8,
                             shared_expert_intermediate_size=8)
    linear = SimpleNamespace(load=lambda *args: None)
    layer = MoeLayer.__new__(MoeLayer)
    with config.using(config.current().replace(**changes)), pytest.MonkeyPatch.context() as mp:
        if w3 is None:  # the routing state is set before the NVFP4 experts would load: stop there
            mp.setattr("sharar.model.flashnext.moe.load_nvfp4_experts", lambda *args: (_ for _ in ()).throw(_Loading()))
            with pytest.raises(_Loading):
                layer.__init__(w, "mlp", torch.device("cpu"), target, linear, "fp8", w3=None)
        else:
            layer.__init__(w, "mlp", torch.device("cpu"), target, linear, "fp8", w3=w3)
    return layer.keep, layer.reuse, layer.fallback


class _Loading(Exception):
    pass


def test_settings_resolve_to_keep_reuse_and_fallback():
    """SHARAR_MOE_KEEP / _REUSE / _REUSE_FALLBACK; keeping topk or more is the served route."""
    assert [_built(moe_keep=k)[:2] for k in (0, 1, 5, 9, 10, 12)] == [(0, 0), (1, 0), (5, 0), (9, 0), (0, 0), (0, 0)]
    assert _built(moe_keep=5, moe_reuse=5) == (5, 5, False)
    assert _built(moe_keep=5, moe_reuse=2, moe_reuse_fallback=True) == (5, 2, True)
    assert _built(moe_keep=10, moe_reuse=3)[:2] == (0, 0)
    assert _built(moe_keep=0, moe_reuse=5)[:2] == (0, 0)  # SHARAR_MOE_KEEP=0 turns off the profiles' keep 5 + reuse 5
    for changes, message in [({"moe_keep": -1}, "SHARAR_MOE_KEEP=-1"), ({"moe_keep": 5, "moe_reuse": -1}, "SHARAR_MOE_REUSE=-1"),
                             ({"moe_keep": 6, "moe_reuse": 5}, "at most 10")]:
        with pytest.raises(ValueError, match=message):
            _built(**changes)
    # the NVFP4 experts take the same selection (an unfilled slot repeats the token's top 1 at weight 0)
    assert _built(w3=None, moe_keep=5, moe_reuse=5) == (5, 5, False)
    assert _built(w3=None, moe_keep=5)[:2] == (5, 0)


def test_keep_alone_routes_decode_and_verify_forwards_to_route_topk(kernels):
    x = torch.randn(20, 32, generator=torch.Generator().manual_seed(5)).bfloat16()
    layer = _layer(keep=5, reuse=0)
    slots = torch.ones(4, dtype=torch.int32)
    assert layer.route(x, _ctx("verify", 20, slots, 5)) == "top5"
    assert layer.route(x, _ctx("decode", 20, torch.ones(20, dtype=torch.int32))) == "top5"
    assert layer.route(x, _ctx("prefill", 20)) == "served"
    layer.keep_rows = 16
    assert layer.route(x, _ctx("decode", 20, torch.ones(20, dtype=torch.int32))) == "served"
    assert kernels == []


def test_reuse_slots_set_the_width_and_the_fallback(kernels):
    x = torch.randn(20, 32, generator=torch.Generator().manual_seed(6)).bfloat16()
    slots = torch.ones(20, dtype=torch.int32)
    assert _layer(keep=5, reuse=5, fallback=False).route(x, _ctx("decode", 20, slots)) == "subst"
    assert _layer(keep=5, reuse=2, fallback=True).route(x, _ctx("decode", 20, slots)) == "subst"
    assert [(c.k, c.keep, c.fallback, c.repeat) for c in kernels] == [(10, 5, False, False), (7, 5, True, False)]
    nvfp4 = _layer(keep=5, reuse=5, fallback=False)
    nvfp4.w3 = None  # the NVFP4 experts need an expert in every slot: unfilled ones repeat the token's top 1
    nvfp4.route(x, _ctx("decode", 20, slots))
    assert (kernels[-1].fallback, kernels[-1].repeat) == (False, True)


def test_decode_and_verify_forwards_take_route_subst_with_their_slots(kernels):
    x = torch.randn(20, 32, generator=torch.Generator().manual_seed(2)).bfloat16()
    verify_slots = torch.tensor([3, 1, 4, 0], dtype=torch.int32)  # bucket 4: 3 requests x 5 rows and a padded one
    decode_slots = torch.tensor([7] * 13 + [0] * 7, dtype=torch.int32)
    assert _layer().route(x, _ctx("verify", 20, verify_slots, 5)) == "subst"
    assert _layer().route(x, _ctx("decode", 20, decode_slots)) == "subst"
    assert _layer(keep=0, reuse=0).route(x, _ctx("decode", 20, decode_slots)) == "served"
    assert [(c.keep, c.steps) for c in kernels] == [(6, 5), (6, 1)]
    assert kernels[0].slots is verify_slots and kernels[1].slots is decode_slots


def test_prefill_and_wide_calls_take_the_served_route(kernels):
    x = torch.randn(20, 32, generator=torch.Generator().manual_seed(3)).bfloat16()
    assert _layer().route(x, _ctx("prefill", 20)) == "served"
    assert _layer(limit=16).route(x, _ctx("decode", 20, torch.ones(20, dtype=torch.int32))) == "served"
    assert kernels == []
