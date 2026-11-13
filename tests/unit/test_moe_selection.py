"""Budgeted expert selection (HIBANA_MOE_KEEP / HIBANA_MOE_BUDGET): `rule`, the selection written out one row at a time
(tests/gpu/test_route_budget.py checks the fused kernel `flashnext_fused.route_budget` against it), the settings, the
guarantee, and which MoE calls take the kernel (decode and verify forwards, with their slots)."""
import sys
from types import SimpleNamespace

import pytest
import torch

from hibana.engine import config
from hibana.engine.context import ForwardContext
from hibana.model.flashnext.moe import MoeLayer

K, E, C = 10, 512, 20  # experts per token, experts, candidates per row (flashnext_fused.ROUTE_CANDIDATES)


def candidates(logits):
    """Each row's C candidates: its largest logits first, an exact tie to the lower expert (-0.0 equals +0.0)."""
    return [sorted(range(len(row)), key=lambda e: (-row[e], e))[:C] for row in logits.float().cpu().tolist()]


def selected(logits, k, keep, cap, real):
    """The call's set: every real row's top `keep`, then the experts with the most router probability within the real
    rows' top k (summed in row order; equal scores to the lower expert) until it holds `cap`. -> (set, its guaranteed
    part, the scores)."""
    p = torch.softmax(logits.float().cpu(), -1)
    cand = candidates(logits)
    core = {e for c, r in zip(cand, real) if r for e in c[:keep]}
    score = torch.zeros(E)
    for r, c in enumerate(cand):
        if real[r]:
            for e in c[:k]:
                score[e] += p[r, e]
    rest = sorted((e for e in range(E) if e not in core and score[e] > 0), key=lambda e: (-float(score[e]), e))
    return core | set(rest[:max(0, cap - len(core))]), core, score


def rule(logits, k, keep, chosen, slots=None, steps=1, repeat=False):
    """Given the call's set `chosen`, each real row takes its candidates in it, in rank order, the first k; slots left
    over (all of a padded row's: slots[r // steps] == 0) hold id E at weight 0, or with `repeat` the row's top 1 (a
    padded row: row 0's). Weights: the fp32 softmax over E, renormalized over the chosen experts. -> CPU (ids, w)."""
    cand = candidates(logits)
    real = [slots is None or int(slots[r // steps]) != 0 for r in range(len(cand))]
    rows = [([e for e in c if e in chosen][:k] if r else []) for c, r in zip(cand, real)]
    ids = torch.tensor([row + [E] * (k - len(row)) for row in rows])
    hole = ids == E
    p = torch.softmax(logits.float().cpu(), dim=-1).gather(1, ids.clamp(max=E - 1)) * ~hole
    total = p.sum(-1, keepdim=True)
    if repeat:
        fill = torch.tensor([c[0] if r else cand[0][0] for c, r in zip(cand, real)])
        ids = torch.where(hole, fill[:, None], ids)
    return ids.to(torch.int32), p / torch.where(total > 0, total, 1.0)


def _logits(rows=12, seed=0, scale=1.2):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, E, generator=g) * scale).bfloat16()


def test_the_set_and_the_rows():
    """The guaranteed experts are in the set and come first in their rows; rows read only the set; the set holds the
    cap (more only for the guaranteed experts); a cap above the rows' top-k union routes every row in full."""
    for seed, keep, cap in [(0, 2, 16), (1, 4, 12), (2, 3, 64), (3, 1, 8)]:
        logits = _logits(16, seed)
        real = [True] * 16
        chosen, core, _ = selected(logits, K, keep, cap, real)
        ids, w = rule(logits, K, keep, chosen)
        cand = candidates(logits)
        assert core <= chosen and len(chosen) <= max(cap, len(core))
        assert set(ids.reshape(-1).tolist()) <= chosen | {E}
        for r, c in enumerate(cand):
            assert ids[r, :keep].tolist() == c[:keep]
        assert torch.allclose(w.sum(-1), torch.ones(16))
    logits = _logits(3, 9)
    chosen, _, _ = selected(logits, K, 2, 512, [True] * 3)
    assert torch.equal(rule(logits, K, 2, chosen)[0], torch.tensor([c[:K] for c in candidates(logits)], dtype=torch.int32))


def test_exact_ties_go_to_the_lower_index():
    logits = torch.zeros(1, E, dtype=torch.bfloat16)
    logits[0, [9, 3, 400]] = 1.0
    assert candidates(logits)[0][:5] == [3, 9, 400, 0, 1]


def test_padded_requests_score_nothing_and_name_no_expert():
    logits = _logits(12, 4)
    slots = torch.tensor([5, 0, 9, 0], dtype=torch.int32)
    real = [bool(slots[r // 3]) for r in range(12)]
    chosen, _, _ = selected(logits, K, 3, 16, real)
    ids, w = rule(logits, K, 3, chosen, slots, steps=3)
    alone, _, _ = selected(logits[torch.tensor(real)], K, 3, 16, [True] * 6)
    assert chosen == alone
    assert ids[~torch.tensor(real)].tolist() == [[E] * K] * 6 and not w[~torch.tensor(real)].any()


# ------------------------------------------------------------------ settings, the guarantee, which calls select

def _layer(keep=4, budget=64, per_request=12, w3=True):
    """The routing state alone, no experts."""
    layer = MoeLayer.__new__(MoeLayer)
    layer.router = torch.randn(E, 32, generator=torch.Generator().manual_seed(1)).bfloat16()
    layer.topk, layer.keep, layer.budget, layer.per_request = K, keep, budget, per_request
    layer.w3 = SimpleNamespace() if w3 else None
    return layer


def _ctx(mode, rows, slots=None, steps=0):
    pos = torch.zeros(rows, dtype=torch.int64)
    if mode == "verify":
        return ForwardContext(mode=mode, kv_pool=None, state_pool=None, positions=pos, seq_slots=slots, verify_steps=steps)
    return ForwardContext(mode=mode, kv_pool=None, state_pool=None, positions=pos, slots=slots)


@pytest.fixture
def kernels(monkeypatch):
    """flashnext_fused stand-in (the module needs Triton): route_budget notes its arguments, route_topk is the served
    route at K and "top<k>" below it; route_topk's arguments in `.topk`."""
    calls = type("Calls", (list,), {})()

    def route_budget(logits, k, keep, budget, slots=None, steps=1, repeat=False, per_request=0, min_cap=0):
        assert logits.shape[1] == E
        calls.append(SimpleNamespace(k=k, keep=keep, budget=budget, slots=slots, steps=steps, repeat=repeat, per_request=per_request,
                                     min_cap=min_cap))
        return "budget"

    def route_topk(logits, k, slots=None, steps=1, repeat=False):
        topk.append(SimpleNamespace(k=k, slots=slots, steps=steps, repeat=repeat))
        return "served" if k == K else f"top{k}"

    topk = calls.topk = []
    stub = SimpleNamespace(route_budget=route_budget, route_topk=route_topk)
    monkeypatch.setitem(sys.modules, "hibana.kernels.flashnext_fused", stub)
    return calls


def _built(w3=SimpleNamespace(base=0, experts=E), **changes):
    w = SimpleNamespace(get=lambda name: torch.zeros(E, 4, dtype=torch.bfloat16))
    target = SimpleNamespace(num_experts=E, experts_per_token=K, hidden_size=4, moe_intermediate_size=8,
                             shared_expert_intermediate_size=8)
    linear = SimpleNamespace(load=lambda *args: None)
    layer = MoeLayer.__new__(MoeLayer)
    with config.using(config.current().replace(**changes)), pytest.MonkeyPatch.context() as mp:
        if w3 is None:  # the routing state is set before the NVFP4 experts would load: stop there
            mp.setattr("hibana.model.flashnext.moe.load_nvfp4_experts", lambda *args: (_ for _ in ()).throw(_Loading()))
            with pytest.raises(_Loading):
                layer.__init__(w, "mlp", torch.device("cpu"), target, linear, "fp8", w3=None)
        else:
            layer.__init__(w, "mlp", torch.device("cpu"), target, linear, "fp8", w3=w3)
    return layer.keep, layer.budget, layer.per_request


class _Loading(Exception):
    pass


def test_settings_resolve_to_keep_and_budget():
    """HIBANA_MOE_KEEP / _BUDGET; keeping topk or more, or 0, is the served route; a budget needs a keep."""
    assert [_built(moe_keep=k)[0] for k in (0, 1, 4, 9, 10, 12)] == [0, 1, 4, 9, 0, 0]
    assert _built(moe_keep=4, moe_budget=64, moe_budget_per_request=12) == (4, 64, 12)
    assert _built(moe_keep=0, moe_budget=64)[:2] == (0, 0)
    assert _built(moe_keep=10, moe_budget=64)[:2] == (0, 0)
    for changes, message in [({"moe_keep": -1}, "HIBANA_MOE_KEEP=-1"), ({"moe_keep": 4, "moe_budget": -1}, "HIBANA_MOE_BUDGET=-1")]:
        with pytest.raises(ValueError, match=message):
            _built(**changes)
    assert _built(w3=None, moe_keep=4, moe_budget=64)[:2] == (4, 64)  # the NVFP4 experts take the same selection


def test_the_guarantee_shrinks_as_the_forward_grows():
    """2 cap / rows within [2, keep], the cap at least MIN_CAP: at 4 and 12:64, requests x tokens 1 x 1, 1 x 5, 1 x 8,
    4 x 5 keep 4; 1 x 9 (a depth-8 draft, cap 16) 3; 2 x 9 2; 8 x 5 3; 16 x 4 and up 2; a keep below 2 stays."""
    layer = _layer()
    shapes = [(1, 1), (1, 5), (1, 8), (4, 5), (1, 9), (2, 9), (8, 5), (16, 4), (64, 4), (128, 1)]
    assert [layer.guarantee(r * s, s) for r, s in shapes] == [4, 4, 4, 4, 3, 2, 3, 2, 2, 2]
    assert _layer(keep=1).guarantee(256, 4) == 1
    assert _layer(per_request=0).guarantee(5, 5) == 4  # a fixed cap of 64


def test_decode_and_verify_forwards_take_route_budget_with_their_slots(kernels):
    x = torch.randn(20, 32, generator=torch.Generator().manual_seed(2)).bfloat16()
    verify_slots = torch.tensor([3, 1, 4, 0], dtype=torch.int32)  # bucket 4: 3 requests x 5 rows and a padded one
    decode_slots = torch.tensor([7] * 13 + [0] * 7, dtype=torch.int32)
    assert _layer().route(x, _ctx("verify", 20, verify_slots, 5)) == "budget"
    assert _layer().route(x, _ctx("decode", 20, decode_slots)) == "budget"
    _layer(w3=False).route(x, _ctx("decode", 20, decode_slots))  # the NVFP4 experts need an expert in every slot
    assert [(c.k, c.keep, c.budget, c.per_request, c.min_cap, c.steps, c.repeat) for c in kernels] == [
        (K, 4, 64, 12, 16, 5, False), (K, 4, 64, 12, 16, 1, False), (K, 4, 64, 12, 16, 1, True)]
    assert kernels[0].slots is verify_slots and kernels[1].slots is decode_slots


def test_keep_alone_takes_route_topk_and_prefill_the_served_route(kernels):
    x = torch.randn(20, 32, generator=torch.Generator().manual_seed(5)).bfloat16()
    alone = _layer(budget=0)
    assert alone.route(x, _ctx("verify", 20, torch.ones(4, dtype=torch.int32), 5)) == "top4"
    assert alone.route(x, _ctx("decode", 20, torch.ones(20, dtype=torch.int32))) == "top4"
    assert _layer().route(x, _ctx("prefill", 20)) == "served"
    exact = _layer(keep=0, budget=0)
    decode_slots, verify_slots = torch.ones(20, dtype=torch.int32), torch.tensor([3, 1, 2, 0], dtype=torch.int32)
    assert exact.route(x, _ctx("decode", 20, decode_slots)) == "served"
    assert exact.route(x, _ctx("verify", 20, verify_slots, 5)) == "served"
    assert kernels == []
    # `exact` (no selection) still keeps a graph's padded requests (dummy slot 0) out of the experts it reads
    served = kernels.topk[-2:]
    assert [(c.k, c.steps, c.repeat) for c in served] == [(K, 1, False), (K, 5, False)]
    assert served[0].slots is decode_slots and served[1].slots is verify_slots
