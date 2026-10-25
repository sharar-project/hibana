"""Speculator parts: declared capabilities, shared draft heads, the depth policy."""
import random
from types import SimpleNamespace

import pytest
import torch

from sharar.engine.contracts import Capabilities, ProposerRequirements, check_capabilities
from sharar.engine.depth import FixedDepth
from sharar.proposers.draft_head import ShortlistHead, TargetHead


def test_declared_capabilities_need_their_methods():
    class Bare:
        requirements = ProposerRequirements("bare", capabilities=Capabilities(stored_payload=True))
    with pytest.raises(TypeError, match="stored_payload"):
        check_capabilities(Bare())

    class Stored(Bare):
        def auxiliary_codec(self):
            return None
    check_capabilities(Stored())

    class Head:
        requirements = ProposerRequirements("mtp", capabilities=Capabilities(draft_head=True))
        head = SimpleNamespace()
    with pytest.raises(TypeError, match="draft_head"):
        check_capabilities(Head())


def test_the_served_proposers_declare_what_they_implement():
    from sharar.engine.contracts import CheckpointCache, StoredPayload, DeviceFeedback
    from sharar.proposers.parallel import ParallelProposer
    from tests.ngram_proposer import NGramProposer
    for protocol in (DeviceFeedback, CheckpointCache, StoredPayload):
        assert all(callable(getattr(ParallelProposer, name, None)) for name in protocol.__dict__
                   if not name.startswith("_") and callable(getattr(protocol, name, None)))
    assert NGramProposer.requirements.capabilities == Capabilities()


def test_target_head_is_the_logits_function():
    logits = torch.randn(5, 40)
    head = TargetHead(lambda x: logits)
    assert head.bytes == 0 and torch.equal(head.argmax(None), logits.argmax(-1))
    values, ids = head.topk(None, 4)
    ref = logits.topk(4, dim=-1)
    assert torch.equal(values, ref.values) and torch.equal(ids, ref.indices)


def test_shortlist_topk_maps_rows_to_global_ids():
    from sharar.model.flashnext.linear import TrunkLinear
    torch.manual_seed(3)
    weight = torch.randn(20, 8).to(torch.bfloat16)
    x = torch.randn(2, 3, 8).to(torch.bfloat16)
    ids = torch.tensor([17, 2, 5, 11, 0])
    head = ShortlistHead(TrunkLinear(weight), ids)
    values, got = head.topk(x, 2)
    rows = ids.sort().values
    ref = (x.reshape(-1, 8) @ weight[rows].T).topk(2, dim=-1)
    assert got.shape == (2, 3, 2) and torch.equal(got.reshape(-1, 2), rows[ref.indices])
    assert torch.equal(values.reshape(-1, 2), ref.values)


def test_fixed_depth_and_the_verify_depth_setter():
    assert FixedDepth(4).choose([object()]) == 4
    with pytest.raises(ValueError):
        FixedDepth(0)
    from sharar.engine.spec import SpecDecoder
    spec = SpecDecoder.__new__(SpecDecoder)
    spec.k = 7
    assert spec.verify_depth == 7  # no policy yet: the proposer's k
    spec.verify_depth = 3
    assert isinstance(spec.depth_policy, FixedDepth) and spec.verify_depth == 3


def test_batch_depth_by_bucket():
    from sharar.engine.depth import BatchDepth, parse_depths
    sizes = (1, 2, 4, 8, 16)
    policy = parse_depths("8:1, 16:0", sizes, 4)
    assert [policy.choose([None] * n) for n in (1, 2, 3, 4, 5, 8, 9, 16)] == [4, 4, 4, 4, 1, 1, 0, 0]
    assert policy.verify_depths == (1, 4) and policy.depth == 4
    assert policy.sizes_for(4, sizes) == (1, 2, 4) and policy.sizes_for(1, sizes) == (8,)
    assert parse_depths(None, sizes, 4) is None and parse_depths("", sizes, 4) is None
    assert parse_depths("16:4", sizes, 4) is None  # every bucket keeps k: the fixed policy
    for bad in ("16", "16:x", "3:1", "16:5", "16:-1"):
        with pytest.raises(ValueError):
            parse_depths(bad, sizes, 4)
    with pytest.raises(ValueError):
        BatchDepth({4: 0}, 4).choose([None] * 5)  # beyond every bucket
    assert FixedDepth(3).sizes_for(3, sizes) == sizes


def test_profile_buckets_up_to_128():
    from sharar.engine.profile import ServingProfile
    assert ServingProfile().graph_sizes == (1, 2, 4)  # the default profile is unchanged
    assert ServingProfile(max_batch=16).graph_sizes == (1, 2, 4, 8, 16)
    assert ServingProfile(max_batch=12).graph_sizes == (1, 2, 4, 8, 12)
    assert ServingProfile(max_batch=128).graph_sizes == (1, 2, 4, 8, 16, 32, 64, 128)
    assert ServingProfile(max_batch=48).graph_sizes == (1, 2, 4, 8, 16, 32, 48)
    with pytest.raises(ValueError):
        ServingProfile(max_batch=129)



def test_acceptance_depth_follows_each_requests_rate_under_the_cap():
    from types import SimpleNamespace

    from sharar.engine.depth import AcceptanceDepth, BatchDepth
    sizes = (1, 2, 4, 8)
    policy = AcceptanceDepth(FixedDepth(8), step_cost=0.14)
    assert policy.verify_depths == tuple(range(1, 9)) and policy.sizes_for(8, sizes) == sizes and policy.sizes_for(9, sizes) == ()
    easy, hard, fresh = (SimpleNamespace(spec_rate=None) for _ in range(3))
    for _ in range(40):  # code-like: every draft accepted; prose-like: one in two
        policy.observe(easy, 8, 8)
        policy.observe(hard, 8, 1)
    assert policy.choose([easy]) == 8 and policy.choose([hard]) <= 2 and 2 < policy.choose([fresh]) < 8
    assert policy.choose([hard, easy]) == 8  # a batch verifies the deepest depth any request wants
    capped = AcceptanceDepth(BatchDepth({1: 8, 2: 4, 4: 4, 8: 0}, 8), step_cost=0.14)
    assert capped.choose([easy]) == 8 and capped.choose([easy, easy]) == 4 and capped.choose([easy] * 5) == 0
    assert capped.sizes_for(5, sizes) == (1,) and capped.sizes_for(3, sizes) == (1, 2, 4)
    # above four requests a batch verifies its bucket's depth, with that depth's graphs only (throughput profile memory)
    wide = AcceptanceDepth(BatchDepth({1: 8, 2: 4, 4: 4, 8: 4, 16: 3, 32: 3}, 8), step_cost=0.14)
    assert wide.choose([hard] * 8) == 4 and wide.choose([hard] * 20) == 3 and wide.choose([hard]) <= 2
    assert wide.verify_depths == tuple(range(1, 9)) and wide.sizes_for(3, (1, 2, 4, 8, 16, 32)) == (1, 2, 4, 16, 32)
    assert wide.sizes_for(2, (1, 2, 4, 8, 16, 32)) == (1, 2, 4) and wide.sizes_for(4, (1, 2, 4, 8, 16, 32)) == (1, 2, 4, 8)
    with pytest.raises(ValueError):
        AcceptanceDepth(FixedDepth(4), 0.0)


# Adaptive verification for a scoring drafter (DFlash2): engine/depth.RowBudget


def _requests(n, greedy=True):
    return [SimpleNamespace(params=SimpleNamespace(greedy=greedy)) for _ in range(n)]


def _padded():
    """Padded graphs at the measured shape on one Spark: rows nearly free to 64, the original weights past it."""
    from sharar.engine.depth import BUDGET_DEPTHS, RowBudget
    policy = RowBudget(15, (1, 2, 4, 8))
    for b in policy.sizes:
        for d in BUDGET_DEPTHS:
            rows = b * (d + 1)
            policy.costs[b, rows] = 52 + 0.1 * rows if b == 1 else 74 + 0.53 * rows if rows <= 64 else 132 + 0.13 * rows
    return policy


def _packed():
    from sharar.engine.depth import RowBudget, ragged_budgets
    budgets = ragged_budgets((1, 2, 4, 8), 16)
    policy = RowBudget(15, tuple(budgets), packed_rows=budgets)
    for b, rows in budgets.items():  # the measured shape: rows nearly free to 64, the original weights past it
        for r in rows:
            policy.costs[b, r] = 55 + 0.3 * r if r <= 64 else 95 + 0.1 * r
    return policy


def test_padded_row_budget_verifies_the_depth_the_drafts_scores_pay_for():
    from sharar.engine.depth import BUDGET_DEPTHS, RowBudget
    fresh = RowBudget(15, (1, 2, 4, 8))
    assert not fresh.packed and fresh.verify_depths == BUDGET_DEPTHS and fresh.depths is None
    assert fresh.choose(_requests(8)) == 15 and fresh.allot(_requests(8), torch.full((8, 15), 0.9), 15) == ([15] * 8, 128)  # unmeasured
    policy = _padded()
    assert policy.choose(_requests(8)) == 15  # greedy: the whole block, scored after drafting
    sure, unsure = torch.full((8, 15), 0.8), torch.full((8, 15), 0.4)  # at 0.97 all fifteen pay even past 64 rows
    deep, _ = policy.allot(_requests(8), sure, 15)
    shallow, _ = policy.allot(_requests(8), unsure, 15)
    assert len(set(deep)) == 1 and len(set(shallow)) == 1  # one depth for the batch
    assert 8 * (deep[0] + 1) <= 64 and shallow[0] < deep[0]  # within the 64-row step; unsure drafts verify fewer
    assert policy.allot(_requests(1), torch.full((1, 15), 0.97), 15) == ([15], 16)  # one request: rows are nearly free
    assert policy.chosen[8, deep[0]] == 1 and policy.chosen[8, shallow[0]] == 1


def test_packed_row_budget_gives_rows_to_the_likeliest_drafts():
    policy = _packed()
    reqs = _requests(8)
    acceptance = torch.full((8, 15), 0.3)
    acceptance[:2] = 0.95  # two requests draft text the target will take
    limits, rows = policy.allot(reqs, acceptance, 15)
    assert rows <= 64 and sum(limits) <= rows - 8
    assert min(limits[:2]) > max(limits[2:])  # the confident requests verify deeper
    assert policy.packed and policy.verify_depths == () and policy.choose(reqs) == 15  # greedy: allotted after drafting


def test_row_budget_explores_and_learns_the_tested_positions():
    for policy in (_packed(), _padded()):
        reqs = _requests(4)
        shy = torch.full((4, 15), 0.5)
        for round_ in range(1, 2 * policy.EXPLORE + 1):
            limits, rows = policy.allot(reqs, shy, 15)
            if round_ % policy.EXPLORE == 0:
                assert limits == [15] * 4  # every draft verified now and then
            for r, limit in zip(reqs, limits):
                policy.observe(r, limit, min(limit, 9))  # the target takes nine drafts in a row, more than 0.5 each says
        assert min(policy.shift[:3]) > 0.1


def test_row_budget_counts_only_the_positions_a_round_tested():
    from sharar.engine.depth import RowBudget
    policy = RowBudget(15, (1,), prior_weight=0.0)
    acc, tested = policy.survival[False]
    for _ in range(100):  # shallow rounds, every draft accepted: positions past 3 untested, not rejected
        policy.observe(None, 3, 3)
    assert tested[3] == 0 and acc[0] / tested[0] == 1.0
    for _ in range(100):  # a full-depth round rejected at position 5
        policy.observe(None, 15, 5)
    assert acc[5] == 0 and tested[6] == 0 and abs(acc[4] / tested[4] - 1.0) < 1e-9


def test_row_budget_keeps_sampled_batches_at_one_depth_from_past_rounds():
    for policy in (_packed(), _padded()):
        for _ in range(2000):
            policy.observe(_requests(1, greedy=False)[0], 15, 15 if random.random() < 0.3 else 2)
        sampled = _requests(3, greedy=False)
        depth = policy.choose(sampled)
        limits, rows = policy.allot(sampled, None, depth)
        assert 1 <= depth <= 15 and limits == [depth] * 3 and rows >= 3 * (depth + 1) + 1  # three requests in bucket 4


def test_row_budget_keeps_sampled_estimates_apart():
    policy = _packed()
    greedy, sampled = _requests(1)[0], _requests(1, greedy=False)[0]
    for _ in range(3000):
        policy.observe(greedy, 15, 15)  # greedy drafts all accepted
        policy.observe(sampled, 15, 1 if random.random() < 0.5 else 0)  # sampled ones rarely
    assert policy.expected_tokens(15) > 10 and policy.expected_tokens(15, sampled=True) < 2
    for _ in range(30):
        policy.observe_round(4, 64, 95.0, tokens=40)
        policy.observe_round(4, 16, 120.0, tokens=6, sampled=True)
    assert (4, True) in policy.overhead and policy.rate[4, True] < policy.rate[4, False]
    assert policy.choose(_requests(4, greedy=False)) <= 3  # sampled batches stop where their own drafts stop paying


def test_row_budget_learns_the_round_time_beyond_its_graphs():
    """Rows follow the drafts' scores, so rows and tokens move together: the round's unmeasured time is one moving mean
    per bucket, and the timed graphs carry the rows."""
    from sharar.engine.depth import RowBudget
    policy = RowBudget(15, (4,), packed_rows={4: (16, 32, 48, 64)})
    for rows in (16, 32, 48, 64):
        policy.costs[4, rows] = 60.0
    for rows in (16, 32, 48, 64) * 20:
        policy.observe_round(4, rows, 60.0 + 10.0 + 0.2 * rows, tokens=rows // 4)
    assert abs(policy.overhead[4, False] - (10.0 + 0.2 * 40)) < 0.5 and policy.cost(4, 16) == 60.0 + policy.overhead[4, False]


def test_dflash2_acceptance_reads_only_earlier_drafts():
    from sharar.proposers.parallel import ParallelProposer
    torch.manual_seed(0)
    candidates = torch.stack([torch.randperm(50)[:4] for _ in range(6)]).unsqueeze(0)  # [1, 6 positions, 4 ranks]
    scores = torch.randn(1, 6, 4, 4)
    owner = SimpleNamespace(calibration=(0.9693, -0.3068, -0.8203, 0.0344), acceptance_features=ParallelProposer.acceptance_features)
    tokens = candidates[:, :, 0].clone()
    a = ParallelProposer.acceptance(owner, candidates, scores, tokens)
    assert a.shape == (1, 6) and bool(((a > 0) & (a < 1)).all())
    other = tokens.clone()
    other[0, 3] = candidates[0, 3, 2]  # a different draft at position 3 changes positions after it only
    b = ParallelProposer.acceptance(owner, candidates, scores, other)
    assert torch.equal(a[:, :4], b[:, :4]) and not torch.equal(a[:, 4], b[:, 4])


def test_each_target_carries_its_pinned_drafts_calibration():
    from sharar import target as targets
    for name in ("qwen38", "flashnext"):
        weights = dict(targets.get(name).draft_calibration)["dflash2"]
        assert len(weights) == 4 and weights[0] > 0  # a higher top probability, a likelier acceptance


def test_ragged_budgets_cover_plain_decode_to_full_depth_and_128_rows():
    from sharar.engine.depth import ragged_budgets
    budgets = ragged_budgets((1, 2, 4, 8, 16), 16)
    assert budgets[1][0] == 1 and budgets[1][-1] == 16 and budgets[16][0] == 16 and budgets[16][-1] == 256
    assert 128 in budgets[8] and 128 in budgets[16] and all(r % 4 == 0 for r in budgets[4])


def test_tensor_parallel_ranks_learn_the_same_round_times_at_the_same_round(monkeypatch):
    """Each rank decides on its own, so every rank must learn the same times at the same round: the slowest rank's,
    every TP_LEARN_EVERY rounds, never the previous round's event as soon as it happens to be done."""
    from sharar.engine import spec as spec_mod

    class Event:
        def __init__(self, ms):
            self.ms = ms

        def synchronize(self):
            pass

        def query(self):
            return True

        def elapsed_time(self, end):
            return end.ms - self.ms

    monkeypatch.setattr(spec_mod, "get_tp_info", lambda: SimpleNamespace(size=2))
    monkeypatch.setattr(spec_mod, "all_reduce_max", lambda values: [v + 1.0 for v in values])  # the other rank: slower
    decoder = spec_mod.SpecDecoder.__new__(spec_mod.SpecDecoder)
    decoder._round_timings = []
    learned = []
    learn = lambda b, rows, ms, tokens, sampled: learned.append((b, rows, ms))  # noqa: E731
    every = decoder.TP_LEARN_EVERY
    for i in range(2 * every + 1):
        decoder._learn_round(learn)
        decoder._round_timings.append((1, 16, Event(0.0), Event(30.0 + i), 5, False))
        assert len(learned) == (every if i > every else 0)
    assert learned == [(1, 16, 31.0 + i) for i in range(every)]
