"""Packed verification (adaptive verification's spec_graphs.RaggedVerifyGraphs): requests verifying depths of their own
in one graph emit what the full-depth round emits, cut at each request's depth. The commits themselves are checked per
kernel (test_ssm_window.py, test_verify_conv.py: ragged sequences against each sequence alone)."""
import gc

import pytest
import torch

from hibana.engine import graph_set
from hibana.engine.sampler import SamplingParams


@pytest.fixture(scope='module')
def runtime(qwen38):
    from hibana import config
    from hibana.engine.config import current, using
    from hibana.engine.depth import RowBudget
    from hibana.engine.executor import Executor
    from hibana.engine.prompts import PARAGRAPH
    from hibana.tokenizer import SnapshotTokenizer
    gc.collect()  # the previous module's executor
    torch.cuda.empty_cache()
    with using(current().replace(spec_policy='budget')), torch.inference_mode():
        ex = Executor(qwen38, num_pages=35000, num_slots=12)
        ex.enable_spec(15, proposer='dflash2', state_strategy='reconstruct', draft_precision='fp8')
        ex.capture_graphs(log=print)
        ex.warmup(log=print)
        assert isinstance(ex.spec.depth_policy, RowBudget) and ex.spec.depth_policy.costs
        tok = SnapshotTokenizer(config.snapshot_dir())
        prompts = [tok.encode(PARAGRAPH * n)[:length] for n, length in ((8, 300), (6, 220), (4, 150), (3, 90))]
        yield ex, prompts
        ex.flush_cache()


def run_rounds(ex, prompts, params, rounds, first_limits=None):
    """Fresh requests, `rounds` decode steps; the first verifies `first_limits` (the policy's own choice after).
    Returns the outputs after prefill, after the first round and at the end, and each round's layout."""
    policy = ex.spec.depth_policy
    allot = policy.allot

    def forced(reqs, acceptance, depth):
        limits, rows = allot(reqs, acceptance, depth)
        if forced.first and first_limits is not None and acceptance is not None:
            bucket = graph_set.pick(policy.sizes, len(reqs))
            limits = list(first_limits)
            rows = next(r for r in policy.budgets[bucket] if r >= sum(n + 1 for n in limits) + bucket - len(reqs))
        forced.first = False
        return limits, rows
    forced.first = True
    layouts = []  # each round's (row total, every sequence's rows)

    def verify(batch, original=ex.spec._verify_ragged):
        layouts.append(batch.ragged[1:3])
        original(batch)
    ex.spec._verify_ragged = verify
    ex.flush_cache()
    reqs = [ex.admit(ids, p) for ids, p in zip(prompts, params)]
    for r in reqs:
        ex.prefill(r)
    prefilled = [r.output_ids[:] for r in reqs]
    first = None
    policy.allot = forced
    try:
        for _ in range(rounds):
            assert not any(r.finished for r in reqs)
            ex.decode_step(reqs)
            if first is None:
                first = [r.output_ids[:] for r in reqs]
    finally:
        del policy.allot, ex.spec._verify_ragged
    outputs = [r.output_ids[:] for r in reqs]
    for r in reqs:
        r.cacheable = False
        ex.release(r)
    return prefilled, first, outputs, layouts


@pytest.mark.parametrize('limits', [(15, 3, 0, 7), (2, 15, 5), (0, 1)])
@pytest.mark.parametrize('relaxed', [False, True])
@torch.inference_mode()
def test_each_request_emits_the_full_rounds_tokens_cut_at_its_depth(runtime, limits, relaxed):
    """(15, 3, 0, 7): four requests, filler rows after a request's own; (2, 15, 5): bucket 4 with a padded sequence;
    (0, 1): plain decode beside one draft. The reference round verifies every row the cut round computes (its filler
    rows too), in the same layout: another layout moves the target's arithmetic, by logits at deep rows
    (docs/operations.md#numerical-limits). Later rounds keep decoding; they are not compared token for token."""
    ex, prompts = runtime
    b = len(limits)
    params = [SamplingParams(max_new_tokens=256, ignore_eos=True, accept_top_k=3 if relaxed else 1,
                             accept_margin=2.0 if relaxed else 0.0) for _ in range(b)]
    prefilled, cut, later, layouts = run_rounds(ex, prompts[:b], params, 4, first_limits=limits)
    _, lengths = layouts[0]
    whole = [n - 1 for n in lengths[:b]]
    fillers = [i for i, (w, n) in enumerate(zip(whole, limits)) if w > n]
    assert bool(fillers) == (limits != (2, 15, 5))  # the padded sequence takes the spare rows there
    _, full, _, same = run_rounds(ex, prompts[:b], params, 1, first_limits=whole)
    assert same[0] == layouts[0]  # the same rows and sequences: the same forward
    accepted = [len(f) - len(p) - 1 for f, p in zip(full, prefilled)]
    assert not fillers or any(accepted[i] > limits[i] for i in fillers), (accepted, limits)  # fillers held accepted drafts
    for i, n in enumerate(limits):
        assert cut[i] == full[i][:len(prefilled[i]) + min(accepted[i], n) + 1], (i, accepted, limits)
        assert len(later[i]) >= len(cut[i]) + 3 and later[i][:len(cut[i])] == cut[i]


@torch.inference_mode()
def test_sampled_rounds_verify_one_depth_from_past_rounds(runtime):
    ex, prompts = runtime
    policy = ex.spec.depth_policy
    before = sum(policy.chosen.values())  # one-depth rounds, counted in allot
    params = [SamplingParams(max_new_tokens=256, ignore_eos=True, temperature=0.7, top_k=20, top_p=0.95, seed=i)
              for i in range(3)]
    prefilled, _, outputs, _ = run_rounds(ex, prompts[:3], params, 3)
    assert sum(policy.chosen.values()) == before + 3
    assert all(len(o) - len(p) >= 3 for o, p in zip(outputs, prefilled))
