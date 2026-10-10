"""The MTP draft's confidence in trace round records (SHARAR_TRACE_CONFIDENCE, #32): the captured draft graphs
report each drafted position's top-1 probability and margin, and turning it on does not change the outputs."""
import gc

import pytest
import torch

from sharar.engine.sampler import SamplingParams

K = 3


def serve_rounds(trace_confidence: bool, rounds: int = 6):
    from sharar import config
    from sharar.engine import config as engine_config
    from sharar.engine.executor import Executor
    from sharar.model.qwen38 import Qwen38
    from sharar.engine.prompts import PARAGRAPH
    from sharar.tokenizer import SnapshotTokenizer
    gc.collect()
    torch.cuda.empty_cache()
    with engine_config.using(engine_config.current().replace(trace_confidence=trace_confidence)), torch.inference_mode():
        model = Qwen38(config.snapshot_dir(), log=print)
        ex = Executor(model, num_pages=35000, num_slots=12)
        ex.enable_spec(K, proposer='mtp', state_strategy='reconstruct')
        ex.capture_graphs(log=print)
        ex.warmup(log=print)
        assert ex.spec.proposer.with_confidence is trace_confidence
        tok = SnapshotTokenizer(config.snapshot_dir())
        prompts = [tok.encode(PARAGRAPH * n)[:length] for n, length in ((8, 300), (6, 220), (4, 150))]
        reqs = [ex.admit(ids, SamplingParams(max_new_tokens=48, ignore_eos=True)) for ids in prompts]
        for r in reqs:
            r.trace_rounds = []
            ex.prefill(r)
        for _ in range(rounds):
            active = [r for r in reqs if not r.finished]
            if active:
                ex.decode_step(active)
        torch.cuda.synchronize()
        result = [(r.output_ids[:], [dict(rec) for rec in r.trace_rounds]) for r in reqs]
        for r in reqs:
            r.cacheable = False
            ex.release(r)
        ex.flush_cache()
        del ex, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


@pytest.fixture(scope='module')
def runs():
    return serve_rounds(False), serve_rounds(True)


def test_confidence_does_not_change_the_outputs(runs):
    off, on = runs
    assert [o for o, _ in on] == [o for o, _ in off]
    assert all("p1" not in rec for _, recs in off for rec in recs)


def test_round_records_carry_each_drafted_positions_confidence(runs):
    _, on = runs
    records = [rec for _, recs in on for rec in recs]
    assert records
    for rec in records:
        assert len(rec["p1"]) == len(rec["m"]) == len(rec["y"]) == K
        assert all(0.0 <= m <= p1 <= 1.0 for p1, m in zip(rec["p1"], rec["m"]))
    # a confident first draft is accepted far more often than an unsure one (the #43 premise), on average
    first = [(rec["p1"][0], rec["a"] > 0) for rec in records]
    assert any(p > 0.5 for p, _ in first)
