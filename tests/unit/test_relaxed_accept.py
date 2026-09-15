"""Relaxed speculative acceptance: opt-in per request, exact by default, counted."""
import pytest
import torch

from sharar.api.validation import _sampling_from
from sharar.engine.contracts import ProposerContext, ProposerRequirements
from sharar.engine.sampler import InvalidRequest, SamplingParams
from tests.unit.test_runtime_safety import tiny_executor


class WrongDraft:
    """Feature-free parallel-style adapter that always proposes token 126."""

    requirements = ProposerRequirements(name="dflash2", greedy_only=False)

    def __init__(self, context, k):
        self.context, self.k = context, k

    def ready(self, req):
        return True

    def propose(self, requests, pages, inputs, *, graphs):
        return torch.full((len(requests), self.k), 126, dtype=torch.int64)

    def ingest_prefill(self, req, ids, start, result):
        pass

    def start(self, req, token):
        pass

    def accepted_batch(self, requests, emitted, features):
        pass

    def release(self, req):
        pass

    def prepare_release(self, req):
        pass

    def capture_graphs(self, log=None):
        pass


def test_params_validate_and_parse():
    SamplingParams(accept_top_k=2, accept_margin=1.0).validate()
    assert not SamplingParams().relaxed and not SamplingParams(accept_top_k=2).relaxed
    assert SamplingParams(accept_top_k=2, accept_margin=0.5).relaxed
    for bad in (dict(accept_top_k=0), dict(accept_top_k=2.0), dict(accept_margin=-1.0), dict(accept_margin=float("nan")),
                dict(accept_top_k=2, accept_margin=1.0, temperature=0.7)):
        with pytest.raises(InvalidRequest):
            SamplingParams(**bad).validate()
    params = _sampling_from({"sampling_params": {"max_new_tokens": 4, "accept_top_k": 2.0, "accept_margin": 1.5}})
    assert params.accept_top_k == 2 and params.accept_margin == 1.5 and params.relaxed
    assert not _sampling_from({"max_tokens": 4}).relaxed


@pytest.mark.parametrize("batch", [1, 3])
def test_relaxed_rule_accepts_within_margin_and_counts(monkeypatch, batch):
    # The tiny target puts logit 6 on the greedy token and 0 elsewhere: token 126
    # is rank 1 with a gap of exactly 6, so margin 6 accepts it and margin 5.9 does not.
    ex = tiny_executor(monkeypatch)
    exact = SamplingParams(max_new_tokens=9)
    expected = ex.generate([7, 7, 7], exact).output_ids
    ex.set_proposer(WrongDraft(ProposerContext(ex.device, ex.max_batch, ex.max_context), k=3))
    strict = ex.generate_batch([([7, 7, 7], SamplingParams(max_new_tokens=9, accept_top_k=2, accept_margin=5.9)) for _ in range(batch)])
    assert all(r.output_ids == expected and r.spec_relaxed == 0 for r in strict)
    assert ex.spec.stats()["relaxed_accepts"] == 0
    loose = ex.generate_batch([([7, 7, 7], SamplingParams(max_new_tokens=9, accept_top_k=2, accept_margin=6.0)) for _ in range(batch)])
    for r in loose:
        assert r.output_ids != expected and 126 in r.output_ids
        assert r.spec_relaxed == sum(1 for t in r.output_ids[1:] if t == 126)  # every 126 was a relaxed accept
        assert r.spec_rounds < strict[0].spec_rounds  # whole blocks are accepted at once
    assert ex.spec.stats()["relaxed_accepts"] == sum(r.spec_relaxed for r in loose)
    assert ex.kv_pool.free_count == 1024 and ex.state_pool.free_count == 15


def test_mixed_batch_keeps_exact_rows_exact(monkeypatch):
    ex = tiny_executor(monkeypatch)
    expected = ex.generate([7, 7, 7], SamplingParams(max_new_tokens=9)).output_ids
    ex.set_proposer(WrongDraft(ProposerContext(ex.device, ex.max_batch, ex.max_context), k=3))
    rows = ex.generate_batch([([7, 7, 7], SamplingParams(max_new_tokens=9)),
                              ([7, 7, 7], SamplingParams(max_new_tokens=9, accept_top_k=2, accept_margin=6.0)),
                              ([7, 7, 7], SamplingParams(max_new_tokens=9))])
    assert rows[0].output_ids == expected and rows[2].output_ids == expected
    assert rows[1].output_ids != expected and rows[1].spec_relaxed > 0
    assert rows[0].spec_relaxed == rows[2].spec_relaxed == 0


def test_server_default_applies_only_to_greedy_requests_that_do_not_choose(monkeypatch):
    from sharar.api import validation
    monkeypatch.setattr(validation, "ACCEPT_DEFAULT", (2, 1.0))
    assert _sampling_from({"max_tokens": 4}).relaxed  # greedy, silent: default applies
    p = _sampling_from({"max_tokens": 4, "accept_top_k": 1})
    assert not p.relaxed and p.accept_top_k == 1  # explicit opt-out
    p = _sampling_from({"max_tokens": 4, "sampling_params": {"accept_top_k": 3, "accept_margin": 2.0}})
    assert (p.accept_top_k, p.accept_margin) == (3, 2.0)  # explicit choice wins
    assert not _sampling_from({"max_tokens": 4, "temperature": 0.7}).relaxed  # sampled requests stay exact
    assert validation.parse_accept_default("off") == (1, 0.0) and validation.parse_accept_default("2:1.0") == (2, 1.0)
    with pytest.raises(ValueError):
        validation.parse_accept_default("nonsense")
    with pytest.raises(InvalidRequest):
        validation.set_accept_default(0, 1.0)


def test_relaxed_positions_are_recorded_masked_and_never_held_out(monkeypatch, tmp_path):
    import json
    from sharar.train.data import load_samples, split
    from sharar.train.loss import block_targets

    ex = tiny_executor(monkeypatch)
    ex.set_proposer(WrongDraft(ProposerContext(ex.device, ex.max_batch, ex.max_context), k=3))
    req = ex.admit([7, 7, 7], SamplingParams(max_new_tokens=9, accept_top_k=2, accept_margin=6.0))
    req.trace_rounds = []
    ex.prefill(req)
    while not req.finished:
        ex.decode_step([req])
    ex.release(req)
    offsets = [j for r in req.trace_rounds for j in r.get("rx", [])]
    assert offsets and all(0 <= j < 3 for j in offsets)
    # a trace file with one relaxed and one exact request, newest last
    def doc(rid, rounds, finish):
        return {"v": 2, "id": rid, "finish": finish, "reason": "length", "prompt_ids": [7, 7, 7], "output_ids": req.output_ids,
                "cached": 0, "params": {"temperature": 0.0, "max_new_tokens": 9}, "spec": {"k": 3, "rounds": rounds}}
    exact_rounds = [{k: v for k, v in r.items() if k != "rx"} for r in req.trace_rounds]
    (tmp_path / "traces-1.jsonl").write_text(json.dumps(doc(1, req.trace_rounds, 1.0)) + "\n" + json.dumps(doc(2, exact_rounds, 2.0)) + "\n")
    samples = load_samples(str(tmp_path), max_length=64, require_features=False, min_output=1)
    relaxed_sample, exact_sample = samples
    assert relaxed_sample.relaxed and not exact_sample.relaxed
    positions = set(relaxed_sample.relaxed)
    assert all(relaxed_sample.ids[p] == 126 for p in positions)  # every masked position is a draft pick
    train, held = split(samples, holdout_tokens=1, min_requests=1)
    assert held == [exact_sample] and train == [relaxed_sample]
    ids = torch.tensor(relaxed_sample.ids)
    banned = torch.zeros(len(ids), dtype=torch.bool)
    banned[list(positions)] = True
    anchors = torch.tensor([relaxed_sample.prompt_len])
    _, _, with_mask, _ = block_targets(ids, anchors, 4, relaxed_sample.prompt_len, None, banned)
    _, _, without, _ = block_targets(ids, anchors, 4, relaxed_sample.prompt_len, None)
    assert with_mask.sum() < without.sum()
    idx = (anchors[:, None] + torch.arange(4)[None, :]).clamp(max=len(ids) - 1)
    assert not bool((with_mask.bool() & banned[idx]).any())
