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

    def prepare_release(self, req, snapshot):
        pass

    def capture_graphs(self, log=None):
        pass


def test_params_validate_and_parse():
    SamplingParams(accept_top_k=2, accept_margin=1.0).validate()
    assert not SamplingParams().relaxed and not SamplingParams(accept_top_k=2).relaxed
    assert SamplingParams(accept_top_k=2, accept_margin=0.5).relaxed
    assert not SamplingParams(accept_top_k=2, accept_margin=1.0, temperature=0.7).relaxed  # K:M is the greedy rule
    for bad in (dict(accept_top_k=0), dict(accept_top_k=2.0), dict(accept_margin=-1.0), dict(accept_margin=float("nan"))):
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


def test_relaxed_positions_are_recorded(monkeypatch):
    ex = tiny_executor(monkeypatch)
    ex.set_proposer(WrongDraft(ProposerContext(ex.device, ex.max_batch, ex.max_context), k=3))
    req = ex.admit([7, 7, 7], SamplingParams(max_new_tokens=9, accept_top_k=2, accept_margin=6.0))
    req.trace_rounds = []
    ex.prefill(req)
    while not req.finished:
        ex.decode_step([req])
    ex.release(req)
    offsets = [(r, j) for r in req.trace_rounds for j in r.get("rx", [])]
    assert offsets and all(0 <= j < 3 for _, j in offsets)
    assert all(j < r["a"] and r["y"][j] != r["z"][j] for r, j in offsets)  # accepted draft picks, not the argmax


def test_sampled_requests_get_the_soft_default_and_ignore_k_m():
    from sharar.api import validation
    saved = validation.ACCEPT_DEFAULT, validation.ACCEPT_SOFT_DEFAULT
    try:
        validation.set_accept_default(3, 2.0, soft=0.5)
        greedy, sampled = _sampling_from({"max_tokens": 4}), _sampling_from({"max_tokens": 4, "temperature": 1.0})
        assert (greedy.accept_top_k, greedy.accept_margin, greedy.relaxed) == (3, 2.0, True)
        assert (sampled.accept_top_k, sampled.accept_margin, sampled.accept_soft, sampled.relaxed) == (1, 0.0, 0.5, True)
        assert not _sampling_from({"max_tokens": 4, "temperature": 1.0, "accept_soft": 0}).relaxed
        explicit = _sampling_from({"max_tokens": 4, "temperature": 1.0, "accept_top_k": 1})
        assert explicit.relaxed  # K:M fields do not turn a sampled request's rule off
        validation.set_accept_default(1, 0.0, soft=0.5)  # independent of the greedy default
        assert not _sampling_from({"max_tokens": 4}).relaxed and _sampling_from({"max_tokens": 4, "temperature": 1.0}).relaxed
    finally:
        validation.ACCEPT_DEFAULT, validation.ACCEPT_SOFT_DEFAULT = saved
