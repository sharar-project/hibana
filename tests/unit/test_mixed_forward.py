"""Real executor/scheduler transactions with an exact small CPU recurrence."""
from types import SimpleNamespace

import pytest
import torch

from sharar.engine.contracts import ProposerContext, ProposerRequirements
from sharar.engine.rejection import Proposal
from sharar.engine.sampler import SamplingParams
from sharar.engine.scheduler import Scheduler, SchedulerConfig
from sharar.proposers.ngram import NGramProposer
from tests.unit.test_runtime_safety import tiny_executor


def runtime(monkeypatch, accepted=3):
    ex = tiny_executor(monkeypatch, cache=True)
    class Oracle(NGramProposer):
        requirements = ProposerRequirements('ngram', greedy_only=False)
        def ready(self, req):
            return req.prefill_done and bool(req.output_ids)
        def propose(self, requests, pages, inputs, *, graphs):
            rows = []
            for req in requests:
                state = int(ex.state_pool.ssm_all[0, req.slot, 0])
                token, row = req.output_ids[-1], []
                for _ in range(self.k):
                    state = (state * 31 + token) % 97
                    token = state + 1
                    row.append(token)
                if accepted < self.k:
                    row[accepted] = (row[accepted] + 1) % 128
                rows.append(row)
            tokens = torch.tensor(rows, dtype=torch.int64)
            return Proposal(tokens, torch.ones((*tokens.shape, 1)), tokens.unsqueeze(-1))
    ex.set_proposer(Oracle(ProposerContext(ex.device, 4, 512), k=3))
    ex.mixed_mode = 'full'
    ex._mixed_verify = SimpleNamespace(plan=lambda *a, **kw: None)
    ex.model.forward_mixed = lambda pi, pc, vi, vc, **kw: (ex.model.forward(pi, pc), ex.model.forward(vi, vc))
    return ex


@pytest.mark.parametrize('accepted', range(4))
@pytest.mark.parametrize('budget', range(1, 5))
@pytest.mark.parametrize('sampled', [False, True])
def test_stop_budget_state_and_rng_match_separate_transaction(monkeypatch, accepted, budget, sampled):
    results = []
    for mixed in (False, True):
        ex = runtime(monkeypatch, accepted)
        old = [ex.admit([1, i + 2, 3], SamplingParams(max_new_tokens=budget + 1,
                        temperature=.7 if sampled else 0, seed=7)) for i in range(3)]
        ex.prefill_batch([(r, 3) for r in old])
        new = ex.admit([7] * 8, SamplingParams(max_new_tokens=5, temperature=.7 if sampled else 0, seed=99))
        if mixed:
            assert ex.mixed_prefill([(new, 8)], old)
        else:
            ex.prefill(new)
            ex.decode_step(old)
        reqs = old + [new]
        results.append(([r.output_ids[:] for r in reqs], [r.committed_len for r in reqs],
                        ex.state_pool.ssm_all.clone(), ex.state_pool.conv_all.clone(),
                        [r.generator.get_state() if sampled else None for r in reqs]))
        assert all(r.reservation is None and r.committed_len == r.context_len - 1 for r in reqs)
        if not sampled:
            assert old[0].spec_emitted == min(accepted + 1, budget)
        for req in reqs:
            req.cacheable = False
            ex.release(req)
        ex.flush_cache()
        assert ex.kv_pool.free_count == 1024 and ex.state_pool.free_count == 15
    a, b = results
    assert a[:2] == b[:2]
    assert torch.equal(a[2], b[2]) and torch.equal(a[3], b[3])
    if sampled:
        assert all(torch.equal(x, y) for x, y in zip(a[4], b[4]))


@pytest.mark.parametrize('bad', ['duplicate', 'shared_slot', 'dummy', 'float_slot', 'finished', 'released', 'cancelled',
                                'extent', 'float_extent', 'pending', 'reservation'])
def test_authorization_rejects_entire_pair_before_mutation(monkeypatch, bad):
    ex = runtime(monkeypatch)
    old = ex.admit([1, 2, 3], SamplingParams())
    ex.prefill(old)
    new = ex.admit([5, 6, 7], SamplingParams())
    count = 3
    if bad == 'duplicate': new = old
    if bad == 'shared_slot': new.slot = old.slot
    if bad == 'dummy': new.slot = 0
    if bad == 'float_slot': new.slot = float(new.slot)
    if bad == 'finished': new.finished = True
    if bad == 'released': new.released = True
    if bad == 'cancelled': new.cancel_requested = True
    if bad == 'extent': count = 4
    if bad == 'float_extent': count = 3.0
    if bad == 'pending': old.committed_len -= 1
    if bad == 'reservation': ex.state.reserve([old], 1)
    before = ex.state_pool.ssm_all.clone(), ex.kv_pool.free_count
    with pytest.raises(ValueError):
        ex.state.authorize_mixed([(new, count)], [old], 4)
    assert torch.equal(before[0], ex.state_pool.ssm_all) and before[1] == ex.kv_pool.free_count


@pytest.mark.parametrize('stage', ['prepare', 'target', 'prefill_commit', 'verify_commit'])
def test_post_prepare_memory_failure_poisoning_prevents_retry_and_cache(monkeypatch, stage):
    ex = runtime(monkeypatch)
    old = ex.admit([1, 2, 3], SamplingParams())
    ex.prefill(old)
    new = ex.admit([7] * 8, SamplingParams())
    def fail(*args, **kwargs):
        raise MemoryError('injected partial failure')
    obj, method = {'prepare': (ex.spec, 'prepare'), 'target': (ex.model, 'forward_mixed'),
                   'prefill_commit': (ex, '_finish_prefill'), 'verify_commit': (ex.spec, 'finish')}[stage]
    monkeypatch.setattr(obj, method, fail)
    with pytest.raises(RuntimeError, match='uncacheable'):
        ex.mixed_prefill([(new, 8)], [old])
    assert not old.cacheable and not new.cacheable
    for req in (old, new):
        ex.release(req)
    ex.flush_cache()
    assert ex.kv_pool.free_count == 1024 and ex.state_pool.free_count == 15


def test_pool_preflight_is_retryable_and_fallback_has_no_side_effects(monkeypatch):
    ex = runtime(monkeypatch)
    old = ex.admit([1, 2, 3], SamplingParams())
    ex.prefill(old)
    new = ex.admit([7] * 8, SamplingParams())
    ex.mixed_mode = 'off'
    assert not ex.mixed_prefill([(new, 8)], [old])
    ex.mixed_mode = 'full'
    def fail(*args): raise MemoryError('preflight')
    monkeypatch.setattr(ex.state, 'ensure_pages', fail)
    with pytest.raises(MemoryError, match='preflight'):
        ex.mixed_prefill([(new, 8)], [old])
    assert all(r.cacheable and r.reservation is None for r in (old, new))
    assert new.prefill_pos == 0 and len(old.output_ids) == 1


@pytest.mark.parametrize('decode_steps', [1, 2])
def test_scheduler_consumes_one_round_and_new_prefill_joins_next(monkeypatch, decode_steps):
    ex = runtime(monkeypatch)
    sched = Scheduler(ex, SchedulerConfig(decode_steps_per_chunk=decode_steps, chunk_size=4))
    old = sched.submit([1, 2, 3], SamplingParams(max_new_tokens=40))
    sched.step()
    before = old.spec_rounds
    new = sched.submit([7] * 8, SamplingParams(max_new_tokens=20))
    sched.step()  # mixed first chunk, separate second chunk, no repeated old round
    assert old.spec_rounds == before + decode_steps
    assert new.spec_rounds == decode_steps - 1 and new.prefill_done
    assert sched.metrics.mixed_steps == 1
    sched.cancel(old)
    sched.cancel(new)
    sched.step()
    ex.flush_cache()
    assert old.released and new.released and not sched.active
    assert ex.kv_pool.free_count == 1024 and ex.state_pool.free_count == 15


@pytest.mark.parametrize('stop_position', range(1, 5))
def test_eos_inside_mixed_round_and_unfinished_prefill(monkeypatch, stop_position):
    ex = runtime(monkeypatch)
    old = ex.admit([1, 2, 3], SamplingParams(max_new_tokens=20))
    ex.prefill(old)
    state, token, expected = int(ex.state_pool.ssm_all[0, old.slot, 0]), old.output_ids[-1], []
    for _ in range(4):
        state = (state * 31 + token) % 97
        token = state + 1
        expected.append(token)
    assert len(set(expected)) == 4
    old.params = SamplingParams(max_new_tokens=20, stop_token_ids=(expected[stop_position - 1],))
    new = ex.admit([7] * 8, SamplingParams())
    ex.mixed_prefill([(new, 2)], [old])
    assert old.finished and old.finish_reason == 'stop' and old.output_ids[1:] == expected[:stop_position]
    assert old.committed_len == 3 + stop_position and old.reservation is None
    assert new.prefill_pos == new.committed_len == 2 and not new.output_ids
    for req in (old, new): ex.release(req)
    ex.flush_cache()
    assert ex.kv_pool.free_count == 1024 and ex.state_pool.free_count == 15


def test_shared_prefix_loan_survives_flush_and_mixed_continuation(monkeypatch):
    ex = runtime(monkeypatch)
    ex.generate([4] * 64, SamplingParams(max_new_tokens=1))
    old = ex.admit([4] * 64 + [8] * 4, SamplingParams(max_new_tokens=12))
    new = ex.admit([4] * 64 + [9] * 4, SamplingParams(max_new_tokens=12))
    assert old.prefix_len == new.prefix_len == 64
    borrowed = old.pages[:64].clone()
    ex.prefill(old)
    ex.flush_cache()
    ex.mixed_prefill([(new, 4)], [old])
    assert torch.equal(borrowed, old.pages[:64]) and torch.equal(borrowed, new.pages[:64])
    assert not set(borrowed.tolist()) & set(ex.kv_pool._free)
    for req in (old, new): ex.release(req)
    ex.flush_cache()
    assert ex.kv_pool.free_count == 1024 and ex.state_pool.free_count == 15


def test_unmeasured_interactive_policy_combination_is_explicit(monkeypatch):
    ex = runtime(monkeypatch)
    with pytest.raises(ValueError, match='fixed prefill policy'):
        Scheduler(ex, SchedulerConfig(interactive_target_ms=250))


def test_greedy_only_adapter_sampled_fallback_has_no_side_effects(monkeypatch):
    ex = runtime(monkeypatch)
    ex.spec.proposer.requirements = ProposerRequirements('ngram', greedy_only=True)
    old = ex.admit([1, 2, 3], SamplingParams(temperature=.7, seed=12))
    ex.prefill(old)
    new = ex.admit([7] * 8, SamplingParams())
    assert not ex.mixed_prefill([(new, 8)], [old])
    assert not new.output_ids and new.prefill_pos == 0 and len(old.output_ids) == 1
    assert all(r.reservation is None and r.cacheable for r in (old, new))


def test_cancel_between_prefill_chunks_never_enters_next_transaction(monkeypatch):
    ex = runtime(monkeypatch)
    sched = Scheduler(ex, SchedulerConfig(chunk_size=4))
    old = sched.submit([1, 2, 3], SamplingParams(max_new_tokens=40))
    sched.step()
    new = sched.submit([7] * 8, SamplingParams(max_new_tokens=20))
    def progress():
        if new.prefill_pos == 4:
            sched.cancel(new)
    sched.on_progress = progress
    sched.step()
    assert new.released and new.finish_reason == 'cancelled' and new.prefill_pos == 4
    assert not old.finished and old.cacheable and sched.metrics.mixed_steps == 1
    sched.cancel(old)
    sched.step()
    ex.flush_cache()
    assert ex.kv_pool.free_count == 1024 and ex.state_pool.free_count == 15
