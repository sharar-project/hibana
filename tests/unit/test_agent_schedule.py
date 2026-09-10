"""Experimental scheduling protects hot suffix work without probe eviction."""
import time

import torch

from bench.archive.agent_schedule import AgentScheduler, prefix_length
from sharar.engine.sampler import SamplingParams
from sharar.engine.scheduler import SchedulerConfig
from tests.unit.test_radix import make
from tests.unit.test_runtime_safety import tiny_executor
from tests.unit.test_scheduler import FakeExecutor


def test_prefix_estimate_uses_recurrent_boundary_without_allocating_or_touching(monkeypatch):
    cache, kv, states = make()
    cache.insert([1, 2, 3, 4], kv.alloc(4), {2: states.alloc_slot(), 4: states.alloc_slot()})
    before = cache.stats()
    def forbidden(*args, **kwargs):
        raise AssertionError('GPU metadata allocation during queue inspection')
    monkeypatch.setattr(torch, 'cat', forbidden)
    assert prefix_length(cache, [1, 2, 3]) == 2
    assert prefix_length(cache, [1, 2, 3, 4, 5]) == 4
    assert prefix_length(cache, [1, 9]) == 0
    assert cache.stats() == before
    cache.flushing = True
    assert prefix_length(cache, [1, 2, 3, 4]) == 0


def primed(monkeypatch):
    ex = tiny_executor(monkeypatch, cache=True)
    ids = [1, 2, 3, 4] * 16
    seed = ex.admit(ids, SamplingParams(max_new_tokens=1))
    ex.prefill(seed)
    ex.release(seed)
    return ex, ids


def test_hot_turn_bypasses_cold_head_that_does_not_fit_free_pages(monkeypatch):
    ex, ids = primed(monkeypatch)
    held, held_ids = ex.kv_pool.alloc_with_ids(ex.kv_pool.free_count - 100)
    sched = AgentScheduler(ex, SchedulerConfig(max_running=1))
    cold = sched.submit([8] * 256, SamplingParams(max_new_tokens=8))
    hot = sched.submit(ids + [5], SamplingParams(max_new_tokens=8))
    original, calls = ex.can_admit, []
    def can_admit(prompt, reserve):
        calls.append(prompt)
        return original(prompt, reserve)
    monkeypatch.setattr(ex, 'can_admit', can_admit)
    sched._admit()
    assert sched.prefilling == [hot] and sched.queue == [cold]
    assert hot.prefix_len == 64 and calls == [hot.prompt_ids]
    assert sched.resume_bypasses == 1
    hot.cacheable = False
    ex.release(hot)
    ex.kv_pool.free_ids(held_ids)
    ex.flush_cache()


def test_aged_cold_request_keeps_admission_priority(monkeypatch):
    ex, ids = primed(monkeypatch)
    sched = AgentScheduler(ex, SchedulerConfig(max_running=1), max_wait_s=2)
    cold = sched.submit([8] * 256, SamplingParams(max_new_tokens=8))
    hot = sched.submit(ids + [5], SamplingParams(max_new_tokens=8))
    cold.t_arrival = time.perf_counter() - 3
    sched._admit()
    assert sched.prefilling == [cold] and sched.queue == [hot]
    cold.cacheable = False
    ex.release(cold)
    ex.flush_cache()


def test_short_resume_finishes_while_long_prefill_makes_bounded_progress():
    ex = FakeExecutor()
    cfg = SchedulerConfig(chunk_size=2048, prefill_budget=4096)
    sched = AgentScheduler(ex, cfg)
    cold = sched.submit([8] * 10000, SamplingParams(max_new_tokens=8))
    hot = sched.submit([1] * 1128, SamplingParams(max_new_tokens=8))
    # Model the state after admission has restored a 1000-token hot prefix.
    for req in (cold, hot):
        ex.admit(req.prompt_ids, req.params, req)
    hot.prefix_len = hot.prefill_pos = 1000
    sched.queue.clear()
    sched.prefilling = [cold, hot]
    sched._prefill_step()
    assert hot.prefill_done and hot in sched.running
    assert cold.prefill_pos == 512 and cold in sched.prefilling
    assert sched.cfg is cfg and cfg.chunk_size == 2048
    sched._prefill_step()
    assert cold.prefill_pos == 2560  # ordinary chunk restored once the short turn finishes
