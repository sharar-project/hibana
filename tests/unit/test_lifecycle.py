"""Request ownership regressions; real control flow and small CPU state buffers."""

import asyncio
import time
from types import SimpleNamespace

import pytest
import torch

from sharar.api.server import EngineLoop
from sharar.engine.request import Request, State
from sharar.engine.sampler import SamplingParams
from sharar.engine.scheduler import Scheduler
from sharar.engine.spec import SpecDecoder
from sharar.engine.contracts import VerifiedFeatures
from sharar.memory.request_state import StateManager, Reservation
from tests.unit.test_radix import make
from tests.unit.test_scheduler import FakeExecutor


@pytest.mark.parametrize('stop', ['length', 'eos'])
def test_finished_prefill_releases_once(stop):
    ex = FakeExecutor(pages=100, slots=4)
    sched = Scheduler(ex)
    params = SamplingParams(max_new_tokens=1 if stop == 'length' else 8,
                            stop_token_ids=(1000,) if stop == 'eos' else ())
    r = sched.submit([1] * 10, params)
    sched.run_until_idle()
    sched.cancel(r)
    sched.step()
    assert r.finished and r.released
    assert ex.released == [r.request_id]
    assert len(ex.kv_pool._free) == 100 and len(ex.state_pool._free) == 4
    assert sched.metrics.finished == 1 and sched.metrics.cancelled == 0


def test_queued_cancellation_is_terminal_and_idempotent():
    ex = FakeExecutor(pages=100, slots=4)
    sched = Scheduler(ex)
    r = sched.submit([1], SamplingParams())
    sched.cancel(r)
    sched.cancel(r)
    sched.step()
    assert r.finished and r.finish_reason == 'cancelled'
    assert r.state == State.CANCELLED and r.released
    assert sched.metrics.cancelled == 1 and sched.active == 0
    assert len(ex.kv_pool._free) == 100 and len(ex.state_pool._free) == 4


def test_idle_engine_publishes_queued_cancellation():
    async def scenario():
        ex = FakeExecutor(pages=1)
        engine = EngineLoop(Scheduler(ex))
        engine.start()
        try:
            ticket = await asyncio.wait_for(engine.submit([1], SamplingParams()), 2)
            engine.cancel(ticket)
            tokens, finished, reason = await asyncio.wait_for(ticket.queue.get(), 2)
            assert tokens == [] and finished and reason == 'cancelled'
            assert not engine.tickets
        finally:
            engine.stop = True
            await asyncio.to_thread(engine.thread.join, 2)
    asyncio.run(scenario())


@pytest.mark.parametrize('accepted', range(4))
@pytest.mark.parametrize('budget', range(1, 5))
@pytest.mark.parametrize('stop_kind', ['length', 'eos'])
def test_speculative_commit_matches_returned_prefix(accepted, budget, stop_kind):
    # Marker j+1 represents the state after processing j+1 verify input tokens.
    pool = SimpleNamespace(ssm_all=torch.zeros((1, 2, 1)), conv_all=torch.zeros((1, 2, 1)))
    freed = []
    ex = SimpleNamespace(state_pool=pool, kv_pool=SimpleNamespace(free_ids=freed.extend))
    ex.state = StateManager(ex.kv_pool, pool)
    spec = SpecDecoder.__new__(SpecDecoder)
    spec.ex, spec.k, spec.steps, spec.timing = ex, 3, 4, False
    spec.proposer = SimpleNamespace(accepted=lambda *args: None)
    spec.inter_ssm_all = torch.arange(1, 5).reshape(1, 1, 4, 1).float()
    spec.conv_windows_all = spec.inter_ssm_all.clone()
    spec.reset_stats()
    draft = [20, 30, 40]
    target = draft + [50]
    if accepted < 3:
        target[accepted] = 99
    proposed_output = draft[:accepted] + [target[accepted]]
    count = min(len(proposed_output), budget)
    stop_ids = (proposed_output[budget - 1],) if stop_kind == 'eos' and budget <= len(proposed_output) else ()
    params = SamplingParams(max_new_tokens=budget + 1 if stop_kind == 'length' else 20,
                            stop_token_ids=stop_ids)
    r = Request([1, 2, 3], params, slot=1, pages=torch.tensor([101, 102, 103]), prefill_pos=3)
    r.committed_len = 3
    r.push(10)
    r.reservation = Reservation(torch.tensor([104, 105, 106, 107]), [104, 105, 106, 107])
    now = time.perf_counter()
    spec._accept([r], torch.tensor([draft]), torch.tensor([target]), VerifiedFeatures(), now, now)
    expected = count if stop_kind == 'length' or stop_ids else len(proposed_output)
    assert r.output_ids == [10] + proposed_output[:expected]
    assert pool.ssm_all[0, 1, 0].item() == expected
    assert pool.conv_all[0, 1, 0].item() == expected
    assert r.pages.tolist() == list(range(101, 104 + expected))
    assert freed == list(range(104 + expected, 108))
    assert spec.emitted == expected
    assert r.committed_len == len(r.prompt_ids) + len(r.output_ids) - 1


def test_flush_preserves_active_prefix_until_last_release():
    cache, kv, st = make()
    pages = kv.alloc(4)
    slot = st.alloc_slot()
    cache.insert([1, 2, 3, 4], pages, {4: slot})
    match = cache.match([1, 2, 3, 4, 5])
    cache.lock(match.node, 2)
    cache.flush()
    assert not set(pages.tolist()) & set(kv._free)
    assert slot not in st._free
    assert cache.match([1, 2, 3, 4, 5]).length == 0
    cache.lock(match.node, -1)
    assert not set(pages.tolist()) & set(kv._free)
    cache.lock(match.node, -1)
    assert len(kv._free) == 999 and len(st._free) == 19
    assert cache.cached_tokens == 0
