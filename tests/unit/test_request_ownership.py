"""Request ownership regressions; real control flow and small CPU state buffers."""

import asyncio
import time
from types import SimpleNamespace

import pytest
import torch

from sharar.engine.loop import EngineLoop, Ticket
from sharar.engine.request import Request, State
from sharar.engine.sampler import SamplingParams
from sharar.engine.scheduler import Scheduler
from sharar.engine.spec import SpecDecoder
from sharar.engine.contracts import VerifiedFeatures
from sharar.cache.request_state import StateManager, Reservation
from tests.unit.test_radix import make, snaps
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


@pytest.mark.parametrize('reason', ['length', 'stop', 'cancelled'])
def test_terminal_notification_observes_retired_ticket(monkeypatch, reason):
    from sharar.engine import loop as server

    async def scenario():
        engine = EngineLoop(Scheduler(FakeExecutor()))
        loop = asyncio.get_running_loop()
        finished = engine.sched.submit([1], SamplingParams(max_new_tokens=1, stop_token_ids=(7,)))
        active = engine.sched.submit([2], SamplingParams(max_new_tokens=8))
        tickets = [Ticket(queue=asyncio.Queue(), loop=loop, ready=loop.create_future(), req=req)
                   for req in (finished, active)]
        engine.tickets.update((t.req.request_id, t) for t in tickets)
        if reason == 'cancelled':
            engine.sched.cancel(finished)
        else:
            finished.push(7 if reason == 'stop' else 8)
        active.push(9)
        notify = server._notify

        def observe_notification(event_loop, callback, event):
            # The HTTP loop can consume a notification as soon as it is posted.
            # Inspect that boundary directly instead of relying on thread timing.
            if event[1]:
                assert finished.request_id not in engine.tickets
            assert active.request_id in engine.tickets
            notify(event_loop, callback, event)

        monkeypatch.setattr(server, '_notify', observe_notification)
        engine._publish()
        tokens = [] if reason == 'cancelled' else [7 if reason == 'stop' else 8]
        assert await asyncio.wait_for(tickets[0].queue.get(), 2) == (tokens, True, reason)
        assert await asyncio.wait_for(tickets[1].queue.get(), 2) == ([9], False, '')
        engine._publish()
        await asyncio.sleep(0)  # execute any erroneously repeated notification
        assert all(t.queue.empty() for t in tickets)
        assert engine.tickets == {active.request_id: tickets[1]}

    asyncio.run(scenario())


@pytest.mark.parametrize('accepted', range(4))
@pytest.mark.parametrize('budget', range(1, 5))
@pytest.mark.parametrize('stop_kind', ['length', 'eos'])
@pytest.mark.parametrize('strategy', ['snapshots', 'reconstruct'])
def test_speculative_commit_matches_returned_prefix(accepted, budget, stop_kind, strategy):
    # Marker j+1 represents the state after processing j+1 verify input tokens.
    pool = SimpleNamespace(ssm_all=torch.zeros((1, 2, 1)), conv_all=torch.zeros((1, 2, 1)), num_slots=2)
    freed = []
    ex = SimpleNamespace(state_pool=pool, kv_pool=SimpleNamespace(free_ids=freed.extend))
    ex.state = StateManager(ex.kv_pool, pool)
    spec = SpecDecoder.__new__(SpecDecoder)
    spec.ex, spec.k, spec.steps, spec.timing = ex, 3, 4, False
    spec.proposer = SimpleNamespace(accepted_batch=lambda *args: None)
    spec.inter_ssm_all = torch.arange(1, 5).reshape(1, 1, 4, 1).float()
    spec.conv_windows_all = spec.inter_ssm_all.clone()
    if strategy == 'reconstruct':
        def replay(pool, slots, counts):
            for row, (slot, count) in enumerate(zip(slots, counts)):
                pool.ssm_all[:, slot] = spec.inter_ssm_all[:, row, count - 1]
        spec.replay = SimpleNamespace(batch=1, steps=4, run=replay)
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
    cache.insert([1, 2, 3, 4], pages, snaps({4: slot}))
    match = cache.match([1, 2, 3, 4, 5])
    first, second = cache.lock(match.node, owner="a"), cache.lock(match.node, owner="b")
    cache.flush()
    assert not set(pages.tolist()) & set(kv._free)
    assert slot not in st._free
    assert cache.match([1, 2, 3, 4, 5]).length == 0
    cache.unlock(first)
    assert not set(pages.tolist()) & set(kv._free)
    cache.unlock(second)
    assert len(kv._free) == 999 and len(st._free) == 19
    assert cache.cached_tokens == 0
