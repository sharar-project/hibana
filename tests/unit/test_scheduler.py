"""Scheduler policy tests against a fake executor with simulated time.

Costs mirror the measured engine: prefill 2,500 tok/s, decode 85 ms per step
(any batch size up to 4). No GPU involved.
"""

import pytest
import torch

from sharar.engine.request import Request, State
from sharar.engine.sampler import SamplingParams
from sharar.engine.scheduler import Metrics, QueueFull, Scheduler, SchedulerConfig


def test_latency_metrics_keep_a_bounded_recent_window():
    metrics = Metrics(finished=3000)
    metrics.ttfts.extend(range(3000))
    result = metrics.summary()
    assert result['finished'] == 3000
    assert result['ttft_window_count'] == result['ttft_window_capacity'] == 2048
    assert min(metrics.ttfts) == 952
    assert result['ttft_p50'] == 1976
    metrics.ttfts.append(float('nan'))
    assert metrics.summary()['ttft_window_count'] == 2047

PREFILL_TOKS_PER_S = 2500.0
DECODE_STEP_S = 0.085


class FakePool:
    def __init__(self, n):
        self._free = list(range(n))

    @property
    def free_count(self):
        return len(self._free)

    def alloc_with_ids(self, n):
        if n > self.free_count:
            raise MemoryError("pool exhausted")
        pages = self.alloc(n)
        return pages, pages.tolist()

    def free_ids(self, ids):
        self._free.extend(ids)

    def alloc(self, n):
        out = torch.tensor([self._free.pop() for _ in range(n)], dtype=torch.int32)
        return out

    def free(self, x):
        self._free.extend(int(v) for v in x.tolist()) if isinstance(x, torch.Tensor) else self._free.append(int(x))


class FakeExecutor:
    """Same interface as engine.executor.Executor; advances a simulated clock."""

    def __init__(self, pages=200_000, slots=8, eos_after=None):
        self.kv_pool = FakePool(pages)
        self.state_pool = FakePool(slots)
        self.cache = None
        self.clock = 0.0
        self.eos_after = eos_after or {}
        self.decode_batches = []
        self.released = []

    def validate_request(self, ids, params):
        params.validate()

    def cache_stats(self):
        return None

    def admission_reserve(self, req, limit):
        return min(limit, req.params.max_new_tokens - len(req.output_ids))

    def can_admit(self, prompt_ids, reserve_tokens=0):
        return len(self.kv_pool._free) >= len(prompt_ids) + reserve_tokens and len(self.state_pool._free) >= 1

    def admit(self, ids, params, req=None):
        req = req or Request(prompt_ids=list(ids), params=params)
        req.slot = int(self.state_pool._free.pop())
        req.pages = self.kv_pool.alloc(len(ids))
        req.prefill_pos = 0
        req.state = State.PREFILL
        req.t_admit = self.clock
        return req

    def prefill(self, req, max_tokens=None):
        n = len(req.prompt_ids)
        end = n if max_tokens is None else min(n, req.prefill_pos + max_tokens)
        self.clock += (end - req.prefill_pos) / PREFILL_TOKS_PER_S
        req.prefill_pos = end
        if end == n:
            self._push(req, 1000)
            return 1000
        return None

    def prefill_batch(self, chunks):
        for req, n in chunks:
            self.prefill(req, max_tokens=n)

    def decode_step(self, reqs):
        if len(self.kv_pool._free) < len(reqs):
            raise MemoryError("pool")
        pages = self.kv_pool.alloc(len(reqs))
        for i, r in enumerate(reqs):
            r.pages = torch.cat([r.pages, pages[i : i + 1]])
        self.clock += DECODE_STEP_S
        self.decode_batches.append(len(reqs))
        for r in reqs:
            tok = 248046 if self.eos_after.get(r.request_id) == len(r.output_ids) else 1000 + len(r.output_ids)
            self._push(r, tok)

    def _push(self, req, tok):
        if not req.output_ids:
            req.t_first_token = self.clock
        req.push(tok)
        if req.t_first_token == 0.0 or len(req.output_ids) == 1:
            req.t_first_token = self.clock  # simulated clock, not perf_counter

    def release(self, req):
        if req.released:
            return
        self.kv_pool.free(req.pages)
        self.state_pool.free(req.slot)
        req.pages, req.slot, req.released = None, -1, True
        self.released.append(req.request_id)

    def retract(self, req):
        self.kv_pool.free(req.pages)
        self.state_pool.free(req.slot)
        req.pages, req.slot = None, -1
        req.prefill_pos, req.output_ids, req.state = 0, [], State.QUEUED


def ttft(req):
    return req.t_first_token - req.t_arrival_sim


def run(sched, ex):
    while sched.active:
        sched.step()


def submit(sched, ex, n_prompt, n_out, ignore_eos=True):
    req = sched.submit([1] * n_prompt, SamplingParams(max_new_tokens=n_out, ignore_eos=ignore_eos))
    req.t_arrival_sim = ex.clock
    return req


def test_four_8k_requests_ttft_p95_under_15s_and_decode_batched():
    ex = FakeExecutor()
    sched = Scheduler(ex, SchedulerConfig(chunk_size=2048, prefill_budget=2048))
    reqs = [submit(sched, ex, 8192, 256) for _ in range(4)]
    run(sched, ex)
    ttfts = sorted(ttft(r) for r in reqs)
    assert all(r.finished and r.finish_reason == "length" for r in reqs)
    # 4 x 8192 tokens at 2.5K tok/s is 13.1 s of prefill; the last request also
    # waits for the decode steps interleaved after every chunk.
    assert ttfts[-1] < 15.0, ttfts
    assert ttfts[0] < 4.0, ttfts
    assert max(ex.decode_batches) == 4  # requests decode together once prefilled
    assert len(ex.released) == 4 and len(ex.kv_pool._free) == 200_000


def test_short_prompts_prefill_together_within_budget():
    ex = FakeExecutor()
    sched = Scheduler(ex, SchedulerConfig(chunk_size=2048))
    reqs = [submit(sched, ex, 128, 4) for _ in range(4)]
    sched.step()
    # one prefill step covered all four 128-token prompts
    assert sched.metrics.prefill_chunks == 1 and sched.metrics.prefill_tokens == 512
    assert all(r.state == State.RUNNING for r in reqs)


def test_queue_bound_and_running_bound():
    ex = FakeExecutor()
    sched = Scheduler(ex, SchedulerConfig(max_running=2, max_queued=3, chunk_size=512))
    reqs = [submit(sched, ex, 100, 8) for _ in range(3)]
    with pytest.raises(QueueFull):
        submit(sched, ex, 100, 8)
    assert sched.metrics.rejected == 1
    sched.step()  # admits two, one stays queued
    assert len(sched.prefilling) + len(sched.running) == 2 and len(sched.queue) == 1
    reqs += [submit(sched, ex, 100, 8) for _ in range(2)]
    assert len(sched.queue) == 3
    run(sched, ex)
    assert all(r.finished for r in reqs) and max(ex.decode_batches) <= 2


def test_cancel_queued_and_running():
    ex = FakeExecutor()
    sched = Scheduler(ex, SchedulerConfig(chunk_size=512))
    a = submit(sched, ex, 300, 50)
    b = submit(sched, ex, 300, 50)
    sched.cancel(b)  # still queued
    assert b.state == State.CANCELLED and len(sched.queue) == 1
    sched.step()  # a prefills and gets its first token
    sched.step()
    assert a.state == State.RUNNING
    sched.cancel(a)
    sched.step()
    assert a.finish_reason == "cancelled" and a.released and sched.active == 0
    assert sched.metrics.cancelled == 2 and len(ex.kv_pool._free) == 200_000


def test_stop_token_finishes_early():
    ex = FakeExecutor(eos_after={1: 3})  # request 1 emits EOS as its 4th token
    sched = Scheduler(ex, SchedulerConfig(chunk_size=512))
    r = submit(sched, ex, 64, 100, ignore_eos=False)
    run(sched, ex)
    assert r.finish_reason == "stop" and len(r.output_ids) == 4


def test_long_request_fits_and_pressure_retracts_newest():
    total = 31742 + 1024 + 64 + 600
    ex = FakeExecutor(pages=total)
    sched = Scheduler(ex, SchedulerConfig(chunk_size=2048, reserve_tokens=64))
    big = submit(sched, ex, 31742, 1024)
    small = submit(sched, ex, 500, 2000)  # both admitted; both decode until pages run out
    run(sched, ex)
    assert big.finished and big.finish_reason == "length" and len(big.output_ids) == 1024
    assert small.finished and len(small.output_ids) == 2000
    # the pool could not hold both to completion: the newer request (small) was
    # retracted, the older finished, then small ran again; nothing leaked
    assert sched.metrics.retractions >= 1
    assert len(ex.kv_pool._free) == total


def test_decode_never_waits_more_than_two_prefill_steps():
    ex = FakeExecutor()
    sched = Scheduler(ex, SchedulerConfig(chunk_size=1024, prefill_budget=1024))
    a = submit(sched, ex, 64, 64)
    run_steps = 0
    while a.state != State.RUNNING:
        sched.step()
        run_steps += 1
    b = submit(sched, ex, 8192, 8)  # a long prompt arrives while a decodes
    gaps = []
    last = ex.clock
    while not a.finished:
        sched.step()
        # every step with prefilling work still advances a's decode by one token
        gaps.append(ex.clock - last)
        last = ex.clock
    # one prefill step, plus one more for requests that arrived meanwhile
    assert max(gaps) <= 2 * (1024 / PREFILL_TOKS_PER_S) + DECODE_STEP_S + 1e-9
