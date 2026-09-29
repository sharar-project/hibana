"""Continuous-batching scheduler around the executor.

One `step()` does, in order: release finished or cancelled requests, admit
queued requests while there is room (FCFS, bounded by max_running, max_queued
and the pools), run one prefill chunk for the oldest request still prefilling,
and run one decode step for every running request. Prefill chunks and decode
steps are separate forwards; interleaving them bounds the decode stall to one
chunk and keeps the CUDA-graph decode path. Retraction under memory pressure
frees the newest running request back to the queue.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from sharar.engine.request import Request
from sharar.engine.contracts import ExecutionBackend
from sharar.engine.sampler import SamplingParams


class QueueFull(Exception):
    pass


@dataclass
class SchedulerConfig:
    max_running: int = 4
    max_queued: int = 16
    chunk_size: int = 2048  # most prompt tokens one request prefills per step
    prefill_budget: int = 4096  # most prompt tokens one step spends across requests (oldest first)
    decode_steps_per_chunk: int = 1  # decode steps run after each prefill step
    reserve_tokens: int = 64  # pages kept spare per admitted request for decoding

    def __post_init__(self) -> None:
        for name in ("max_running", "max_queued", "chunk_size", "prefill_budget", "decode_steps_per_chunk"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.reserve_tokens) is not int or self.reserve_tokens < 0:
            raise ValueError("reserve_tokens must be a non-negative integer")


@dataclass
class Metrics:
    admitted: int = 0
    finished: int = 0
    cancelled: int = 0
    retractions: int = 0
    rejected: int = 0
    prefill_chunks: int = 0
    prefill_tokens: int = 0
    decode_steps: int = 0
    decode_tokens: int = 0
    steps: int = 0
    step_seconds: float = 0.0
    # A long-lived local server must not retain/sort its entire request history.
    ttfts: deque[float] = field(default_factory=lambda: deque(maxlen=2048))

    @staticmethod
    def latency_summary(values, capacity: int = 2048) -> dict:
        ttfts = sorted(t for t in values if t == t)  # drop NaN
        p = lambda q: ttfts[min(len(ttfts) - 1, int(q * len(ttfts)))] if ttfts else None  # noqa: E731
        return {"ttft_p50": p(0.5), "ttft_p95": p(0.95),
                "ttft_window_count": len(ttfts), "ttft_window_capacity": capacity}

    def summary(self, *, include_latencies: bool = True) -> dict:
        return {
            "admitted": self.admitted, "finished": self.finished, "cancelled": self.cancelled,
            "retractions": self.retractions, "rejected": self.rejected,
            "prefill_chunks": self.prefill_chunks, "prefill_tokens": self.prefill_tokens,
            "decode_steps": self.decode_steps, "decode_tokens": self.decode_tokens,
            "steps": self.steps, "avg_step_ms": 1000 * self.step_seconds / max(1, self.steps),
            **(self.latency_summary(self.ttfts, self.ttfts.maxlen) if include_latencies else {}),
        }


class Scheduler:
    def __init__(self, executor: ExecutionBackend, cfg: SchedulerConfig | None = None, on_progress=None):
        self.ex = executor
        self.cfg = cfg or SchedulerConfig()
        if self.cfg.max_running > getattr(executor, "max_batch", self.cfg.max_running):
            raise ValueError("max_running exceeds executor batch capacity")
        # Called after every prefill step and every decode step, so a server
        # can deliver first tokens without waiting for the rest of the step,
        # and hand over requests that arrived meanwhile (they are admitted and
        # prefilled before the decode step, like the reference's prefill-first
        # policy, bounded by one extra prefill step).
        self.on_progress = on_progress
        # Optional engine/traces.py sink; recording is decided per request at submission.
        self.sink = None
        self.queue: list[Request] = []
        self.prefilling: list[Request] = []
        self.running: list[Request] = []
        self.metrics = Metrics()
        self._next_id = 1


    # ---- API ----
    def submit(self, prompt_ids: list[int], params: SamplingParams) -> Request:
        self.ex.validate_request(prompt_ids, params)
        if len(self.queue) >= self.cfg.max_queued:
            self.metrics.rejected += 1
            raise QueueFull(f"{len(self.queue)} requests queued")
        req = Request(prompt_ids=list(prompt_ids), params=params, request_id=self._next_id)
        self._next_id += 1
        if self.sink is not None and params.trace:
            req.trace_rounds = []
            req.trace_features = self.sink.recorder()
            req.trace_topk = self.sink.topk_recorder()
        self.queue.append(req)
        return req

    def cancel(self, req: Request) -> None:
        if req.finished:
            return
        req.cancel()
        if req in self.queue:
            self.queue.remove(req)
            req._finish("cancelled")
            req.released = True  # queued requests own no resources
            self.metrics.cancelled += 1

    @property
    def active(self) -> int:
        return len(self.queue) + len(self.prefilling) + len(self.running)

    def stats(self, *, include_latencies: bool = True) -> dict:
        d = self.metrics.summary(include_latencies=include_latencies)
        d.update({"queued": len(self.queue), "prefilling": len(self.prefilling), "running": len(self.running),
                  "floor_holds": getattr(self.ex, "floor_holds", 0)})
        cache = self.ex.cache_stats()
        if cache is not None:
            d["cache"] = cache
        if self.sink is not None:
            d["traces"] = self.sink.stats()
        return d

    # ---- loop ----
    def step(self) -> None:
        t0 = time.perf_counter()
        self._reap()
        self._admit()
        if self.prefilling:
            self._prefill_step()
            if self.on_progress:
                self.on_progress()
            self._admit()
            if self.prefilling:
                self._prefill_step()
                if self.on_progress:
                    self.on_progress()
            for _ in range(self.cfg.decode_steps_per_chunk):
                self._decode()
        else:
            self._decode()
        if self.on_progress:
            self.on_progress()
        self._reap()
        self.metrics.steps += 1
        self.metrics.step_seconds += time.perf_counter() - t0

    def run_until_idle(self) -> None:
        while self.active:
            self.step()

    # ---- internals ----
    def _reap(self) -> None:
        for lst in (self.prefilling, self.running):
            for req in list(lst):
                if req.finished or req.cancel_requested:
                    if req.cancel_requested and not req.finished:
                        req._finish("cancelled")
                    self.ex.release(req)
                    lst.remove(req)
                    if req.trace_rounds is not None and self.sink is not None and req.output_ids:
                        self.sink.record(req)
                    if req.finish_reason == "cancelled":
                        self.metrics.cancelled += 1
                    else:
                        self.metrics.finished += 1
                        self.metrics.ttfts.append(req.ttft)

    def _admit(self) -> None:
        while self.queue and len(self.prefilling) + len(self.running) < self.cfg.max_running:
            req = self.queue[0]
            reserve = self.ex.admission_reserve(req, self.cfg.reserve_tokens)
            if not self.ex.can_admit(req.prefill_ids, reserve):
                break
            self.ex.admit(req.prompt_ids, req.params, req)
            self.queue.pop(0)
            self.prefilling.append(req)
            self.metrics.admitted += 1

    def _prefill_step(self) -> None:
        """Spend up to chunk_size prompt tokens on the prefilling requests,
        oldest first, in one forward."""
        # on_progress may cancel an admitted request between the two chunks.
        # Remove it before reserving the next prefill chunk.
        self._reap()
        budget = self.cfg.prefill_budget
        chunks: list[tuple[Request, int]] = []
        for req in self.prefilling:
            if budget <= 0:
                break
            n = min(budget, self.cfg.chunk_size, req.prefill_len - req.prefill_pos)
            if n > 0:
                chunks.append((req, n))
                budget -= n
        if not chunks:
            return
        try:
            self.ex.prefill_batch(chunks)
        except MemoryError:
            self._retract(chunks[-1][0])
            return
        self.metrics.prefill_chunks += 1
        self.metrics.prefill_tokens += sum(n for _, n in chunks)
        for req, _ in chunks:
            if req.prefill_done:
                self.prefilling.remove(req)
                self.running.append(req)  # terminal requests still need to be reaped
        self._reap()

    def _decode(self) -> None:
        self._reap()
        if not self.running:
            return
        batch = list(self.running)
        before = {r.request_id: len(r.output_ids) for r in batch}
        while batch:
            try:
                self.ex.decode_step(batch)
                break
            except MemoryError:
                victim = max(batch, key=lambda r: r.request_id)  # newest arrival
                self._retract(victim)
                batch = [r for r in batch if r is not victim]
        self.metrics.decode_steps += 1
        self.metrics.decode_tokens += sum(len(r.output_ids) - before[r.request_id] for r in batch)

    def _retract(self, req: Request) -> None:
        for lst in (self.prefilling, self.running):
            if req in lst:
                lst.remove(req)
        self.ex.retract(req)
        self.queue.insert(0, req)
        self.metrics.retractions += 1
