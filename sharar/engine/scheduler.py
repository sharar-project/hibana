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
from dataclasses import dataclass, field

from sharar.engine.request import Request, State
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
    ttfts: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        ttfts = sorted(self.ttfts)
        p = lambda q: ttfts[min(len(ttfts) - 1, int(q * len(ttfts)))] if ttfts else float("nan")  # noqa: E731
        return {
            "admitted": self.admitted, "finished": self.finished, "cancelled": self.cancelled,
            "retractions": self.retractions, "rejected": self.rejected,
            "prefill_chunks": self.prefill_chunks, "prefill_tokens": self.prefill_tokens,
            "decode_steps": self.decode_steps, "decode_tokens": self.decode_tokens,
            "steps": self.steps, "avg_step_ms": 1000 * self.step_seconds / max(1, self.steps),
            "ttft_p50": p(0.5), "ttft_p95": p(0.95),
        }


class Scheduler:
    def __init__(self, executor, cfg: SchedulerConfig | None = None):
        self.ex = executor
        self.cfg = cfg or SchedulerConfig()
        self.queue: list[Request] = []
        self.prefilling: list[Request] = []
        self.running: list[Request] = []
        self.metrics = Metrics()
        self._next_id = 1

    # ---- API ----
    def submit(self, prompt_ids: list[int], params: SamplingParams) -> Request:
        if len(self.queue) >= self.cfg.max_queued:
            self.metrics.rejected += 1
            raise QueueFull(f"{len(self.queue)} requests queued")
        req = Request(prompt_ids=list(prompt_ids), params=params, request_id=self._next_id)
        self._next_id += 1
        self.queue.append(req)
        return req

    def cancel(self, req: Request) -> None:
        req.cancel()
        if req in self.queue:
            self.queue.remove(req)
            req.state = State.CANCELLED
            self.metrics.cancelled += 1

    @property
    def active(self) -> int:
        return len(self.queue) + len(self.prefilling) + len(self.running)

    def stats(self) -> dict:
        d = self.metrics.summary()
        d.update({"queued": len(self.queue), "prefilling": len(self.prefilling), "running": len(self.running)})
        if getattr(self.ex, "cache", None) is not None:
            d["cache"] = self.ex.cache.stats()
        return d

    # ---- loop ----
    def step(self) -> None:
        t0 = time.perf_counter()
        self._reap()
        self._admit()
        if self.prefilling:
            self._prefill_step()
            for _ in range(self.cfg.decode_steps_per_chunk):
                self._decode()
        else:
            self._decode()
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
                    lst.remove(req)
                    if req.cancel_requested and not req.finished:
                        req.state = State.CANCELLED
                        req.finished, req.finish_reason = True, "cancelled"
                        self.metrics.cancelled += 1
                    else:
                        self.metrics.finished += 1
                        self.metrics.ttfts.append(req.ttft)
                    self.ex.release(req)

    def _admit(self) -> None:
        while self.queue and len(self.prefilling) + len(self.running) < self.cfg.max_running:
            req = self.queue[0]
            if not self.ex.can_admit(len(req.prompt_ids), self.cfg.reserve_tokens):
                break
            self.queue.pop(0)
            self.ex.admit(req.prompt_ids, req.params, req)
            self.prefilling.append(req)
            self.metrics.admitted += 1

    def _prefill_step(self) -> None:
        """Spend up to chunk_size prompt tokens on the prefilling requests,
        oldest first, in one forward."""
        budget = self.cfg.prefill_budget
        chunks: list[tuple[Request, int]] = []
        for req in self.prefilling:
            if budget <= 0:
                break
            n = min(budget, self.cfg.chunk_size, len(req.prompt_ids) - req.prefill_pos)
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
                if not req.finished:
                    self.running.append(req)

    def _decode(self) -> None:
        if not self.running:
            return
        batch = list(self.running)
        while batch:
            try:
                self.ex.decode_step(batch)
                break
            except MemoryError:
                victim = max(batch, key=lambda r: r.request_id)  # newest arrival
                self._retract(victim)
                batch = [r for r in batch if r is not victim]
        self.metrics.decode_steps += 1
        self.metrics.decode_tokens += len(batch)

    def _retract(self, req: Request) -> None:
        for lst in (self.prefilling, self.running):
            if req in lst:
                lst.remove(req)
        self.ex.retract(req)
        self.queue.insert(0, req)
        self.metrics.retractions += 1
