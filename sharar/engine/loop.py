"""Engine thread ownership and asynchronous request tickets."""
from __future__ import annotations
import asyncio
import logging
import math
import os
import queue
import secrets
import threading
import time
from dataclasses import dataclass, field

import torch

from sharar.distributed import broadcast_pyobj, expect_peer_exit, get_tp_info
from sharar.engine.grammar import Grammars
from sharar.engine.request import Request
from sharar.engine.sampler import InvalidRequest, SamplingParams
from sharar.engine.scheduler import Metrics, QueueFull, Scheduler

class EngineUnavailable(RuntimeError):
    pass


def _resolve(future, value=None, error=None) -> None:
    if not future.done():
        future.set_exception(error) if error is not None else future.set_result(value)


def _notify(loop, callback, *args) -> None:
    try:
        loop.call_soon_threadsafe(callback, *args)
    except RuntimeError:
        if not loop.is_closed():
            raise


@dataclass
class Ticket:
    queue: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    ready: asyncio.Future
    req: Request | None = None
    t_submit: float = field(default_factory=time.perf_counter)


class EngineLoop:
    """Owns the scheduler on one thread. The HTTP side never touches the
    scheduler directly: it posts commands to a thread-safe inbox and receives
    results through per-request asyncio queues, so the event loop is never
    blocked and the engine thread never waits on HTTP.

    Every tensor-parallel rank runs one (mini-sglang's scheduler IO): the
    primary rank takes commands from the HTTP inbox and broadcasts the ones it
    applied; the other ranks receive and apply them at the same points (every
    `_publish`: before and after a step, and each `on_progress` inside it), so
    their schedulers take the same steps and their forwards meet the
    primary's collectives. An engine failure ends the process, since the other
    ranks' collectives cannot complete without it."""

    def __init__(self, sched: Scheduler, grammars: Grammars | None = None):
        self.sched = sched
        self.grammars = grammars  # ranks above 0 compile request constraints again from their source
        sched.on_progress = self._publish
        self.tp = get_tp_info()
        self._recv_commands = self._recv_commands_rank0 if self.tp.is_primary() else self._recv_commands_rank1
        self._applied: list[tuple] = []  # commands applied since the last broadcast
        self.inbox: queue.Queue = queue.Queue()
        self.tickets: dict[int, Ticket] = {}
        self.stop = False
        self.drain_deadline: float | None = None
        self.failure: str | None = None
        self._status_lock = threading.Lock()
        self.last_stats: dict = {}
        self._latency_finished = -1
        self._latencies: tuple[float, ...] = ()
        self.thread = threading.Thread(target=self._run, name="sharar-engine", daemon=True)

    def start(self) -> None:
        self.thread.start()

    @property
    def healthy(self) -> bool:
        return self.failure is None and not self.stop and self.drain_deadline is None and self.thread.is_alive()

    def drain(self, timeout: float = 30.0) -> None:
        """Close admission and finish accepted work, with a bounded deadline.

        Nonblocking: callers can continue consuming streams while we drain.
        Unexpected forward failures still invalidate shared state.
        """
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("drain timeout must be finite and nonnegative")
        with self._status_lock:
            deadline = time.monotonic() + timeout
            self.drain_deadline = min(self.drain_deadline, deadline) if self.drain_deadline is not None else deadline

    def _enqueue(self, cmd) -> None:
        # Serialize admission with failure/drain so no future is left in an
        # inbox whose sole consumer has exited.
        with self._status_lock:
            available = self.failure is None and not self.stop and self.thread.is_alive()
            if not available or (self.drain_deadline is not None and cmd[0] != "cancel"):
                raise EngineUnavailable("engine unavailable")
            self.inbox.put(cmd)

    async def submit(self, prompt_ids: list[int], params: SamplingParams) -> Ticket:
        loop = asyncio.get_running_loop()
        ticket = Ticket(queue=asyncio.Queue(), loop=loop, ready=loop.create_future())
        self._enqueue(("submit", prompt_ids, params, ticket))
        try:
            await ticket.ready
        except asyncio.CancelledError:
            self.cancel(ticket)
            raise
        return ticket

    def cancel(self, ticket: Ticket) -> None:
        try:
            self._enqueue(("cancel", ticket))
        except EngineUnavailable:
            pass  # failure/shutdown owns cleanup

    async def flush_cache(self) -> None:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._enqueue(("flush", fut, loop))
        await fut

    def stats(self) -> dict:
        # Called by the HTTP side. Sorting a bounded immutable snapshot never
        # stalls the engine or iterates its concurrently changing deque.
        return {**self.last_stats, **Metrics.latency_summary(self._latencies)}

    # ---- engine thread ----
    def _apply(self, cmd: tuple):
        """Apply one scheduler command, alike on every rank: ("submit", ids, params), ("cancel", request_id),
        ("flush",) or ("stop",). The primary records it for the other ranks."""
        kind, result = cmd[0], None
        if kind == "submit":
            result = self.sched.submit(cmd[1], cmd[2])
        elif kind == "cancel":
            req = next((r for r in (*self.sched.queue, *self.sched.prefilling, *self.sched.running) if r.request_id == cmd[1]), None)
            if req is not None:
                self.sched.cancel(req)
        elif kind == "flush":
            self.sched.ex.flush_cache()
        elif kind == "stop":
            self.stop = True
            expect_peer_exit()
        if self.tp.is_primary():
            self._applied.append(cmd)
        return result

    def _recv_commands_rank0(self) -> None:
        """Apply what arrived over HTTP, then hand it to the other ranks."""
        try:
            while True:
                self._handle(self.inbox.get_nowait())
        except queue.Empty:
            pass
        self._broadcast_applied()

    def _recv_commands_rank1(self) -> None:
        """Apply what the primary applied since the last call."""
        for cmd in broadcast_pyobj(None):
            if cmd[0] == "submit" and cmd[2].constraint is not None:
                cmd[2].constraint = self.grammars.recompile(cmd[2].constraint)
            self._apply(cmd)

    def _broadcast_applied(self) -> None:
        if self.tp.size > 1:
            broadcast_pyobj(self._applied)
        self._applied = []

    def _handle(self, cmd) -> None:
        kind = cmd[0]
        if kind == "submit":
            _, ids, params, ticket = cmd
            if self.tp.size > 1 and not params.greedy and params.seed is None:
                params.seed = secrets.randbits(64)  # every rank samples with the same generator
            try:
                req = self._apply(("submit", ids, params))
            except (QueueFull, InvalidRequest) as exc:
                _notify(ticket.loop, _resolve, ticket.ready, None, exc)
                return
            except Exception:
                _notify(ticket.loop, _resolve, ticket.ready, None, EngineUnavailable("engine unavailable"))
                raise
            ticket.req = req
            self.tickets[req.request_id] = ticket
            _notify(ticket.loop, _resolve, ticket.ready, True)
        elif kind == "cancel":
            ticket = cmd[1]
            if ticket.req is not None:
                self._apply(("cancel", ticket.req.request_id))
        elif kind == "flush":
            _, fut, loop = cmd
            try:
                self._apply(("flush",))
            except Exception:
                _notify(loop, _resolve, fut, None, EngineUnavailable("engine unavailable"))
                raise
            _notify(loop, _resolve, fut, True)

    def _run(self) -> None:
        # InferenceMode is thread-local. Pools/graphs may have been created
        # under it during startup; all mutations belong to this owner thread.
        with torch.inference_mode():
            self._run_inference()

    def _run_inference(self) -> None:
        try:
            while not self.stop:
                try:
                    # Idle, the primary waits for HTTP; the other ranks wait in `_publish` for its broadcast.
                    if self.tp.is_primary() and not self.sched.active:
                        self._handle(self.inbox.get(timeout=0.05))
                except queue.Empty:
                    pass
                self._publish()  # also delivers cancellation when admission is idle
                if self.stop:
                    break  # a rank above 0 received the primary's ("stop",)
                if self.sched.active:
                    self.sched.step()
                self._publish()
                self.last_stats = self.sched.stats(include_latencies=False)
                if self.sched.metrics.finished != self._latency_finished:
                    self._latencies = tuple(self.sched.metrics.ttfts)
                    self._latency_finished = self.sched.metrics.finished
                if self.drain_deadline is not None:
                    if not self.sched.active and self.inbox.empty():
                        self.stop = True
                    elif time.monotonic() >= self.drain_deadline:
                        raise EngineUnavailable("engine drain deadline exceeded")
            if self.tp.is_primary():
                self._apply(("stop",))
                self._broadcast_applied()
        except Exception as exc:
            logging.exception("sharar engine failed")
            self._fail(exc)
        finally:
            if self.failure is None and (self.tickets or self.sched.active or not self.inbox.empty()):
                self._fail(EngineUnavailable("engine stopped"))

    def _fail(self, exc: Exception) -> None:
        with self._status_lock:
            self.failure = f"{type(exc).__name__}: {exc}"
        # Notify first: even a broken CUDA context must not strand HTTP clients.
        error = EngineUnavailable("engine unavailable")
        for ticket in self.tickets.values():
            _notify(ticket.loop, _resolve, ticket.ready, None, error)
            _notify(ticket.loop, ticket.queue.put_nowait, ([], True, "error"))
        self.tickets.clear()
        while True:
            try:
                cmd = self.inbox.get_nowait()
            except queue.Empty:
                break
            if cmd[0] == "submit":
                ticket = cmd[3]
                _notify(ticket.loop, _resolve, ticket.ready, None, error)
            elif cmd[0] == "flush":
                _notify(cmd[2], _resolve, cmd[1], None, error)
        for requests in (self.sched.queue, self.sched.prefilling, self.sched.running):
            for req in requests:
                req.cacheable = False  # a failed forward may have partially mutated state
                req._finish("error")
                try:
                    if req.slot >= 0 or req.pages is not None or req.match is not None:
                        self.sched.ex.release(req)
                    else:
                        req.released = True
                except Exception:
                    logging.exception("resource cleanup failed for request %s", req.request_id)
            requests.clear()
        self.last_stats = self.sched.stats()
        if self.tp.size > 1:
            os._exit(1)

    def _publish(self) -> None:
        # Hand over requests that arrived during the step, then deliver tokens.
        self._recv_commands()
        done = []
        for rid, t in self.tickets.items():
            new = t.req.new_tokens()
            if t.req.finished:
                done.append((rid, t, new))
            elif new:
                _notify(t.loop, t.queue.put_nowait, (new, False, t.req.finish_reason))
        for rid, t, new in done:
            # The HTTP loop may consume the final event immediately, so retire
            # its ticket before handing completion across the thread boundary.
            self.tickets.pop(rid, None)
            _notify(t.loop, t.queue.put_nowait, (new, True, t.req.finish_reason))
