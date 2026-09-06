"""HTTP server: OpenAI-compatible chat/completions plus the SGLang-style
/generate the benchmark harness speaks.

The scheduler runs in one background thread (`EngineLoop`); request handlers
enqueue work and await per-request asyncio queues that the loop fills after
every step. All GPU work stays on the engine thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Request as HttpRequest
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from sharar import config
from sharar.api.chat_template import ChatTemplate
from sharar.api.detok import Detokenizer
from sharar.api.parsers.reasoning import ReasoningParser
from sharar.api.parsers.tools import has_tool_call, parse_tool_calls
from sharar.engine.request import Request
from sharar.engine.sampler import InvalidRequest, SamplingParams
from sharar.engine.scheduler import QueueFull, Scheduler
from sharar.tokenizer import SnapshotTokenizer

MODEL_ID = config.MODEL_ID


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
    blocked and the engine thread never waits on HTTP."""

    def __init__(self, sched: Scheduler):
        self.sched = sched
        sched.on_progress = self._publish
        self.inbox: queue.Queue = queue.Queue()
        self.tickets: dict[int, Ticket] = {}
        self.stop = False
        self.drain_deadline: float | None = None
        self.failure: str | None = None
        self._status_lock = threading.Lock()
        self.last_stats: dict = {}
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
        return self.last_stats

    # ---- engine thread ----
    def _handle(self, cmd) -> None:
        kind = cmd[0]
        if kind == "submit":
            _, ids, params, ticket = cmd
            try:
                req = self.sched.submit(ids, params)
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
                self.sched.cancel(ticket.req)
        elif kind == "flush":
            _, fut, loop = cmd
            try:
                self.sched.ex.flush_cache()
            except Exception:
                _notify(loop, _resolve, fut, None, EngineUnavailable("engine unavailable"))
                raise
            _notify(loop, _resolve, fut, True)

    def _run(self) -> None:
        import torch
        # InferenceMode is thread-local. Pools/graphs may have been created
        # under it during startup; all mutations belong to this owner thread.
        with torch.inference_mode():
            self._run_inference()

    def _run_inference(self) -> None:
        try:
            while not self.stop:
                try:
                    cmd = self.inbox.get(timeout=0.05) if not self.sched.active else self.inbox.get_nowait()
                    self._handle(cmd)
                except queue.Empty:
                    pass
                self._publish()  # also delivers cancellation when admission is idle
                if self.sched.active:
                    self.sched.step()
                self._publish()
                self.last_stats = self.sched.stats()
                if self.drain_deadline is not None:
                    if not self.sched.active and self.inbox.empty():
                        self.stop = True
                    elif time.monotonic() >= self.drain_deadline:
                        raise EngineUnavailable("engine drain deadline exceeded")
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

    def _publish(self) -> None:
        # Hand over requests that arrived during the step, then deliver tokens.
        try:
            while True:
                self._handle(self.inbox.get_nowait())
        except queue.Empty:
            pass
        done = []
        for rid, t in self.tickets.items():
            new = t.req.new_tokens()
            if new or t.req.finished:
                _notify(t.loop, t.queue.put_nowait, (new, t.req.finished, t.req.finish_reason))
            if t.req.finished:
                done.append(rid)
        for rid in done:
            self.tickets.pop(rid, None)


def _sampling_from(body: dict, max_default: int = 512) -> SamplingParams:
    sp = body.get("sampling_params")
    if sp is None:
        sp = {}
    if not isinstance(sp, dict):
        raise InvalidRequest("sampling_params must be an object")
    max_new = next((x for x in (body.get("max_tokens"), body.get("max_completion_tokens"), sp.get("max_new_tokens")) if x is not None), max_default)
    temperature = body.get("temperature", sp.get("temperature", 0.0))
    top_p = body.get("top_p", sp.get("top_p", 0.95))
    top_k = body.get("top_k", sp.get("top_k", 20))
    seed = body.get("seed", sp.get("seed"))
    # JSON has one number type. Accept finite integral floats at the HTTP
    # boundary; the engine contract stays strictly integer (and rejects bool).
    def integer(value):
        return int(value) if type(value) is float and math.isfinite(value) and value.is_integer() else value

    params = SamplingParams(
        max_new_tokens=integer(max_new),
        temperature=temperature if temperature is not None else 0.0,
        top_p=top_p if top_p is not None else 1.0,
        top_k=integer(top_k) if top_k is not None else -1,
        seed=integer(seed),
        ignore_eos=sp.get("ignore_eos", body.get("ignore_eos", False)),
    )
    params.validate()
    return params


def _stops(body: dict) -> list[str]:
    stop = body.get("stop") or (body.get("sampling_params") or {}).get("stop")
    if stop is None:
        return []
    stops = [stop] if isinstance(stop, str) else stop
    if not isinstance(stops, list) or any(not isinstance(s, str) or not s for s in stops):
        raise InvalidRequest("stop must be a non-empty string or a list of non-empty strings")
    return stops


def _validate_chat_controls(body: dict) -> None:
    """Accept only unconstrained output; fail before rendering or admission."""
    choice = body.get("tool_choice")
    if choice is not None and choice != "auto":
        raise InvalidRequest("tool_choice supports only null or 'auto'; forced or disabled tool calls are not supported")
    response_format = body.get("response_format")
    if response_format is not None and response_format != {"type": "text"}:
        raise InvalidRequest('response_format supports only null or {"type":"text"}; constrained output is not supported')


def create_app(engine: EngineLoop, tok: SnapshotTokenizer, tmpl: ChatTemplate, describe: dict) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        yield
        # Uvicorn drains HTTP connections first, then runs ASGI shutdown.
        # Its restored SIGTERM can terminate the process before the caller's
        # Python finally block executes, so engine cleanup belongs here.
        engine.drain(5)
        await asyncio.to_thread(engine.thread.join, 6)

    app = FastAPI(title="sharar", lifespan=lifespan)
    started = time.time()

    @app.exception_handler(InvalidRequest)
    async def invalid_request(http, exc):
        return JSONResponse({"error": {"message": str(exc), "type": "invalid_request_error"}}, status_code=400)

    @app.exception_handler(EngineUnavailable)
    async def engine_unavailable(http, exc):
        return JSONResponse({"error": {"message": "engine unavailable", "type": "server_error"}}, status_code=503)

    async def _stream(events, ticket):
        try:
            async for event in events:
                yield event
        finally:
            engine.cancel(ticket)

    async def _collect(ticket: Ticket, stop: list[str], http: HttpRequest | None) -> tuple[list[int], str, str]:
        """Drain a ticket to completion. Returns (ids, text, finish_reason)."""
        detok = Detokenizer(tok, stop)
        ids: list[int] = []
        reason = ""
        while True:
            if http is not None and await http.is_disconnected():
                engine.cancel(ticket)
                return ids, detok.text, "cancelled"
            try:
                new, finished, reason = await asyncio.wait_for(ticket.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                engine.cancel(ticket)
                raise
            if reason == "error":
                raise EngineUnavailable("engine unavailable")
            ids += new
            detok.update(ids)
            if detok.stopped:
                engine.cancel(ticket)
                return ids, detok.text, "stop"
            if finished:
                return ids, detok.text, reason

    # ---- plumbing ----
    @app.get("/health")
    @app.get("/ready")
    async def health():
        return JSONResponse({"status": "ok" if engine.healthy else "unavailable", "uptime_seconds": time.time() - started}, status_code=200 if engine.healthy else 503)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "sharar", "root": MODEL_ID}]}

    @app.get("/get_server_info")
    @app.get("/describe")
    async def server_info():
        info = {**describe, "scheduler": engine.stats()}
        if engine.sched.ex.spec is not None:
            info["spec_stats"] = engine.sched.ex.spec.stats()
        return info

    @app.get("/metrics")
    async def metrics():
        st = engine.stats()
        lines = [f"sharar_{k} {v}" for k, v in st.items() if isinstance(v, (int, float))]
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.post("/flush_cache")
    async def flush_cache():
        await engine.flush_cache()
        return PlainTextResponse("Cache invalidated; active readers release pinned entries on completion.\n")

    @app.post("/v1/tokenize")
    @app.post("/tokenize")
    async def tokenize(body: dict):
        text = body.get("prompt") if body.get("prompt") is not None else body.get("text", "")
        ids = tok.encode(text)
        return {"tokens": ids, "count": len(ids), "max_model_len": config.CONTEXT_LENGTH}

    # ---- /generate (SGLang style) ----
    @app.post("/generate")
    async def generate(body: dict, http: HttpRequest):
        if body.get("input_ids") is not None:
            ids = body["input_ids"]
        else:
            if not isinstance(body.get("text", ""), str):
                raise InvalidRequest("text must be a string")
            ids = tok.encode(body.get("text", ""))
        if not ids:
            raise HTTPException(400, "empty prompt")
        params = _sampling_from(body, max_default=128)
        stop = _stops(body)
        try:
            ticket = await engine.submit(ids, params)
        except QueueFull as exc:
            raise HTTPException(429, str(exc))
        t0 = time.perf_counter()

        def meta(n_out: int, reason: str | None) -> dict:
            req = ticket.req
            info = {
                "id": body.get("rid") or uuid.uuid4().hex,
                "prompt_tokens": len(ids),
                "completion_tokens": n_out,
                "cached_tokens": req.prefix_len,
                "finish_reason": {"type": reason} if reason else None,
                "e2e_latency": time.perf_counter() - t0,
            }
            if req.spec_rounds:
                info["spec_verify_ct"] = req.spec_rounds
                # SGLang's per-request field includes the prefill output.
                info["spec_accept_length"] = n_out / req.spec_rounds
                info["spec_tokens_per_round"] = req.spec_emitted / req.spec_rounds
                info["spec_accept_rate"] = req.spec_accepted / max(1, req.spec_proposed)
            return info

        if body.get("stream"):
            async def events():
                detok = Detokenizer(tok, stop)
                out: list[int] = []
                while True:
                    if await http.is_disconnected():
                        engine.cancel(ticket)
                        break
                    try:
                        new, finished, reason = await asyncio.wait_for(ticket.queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    if reason == "error":
                        yield 'data: {"error": {"message": "engine unavailable", "type": "server_error"}}\n\n'
                        break
                    out += new
                    detok.update(out)
                    if detok.stopped:
                        engine.cancel(ticket)
                        finished, reason = True, "stop"
                    yield f"data: {json.dumps({'text': detok.text, 'output_ids': out, 'meta_info': meta(len(out), reason if finished else None)})}\n\n"
                    if finished:
                        break
                yield "data: [DONE]\n\n"
            return StreamingResponse(_stream(events(), ticket), media_type="text/event-stream")
        out, text, reason = await _collect(ticket, stop, http)
        return {"text": text, "output_ids": out, "meta_info": meta(len(out), reason)}

    # ---- OpenAI completions ----
    @app.post("/v1/completions")
    async def completions(body: dict, http: HttpRequest):
        prompt = body.get("prompt", "")
        if isinstance(prompt, str):
            ids = tok.encode(prompt)
        elif isinstance(prompt, list) and all(type(t) is int for t in prompt):
            ids = prompt
        else:
            raise InvalidRequest("prompt must be a string or one token ID list")
        params = _sampling_from(body, max_default=16)
        stop = _stops(body)
        try:
            ticket = await engine.submit(ids, params)
        except QueueFull as exc:
            raise HTTPException(429, str(exc))
        cid = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        if body.get("stream"):
            async def events():
                detok = Detokenizer(tok, stop)
                out: list[int] = []
                while True:
                    if await http.is_disconnected():
                        engine.cancel(ticket)
                        break
                    try:
                        new, finished, reason = await asyncio.wait_for(ticket.queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    if reason == "error":
                        yield 'data: {"error": {"message": "engine unavailable", "type": "server_error"}}\n\n'
                        break
                    out += new
                    delta = detok.update(out)
                    if detok.stopped:
                        engine.cancel(ticket)
                        finished, reason = True, "stop"
                    chunk = {"id": cid, "object": "text_completion", "created": created, "model": MODEL_ID,
                             "choices": [{"index": 0, "text": delta, "finish_reason": reason if finished else None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    if finished:
                        break
                yield "data: [DONE]\n\n"
            return StreamingResponse(_stream(events(), ticket), media_type="text/event-stream")
        out, text, reason = await _collect(ticket, stop, http)
        return {"id": cid, "object": "text_completion", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "text": text, "finish_reason": reason}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}}

    # ---- OpenAI chat ----
    @app.post("/v1/chat/completions")
    async def chat(body: dict, http: HttpRequest):
        _validate_chat_controls(body)
        messages = body.get("messages") or []
        if not messages:
            raise HTTPException(400, "messages required")
        kwargs = dict(body.get("chat_template_kwargs") or {})
        thinking = bool(kwargs.pop("enable_thinking", True))
        tools = body.get("tools")
        rendered = tmpl.render(messages, enable_thinking=thinking, tools=tools, **kwargs)
        ids = tok.encode(rendered)
        params = _sampling_from(body, max_default=512)
        stop = _stops(body)
        try:
            ticket = await engine.submit(ids, params)
        except QueueFull as exc:
            raise HTTPException(429, str(exc))
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        in_reasoning = thinking and rendered.rstrip().endswith("<think>")
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        def usage(n_out: int, reasoning_tokens: int | None = None) -> dict:
            u = {"prompt_tokens": len(ids), "completion_tokens": n_out, "total_tokens": len(ids) + n_out}
            if reasoning_tokens is not None:
                u["reasoning_tokens"] = reasoning_tokens
            return u

        if body.get("stream"):
            async def events():
                detok = Detokenizer(tok, stop)
                rp = ReasoningParser(in_reasoning)
                out: list[int] = []
                buffered_tool = ""
                sent_role = False
                while True:
                    if await http.is_disconnected():
                        engine.cancel(ticket)
                        break
                    try:
                        new, finished, reason = await asyncio.wait_for(ticket.queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    if reason == "error":
                        yield 'data: {"error": {"message": "engine unavailable", "type": "server_error"}}\n\n'
                        break
                    out += new
                    delta_text = detok.update(out)
                    if detok.stopped:
                        engine.cancel(ticket)
                        finished, reason = True, "stop"
                    r_delta, c_delta = rp.feed(delta_text)
                    if finished:
                        r2, c2 = rp.finish()
                        r_delta += r2
                        c_delta += c2
                    delta: dict[str, Any] = {}
                    if not sent_role:
                        delta["role"] = "assistant"
                        sent_role = True
                    if r_delta:
                        delta["reasoning_content"] = r_delta
                    if c_delta:
                        if tools and (buffered_tool or has_tool_call(rp.content)):
                            buffered_tool = rp.content
                        else:
                            delta["content"] = c_delta
                    finish = None
                    if finished:
                        finish = reason
                        if buffered_tool:
                            content, calls = parse_tool_calls(buffered_tool, tools)
                            if calls:
                                delta["tool_calls"] = [{"index": i, **c} for i, c in enumerate(calls)]
                                finish = "tool_calls"
                    if delta or finished:
                        chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    if finished:
                        if include_usage:
                            yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [], 'usage': usage(len(out))})}\n\n"
                        break
                yield "data: [DONE]\n\n"
            return StreamingResponse(_stream(events(), ticket), media_type="text/event-stream")

        out, text, reason = await _collect(ticket, stop, http)
        rp = ReasoningParser(in_reasoning)
        rp.feed(text)
        rp.finish()
        content, calls = (parse_tool_calls(rp.content, tools) if tools else (rp.content, []))
        message: dict[str, Any] = {"role": "assistant", "content": content if content or not calls else None}
        if rp.reasoning:
            message["reasoning_content"] = rp.reasoning
        if calls:
            message["tool_calls"] = calls
            reason = "tool_calls"
        reasoning_tokens = len(tok.encode(rp.reasoning)) if rp.reasoning else 0
        return {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                "usage": usage(len(out), reasoning_tokens)}

    return app


def serve(host: str, port: int, log=print, spec: int = 0, proposer: str = "mtp", state_strategy: str = "snapshots") -> None:
    import uvicorn

    from sharar.engine.executor import Executor
    from sharar.engine.scheduler import SchedulerConfig
    from sharar.model.qwen38 import Qwen38
    from sharar.memory.spec_budget import SpecConfig

    snap = config.snapshot_dir()
    tok = SnapshotTokenizer(snap)
    tmpl = ChatTemplate(snap)
    log(f"loading {MODEL_ID}@{config.MODEL_REVISION[:8]}")
    model = Qwen38(snap, log=log)
    ex = Executor(model, log=log, spec_config=SpecConfig(proposer, spec, state_strategy) if spec else None)
    ex.capture_graphs(log=log)
    ex.warmup(log=log)
    cfg = SchedulerConfig(max_running=ex.profile.max_batch, max_queued=ex.profile.max_queued,
                          chunk_size=ex.profile.chunk_size, prefill_budget=ex.profile.prefill_budget)
    sched = Scheduler(ex, cfg)
    describe = {
        "engine": "sharar",
        "model_path": MODEL_ID,
        "revision": config.MODEL_REVISION,
        "context_length": config.CONTEXT_LENGTH,
        "max_running_requests": cfg.max_running,
        "max_queued_requests": cfg.max_queued,
        "chunked_prefill_size": cfg.chunk_size,
        "prefill_budget": cfg.prefill_budget,
        "kv_pages": ex.kv_pool.num_pages,
        "state_slots": ex.state_pool.num_slots,
        "ledger_gib": {k: round(v / 2**30, 2) for k, v in ex.ledger.entries.items()},
        "speculative": {"algorithm": ex.spec.proposer.requirements.name, "num_draft_tokens": spec} if spec else None,
        "attention_backend": "flashinfer",
        "gdn_backend": "triton",
        "kv_cache_dtype": "float8_e4m3fn",
        "mamba_ssm_dtype": "float32",
    }
    engine = EngineLoop(sched)
    engine.start()
    app = create_app(engine, tok, tmpl, describe)
    log(f"serving on http://{host}:{port}")
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning", timeout_graceful_shutdown=30)
    finally:
        engine.drain(timeout=5)
        engine.thread.join(timeout=6)
