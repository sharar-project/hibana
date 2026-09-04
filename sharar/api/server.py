"""HTTP server: OpenAI-compatible chat/completions plus the SGLang-style
/generate the benchmark harness speaks.

The scheduler runs in one background thread (`EngineLoop`); request handlers
enqueue work and await per-request asyncio queues that the loop fills after
every step. All GPU work stays on the engine thread.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import uuid
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
from sharar.engine.sampler import SamplingParams
from sharar.engine.scheduler import QueueFull, Scheduler
from sharar.tokenizer import SnapshotTokenizer

MODEL_ID = config.MODEL_ID


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
        self.last_stats: dict = {}
        self.thread = threading.Thread(target=self._run, name="sharar-engine", daemon=True)

    def start(self) -> None:
        self.thread.start()

    async def submit(self, prompt_ids: list[int], params: SamplingParams) -> Ticket:
        loop = asyncio.get_running_loop()
        ticket = Ticket(queue=asyncio.Queue(), loop=loop, ready=loop.create_future())
        self.inbox.put(("submit", prompt_ids, params, ticket))
        await ticket.ready  # raises QueueFull if rejected
        return ticket

    def cancel(self, ticket: Ticket) -> None:
        self.inbox.put(("cancel", ticket))

    async def flush_cache(self) -> None:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.inbox.put(("flush", fut, loop))
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
            except QueueFull as exc:
                ticket.loop.call_soon_threadsafe(ticket.ready.set_exception, exc)
                return
            ticket.req = req
            self.tickets[req.request_id] = ticket
            ticket.loop.call_soon_threadsafe(ticket.ready.set_result, True)
        elif kind == "cancel":
            ticket = cmd[1]
            if ticket.req is not None:
                self.sched.cancel(ticket.req)
        elif kind == "flush":
            _, fut, loop = cmd
            self.sched.ex.flush_cache()
            loop.call_soon_threadsafe(fut.set_result, True)

    def _run(self) -> None:
        while not self.stop:
            # Drain commands; block briefly only when idle.
            try:
                timeout = None if self.sched.active else 0.05
                cmd = self.inbox.get(timeout=timeout) if not self.sched.active else self.inbox.get_nowait()
                self._handle(cmd)
                while True:
                    self._handle(self.inbox.get_nowait())
            except queue.Empty:
                pass
            if not self.sched.active:
                continue
            self.sched.step()  # publishes through on_progress after each sub-step
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
                t.loop.call_soon_threadsafe(t.queue.put_nowait, (new, t.req.finished, t.req.finish_reason))
            if t.req.finished:
                done.append(rid)
        for rid in done:
            self.tickets.pop(rid, None)


def _sampling_from(body: dict, max_default: int = 512) -> SamplingParams:
    sp = body.get("sampling_params") or {}
    max_new = body.get("max_tokens") or body.get("max_completion_tokens") or sp.get("max_new_tokens") or max_default
    temperature = body.get("temperature", sp.get("temperature", 0.0))
    top_p = body.get("top_p", sp.get("top_p", 0.95))
    top_k = body.get("top_k", sp.get("top_k", 20))
    seed = body.get("seed", sp.get("seed"))
    return SamplingParams(
        max_new_tokens=int(max_new),
        temperature=float(temperature) if temperature is not None else 0.0,
        top_p=float(top_p) if top_p is not None else 1.0,
        top_k=int(top_k) if top_k is not None else -1,
        seed=seed,
        ignore_eos=bool(sp.get("ignore_eos", body.get("ignore_eos", False))),
    )


def _stops(body: dict) -> list[str]:
    stop = body.get("stop") or (body.get("sampling_params") or {}).get("stop")
    if stop is None:
        return []
    return [stop] if isinstance(stop, str) else list(stop)


def create_app(engine: EngineLoop, tok: SnapshotTokenizer, tmpl: ChatTemplate, describe: dict) -> FastAPI:
    app = FastAPI(title="sharar")
    started = time.time()

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
            ids += new
            detok.update(ids)
            if detok.stopped:
                engine.cancel(ticket)
                return ids, detok.text, "stop"
            if finished:
                return ids, detok.text, reason

    # ---- plumbing ----
    @app.get("/health")
    async def health():
        return {"status": "ok", "uptime_seconds": time.time() - started}

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
        return PlainTextResponse("Cache flushed.\n")

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
            ids = [int(t) for t in body["input_ids"]]
        else:
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
            if req.spec_rounds:  # same meaning as SGLang's fields: tokens per verify round, accepted/proposed
                info["spec_verify_ct"] = req.spec_rounds
                info["spec_accept_length"] = n_out / req.spec_rounds
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
                    out += new
                    detok.update(out)
                    if detok.stopped:
                        engine.cancel(ticket)
                        finished, reason = True, "stop"
                    yield f"data: {json.dumps({'text': detok.text, 'output_ids': out, 'meta_info': meta(len(out), reason if finished else None)})}\n\n"
                    if finished:
                        break
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")
        out, text, reason = await _collect(ticket, stop, http)
        return {"text": text, "output_ids": out, "meta_info": meta(len(out), reason)}

    # ---- OpenAI completions ----
    @app.post("/v1/completions")
    async def completions(body: dict, http: HttpRequest):
        prompt = body.get("prompt", "")
        ids = [int(t) for t in prompt] if isinstance(prompt, list) and prompt and isinstance(prompt[0], int) else tok.encode(prompt if isinstance(prompt, str) else prompt[0])
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
            return StreamingResponse(events(), media_type="text/event-stream")
        out, text, reason = await _collect(ticket, stop, http)
        return {"id": cid, "object": "text_completion", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "text": text, "finish_reason": reason}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}}

    # ---- OpenAI chat ----
    @app.post("/v1/chat/completions")
    async def chat(body: dict, http: HttpRequest):
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
            return StreamingResponse(events(), media_type="text/event-stream")

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


def serve(host: str, port: int, log=print, spec: int = 0) -> None:
    import uvicorn

    from sharar.engine.executor import Executor
    from sharar.engine.scheduler import SchedulerConfig
    from sharar.model.qwen38 import Qwen38

    snap = config.snapshot_dir()
    tok = SnapshotTokenizer(snap)
    tmpl = ChatTemplate(snap)
    log(f"loading {MODEL_ID}@{config.MODEL_REVISION[:8]}")
    model = Qwen38(snap, log=log)
    ex = Executor(model, log=log)
    if spec:
        ex.enable_spec(spec, log=log)
    ex.capture_graphs(log=log)
    ex.warmup(log=log)
    cfg = SchedulerConfig()
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
        "speculative": {"algorithm": "mtp", "num_draft_tokens": spec} if spec else None,
        "attention_backend": "flashinfer",
        "gdn_backend": "triton",
        "kv_cache_dtype": "float8_e4m3fn",
        "mamba_ssm_dtype": "float32",
    }
    engine = EngineLoop(sched)
    engine.start()
    app = create_app(engine, tok, tmpl, describe)
    log(f"serving on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
