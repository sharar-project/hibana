"""HTTP server: OpenAI-compatible chat/completions plus the SGLang-style
/generate the benchmark harness speaks.

The scheduler runs in one background thread (`EngineLoop`); request handlers
enqueue work and await per-request asyncio queues that the loop fills after
every step. All GPU work stays on the engine thread.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request as HttpRequest
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from sharar import config
from sharar.api.chat_template import ChatTemplate
from sharar.api.detok import Detokenizer
from sharar.api.parsers.reasoning import ReasoningParser
from sharar.api.parsers.tools import has_tool_call, parse_tool_calls
from sharar.api.validation import _sampling_from, _stops, _validate_chat_controls
from sharar.engine.loop import EngineUnavailable, Ticket, EngineLoop
from sharar.engine.sampler import InvalidRequest
from sharar.engine.scheduler import QueueFull, Scheduler
from sharar.tokenizer import SnapshotTokenizer

MODEL_ID = config.MODEL_ID


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
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        return_token_ids = bool(body.get("return_token_ids"))

        def usage(n_out: int) -> dict:
            return {"prompt_tokens": len(ids), "completion_tokens": n_out,
                    "total_tokens": len(ids) + n_out,
                    "prompt_tokens_details": {"cached_tokens": ticket.req.prefix_len}}

        if body.get("stream"):
            async def events():
                detok = Detokenizer(tok, stop)
                out: list[int] = []
                sent_prompt_ids = False
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
                    if return_token_ids:
                        chunk["choices"][0]["token_ids"] = new
                        if not sent_prompt_ids:
                            chunk["choices"][0]["prompt_token_ids"] = ids
                            sent_prompt_ids = True
                    yield f"data: {json.dumps(chunk)}\n\n"
                    if finished:
                        if include_usage:
                            yield f"data: {json.dumps({'id': cid, 'object': 'text_completion', 'created': created, 'model': MODEL_ID, 'choices': [], 'usage': usage(len(out))})}\n\n"
                        break
                yield "data: [DONE]\n\n"
            return StreamingResponse(_stream(events(), ticket), media_type="text/event-stream")
        out, text, reason = await _collect(ticket, stop, http)
        choice = {"index": 0, "text": text, "finish_reason": reason}
        if return_token_ids:
            choice.update(token_ids=out, prompt_token_ids=ids)
        return {"id": cid, "object": "text_completion", "created": created, "model": MODEL_ID,
                "choices": [choice], "usage": usage(len(out))}

    # ---- OpenAI chat ----
    @app.post("/v1/chat/completions")
    async def chat(body: dict, http: HttpRequest):
        _validate_chat_controls(body)
        messages = body.get("messages") or []
        if not messages:
            raise HTTPException(400, "messages required")
        kwargs = dict(body.get("chat_template_kwargs") or {})
        thinking = bool(kwargs.pop("enable_thinking", True))
        tools = None if body.get("tool_choice") == "none" else body.get("tools")  # "none": generate without tool schemas
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


def serve(host: str, port: int, log=print, spec: int = 0, proposer: str = "mtp", state_strategy: str = "snapshots", draft_precision: str = 'bf16',
          traces: str | None = None, traces_max_mib: int = 32768, draft_dir: str | None = None) -> None:
    from sharar.memory.spec_budget import SpecConfig

    if not spec and draft_precision != 'bf16':
        raise ValueError('draft precision requires speculation')
    if draft_dir is not None and not spec:
        raise ValueError('a draft directory requires speculation')
    spec_config = SpecConfig(proposer, spec, state_strategy, draft_precision, draft_dir) if spec else None
    if spec_config is not None:
        from sharar.proposers.factory import draft_snapshot
        if proposer in ('dspark', 'dflash2'):
            draft_snapshot(proposer, draft_dir)  # fail before loading the target
    import uvicorn

    from sharar.engine.executor import Executor
    from sharar.engine.scheduler import SchedulerConfig
    from sharar.model.qwen38 import Qwen38

    snap = config.snapshot_dir()
    tok = SnapshotTokenizer(snap)
    tmpl = ChatTemplate(snap)
    log(f"loading {MODEL_ID}@{config.MODEL_REVISION[:8]}")
    model = Qwen38(snap, log=log)
    ex = Executor(model, log=log, spec_config=spec_config)
    ex.capture_graphs(log=log)
    ex.warmup(log=log)
    ex.enable_cold_tier(log=log)
    cfg = SchedulerConfig(max_running=ex.profile.max_batch, max_queued=ex.profile.max_queued,
                          chunk_size=ex.profile.chunk_size, prefill_budget=ex.profile.prefill_budget)
    sched = Scheduler(ex, cfg)
    sink = None
    if traces:
        from sharar.engine.traces import TraceSink

        sink = TraceSink(traces, max_bytes=traces_max_mib * 2**20)
        sched.sink = sink
        log(f"recording opt-in request traces under {sink.directory} (cap {traces_max_mib} MiB)")
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
        "speculative": {"algorithm": ex.spec.proposer.requirements.name, "num_draft_tokens": spec,
                        "draft_precision": draft_precision, "draft": getattr(ex.spec.proposer.model, "source", None)
                        if hasattr(ex.spec.proposer, "model") else None} if spec else None,
        "attention_backend": "flashinfer",
        "gdn_backend": "triton",
        "kv_cache_dtype": "float8_e4m3fn",
        "mamba_ssm_dtype": "float32",
        "traces": {"dir": str(sink.directory), "max_mib": traces_max_mib} if sink is not None else None,
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
        if sink is not None:
            sink.close()
