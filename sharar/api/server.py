"""HTTP server: OpenAI-compatible chat/completions, the Anthropic Messages
API on the same chat path, and the SGLang-style /generate the benchmark
harness speaks.

The scheduler runs in one background thread (`EngineLoop`); request handlers
enqueue work and await per-request asyncio queues that the loop fills after
every step. All GPU work stays on the engine thread.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request as HttpRequest
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from jinja2 import TemplateError

from sharar import target as targets
from sharar.distributed import get_tp_info
from sharar.api import messages as messages_api
from sharar.api import refusal_guard
from sharar.api.chat_template import ChatTemplate
from sharar.api.detok import Detokenizer
from sharar.api.parsers.reasoning import ReasoningParser
from sharar.api.parsers.tools import TOOL_CALL_START, has_tool_call, parse_tool_calls
from sharar.api.validation import _output_spec, _sampling_from, _stops, _template_kwargs
from sharar.engine.loop import EngineUnavailable, Ticket, EngineLoop
from sharar.engine.grammar import Grammars
from sharar.engine.sampler import InvalidRequest, SamplingParams
from sharar.engine.scheduler import QueueFull, Scheduler
from sharar.tokenizer import SnapshotTokenizer



MAX_CHOICES = 16  # n per chat request: the scheduler queues what exceeds four running


def create_app(engine: EngineLoop, tok: SnapshotTokenizer, tmpl: ChatTemplate, describe: dict, grammars: Grammars | None = None) -> FastAPI:
    MODEL_ID = describe.get("model_path", targets.DEFAULT_TARGET.model_id)  # the served checkpoint id, reported as the model name
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
    # Messages API requests this server declines or reduces, to see what harnesses send.
    messages_stats = {"forced_tool_choice_rejected": 0, "server_tools_ignored": 0}
    # Requests with constrained output (#16), by kind.
    constrained_stats = {"json_schema": 0, "json_object": 0, "tool_call": 0}
    # Turns the refusal guard changed (#108): a reminder added after repeated refusals, or the turn ended by the server.
    guard_stats = {"refusal_reminders": 0, "refusal_stops": 0}

    def _guard(messages: list[dict], body: dict) -> tuple[list[dict], refusal_guard.Refusals | None]:
        """The messages to render (with a reminder after repeated refusals) and, when the server ends the turn
        itself, the refusals that made it (api/refusal_guard.py)."""
        action, found = refusal_guard.decide(messages, body)
        if action == "nudge":
            guard_stats["refusal_reminders"] += 1
            return [*messages, refusal_guard.reminder(found)], None
        if action == "stop":
            guard_stats["refusal_stops"] += 1
            return messages, found
        return messages, None
    grammar_builder = [grammars]  # made from the tokenizer on the first constrained request

    def _constraint(spec: tuple[str, object], tools: list | None, in_reasoning: bool):
        """The request's Constraint (engine/grammar.py); compiling runs off the event loop."""
        if grammar_builder[0] is None:
            grammar_builder[0] = Grammars.from_tokenizer(tok._tok, describe.get("vocab_size", targets.DEFAULT_TARGET.vocab_size),
                                                         SamplingParams().stop_token_ids)
        g, (kind, arg), armed = grammar_builder[0], spec, not in_reasoning
        try:
            constraint = (g.tool_call(tools or [], arg, armed) if kind == "tool_call" else
                          g.json_object(armed) if kind == "json_object" else g.json_schema(arg, armed))
        except ValueError as exc:
            raise InvalidRequest(str(exc))
        constrained_stats[kind] += 1
        return constraint

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

    async def _collect(ticket: Ticket, detok: Detokenizer, http: HttpRequest | None) -> tuple[list[int], str, str]:
        """Drain a ticket to completion. Returns (ids, text, finish_reason)."""
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

    def _chat_prompt(messages: list[dict], tools: list | None, thinking: bool, kwargs: dict) -> tuple[list[int], bool]:
        """Render and encode a chat. Returns (ids, whether the output starts inside a thinking block)."""
        rendered = tmpl.render(fold_system_messages(_tool_arguments_as_objects(messages)), enable_thinking=thinking, tools=tools, **kwargs)
        return tok.encode(rendered), thinking and rendered.rstrip().endswith("<think>")

    async def _submit(ids: list[int], params) -> Ticket:
        try:
            return await engine.submit(ids, params)
        except QueueFull as exc:
            raise HTTPException(429, str(exc))

    async def _chat_steps(ticket: Ticket, detok: Detokenizer, http: HttpRequest, in_reasoning: bool, tools: list | None):
        """Streamed chat output, one step per engine update: yields
        (reasoning_delta, content_delta, tool_calls, finished, reason, n_out).
        Content is held back once a tool call starts; the final step carries
        the parsed calls, with reason "tool_calls" when there are any. Returns
        early when the client disconnects; raises EngineUnavailable on engine failure."""
        rp = ReasoningParser(in_reasoning, tools=bool(tools))
        out: list[int] = []
        buffered_tool = ""
        while True:
            if await http.is_disconnected():
                engine.cancel(ticket)
                return
            try:
                new, finished, reason = await asyncio.wait_for(ticket.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if reason == "error":
                raise EngineUnavailable("engine unavailable")
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
            if c_delta and tools and (buffered_tool or has_tool_call(rp.content)):
                # Content before a call that starts within this update is still content.
                keep = 0 if buffered_tool else max(0, rp.content.find(TOOL_CALL_START) - (len(rp.content) - len(c_delta)))
                buffered_tool = rp.content
                c_delta = c_delta[:keep]
            calls: list[dict] = []
            if finished and buffered_tool:
                _, calls = parse_tool_calls(buffered_tool, tools)
                if calls:
                    reason = "tool_calls"
            yield r_delta, c_delta, calls, finished, reason, len(out)
            if finished:
                return

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
        from sharar.engine.config import current
        info = {**describe, "config": current().describe(), "scheduler": engine.stats()}
        if engine.sched.ex.spec is not None:
            info["spec_stats"] = engine.sched.ex.spec.stats()
        info["messages_api"] = dict(messages_stats)
        info["constrained"] = dict(constrained_stats)
        info["refusal_guard"] = dict(guard_stats)
        return info

    @app.get("/metrics")
    async def metrics():
        st = engine.stats()
        lines = [f"sharar_{k} {v}" for k, v in st.items() if isinstance(v, (int, float))]
        lines += [f"sharar_messages_{k} {v}" for k, v in messages_stats.items()]
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
        return {"tokens": ids, "count": len(ids), "max_model_len": describe.get("context_length", targets.DEFAULT_TARGET.context_length)}

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
                if req.params.relaxed:
                    info["spec_relaxed_tokens"] = req.spec_relaxed
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
        out, text, reason = await _collect(ticket, Detokenizer(tok, stop), http)
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
        out, text, reason = await _collect(ticket, Detokenizer(tok, stop), http)
        choice = {"index": 0, "text": text, "finish_reason": reason}
        if return_token_ids:
            choice.update(token_ids=out, prompt_token_ids=ids)
        return {"id": cid, "object": "text_completion", "created": created, "model": MODEL_ID,
                "choices": [choice], "usage": usage(len(out))}

    # ---- OpenAI chat ----
    @app.post("/v1/chat/completions")
    async def chat(body: dict, http: HttpRequest):
        spec = _output_spec(body)
        messages = body.get("messages") or []
        if not messages:
            raise HTTPException(400, "messages required")
        kwargs, thinking = _template_kwargs(body)
        tools = None if body.get("tool_choice") == "none" else body.get("tools")  # "none": generate without tool schemas
        messages, stopped = _guard(messages, body)
        if stopped is not None:
            return _chat_stopped(body, refusal_guard.stop_message(stopped))
        ids, in_reasoning = _chat_prompt(messages, tools, thinking, kwargs)
        # Without max_tokens (or max_completion_tokens) a reply may run to the end of the context window, as in the
        # OpenAI API, vLLM and SGLang: a fixed default cut off clients that leave it unset (Aider) mid-thought.
        limit = describe.get("context_length", targets.DEFAULT_TARGET.context_length)
        if "temperature" not in body and "temperature" not in (body.get("sampling_params") or {}):
            # OpenAI's default temperature, and both targets' generation_config (top_p 0.95 and top_k 20 are the
            # defaults). OpenCode sends no sampling fields; greedy thinking looped to max_tokens in its runs (#28).
            body = {**body, "temperature": 1.0}
        params = _sampling_from(body, max_default=max(1, limit - len(ids)))
        if spec is not None:
            params.constraint = await asyncio.to_thread(_constraint, spec, tools, in_reasoning)
        stop = _stops(body)
        n = body.get("n", 1)
        if type(n) is not int or not 1 <= n <= MAX_CHOICES:
            raise InvalidRequest(f"n must be an integer in [1, {MAX_CHOICES}]")
        if n > 1:
            if body.get("stream"):
                raise InvalidRequest("n > 1 requires stream: false")
            return await _chat_choices(ids, params, stop, in_reasoning, tools, n, http)
        ticket = await _submit(ids, params)
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        def usage(n_out: int, reasoning_tokens: int | None = None) -> dict:
            u = {"prompt_tokens": len(ids), "completion_tokens": n_out, "total_tokens": len(ids) + n_out,
                 "prompt_tokens_details": {"cached_tokens": ticket.req.prefix_len}}
            if reasoning_tokens is not None:
                u["reasoning_tokens"] = reasoning_tokens
            return u

        if body.get("stream"):
            async def events():
                sent_role = False
                try:
                    async for r_delta, c_delta, calls, finished, reason, n_out in _chat_steps(
                            ticket, Detokenizer(tok, stop), http, in_reasoning, tools):
                        delta: dict[str, Any] = {}
                        if not sent_role:
                            delta["role"] = "assistant"
                            sent_role = True
                        if r_delta:
                            delta["reasoning_content"] = r_delta
                        if c_delta:
                            delta["content"] = c_delta
                        if calls:
                            delta["tool_calls"] = [{"index": i, **c} for i, c in enumerate(calls)]
                        if delta or finished:
                            chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                                     "choices": [{"index": 0, "delta": delta, "finish_reason": reason if finished else None}]}
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        if finished and include_usage:
                            yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [], 'usage': usage(n_out)})}\n\n"
                except EngineUnavailable:
                    yield 'data: {"error": {"message": "engine unavailable", "type": "server_error"}}\n\n'
                yield "data: [DONE]\n\n"
            return StreamingResponse(_stream(events(), ticket), media_type="text/event-stream")

        out, text, reason = await _collect(ticket, Detokenizer(tok, stop), http)
        reasoning, content, calls = _split_chat(text, in_reasoning, tools)
        message: dict[str, Any] = {"role": "assistant", "content": content if content or not calls else None}
        if reasoning:
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = calls
            reason = "tool_calls"
        reasoning_tokens = len(tok.encode(reasoning)) if reasoning else 0
        return {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                "usage": usage(len(out), reasoning_tokens)}

    def _chat_stopped(body: dict, text: str):
        """A chat turn the server ends itself (the refusal guard): the text as the reply, no generation."""
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if body.get("stream"):
            def chunk(delta, finish):
                return f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]})}\n\n"
            events = [chunk({"role": "assistant", "content": text}, None), chunk({}, "stop")]
            if (body.get("stream_options") or {}).get("include_usage"):
                events.append(f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [], 'usage': usage})}\n\n")
            return StreamingResponse(iter(events + ["data: [DONE]\n\n"]), media_type="text/event-stream")
        return {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": usage}

    async def _chat_choices(ids, params, stop, in_reasoning, tools, n, http):
        """n choices from one prefill. A one-token primer prefills the prompt
        and, when the engine reaps it at the start of its next step, publishes
        it to the prefix cache; the n samples are admitted after that reap, so
        each resumes from the cached prompt (at most 64 tokens re-prefilled)
        and they decode side by side. Seeds are seed + i when a seed is given."""
        primer = await _submit(ids, dataclasses.replace(params, max_new_tokens=1, trace=False, seed=None))
        await _collect(primer, Detokenizer(tok, []), http)
        base = params.seed
        tickets = [await _submit(ids, dataclasses.replace(params, seed=None if base is None else base + i)) for i in range(n)]
        results = await asyncio.gather(*[_collect(t, Detokenizer(tok, stop), http) for t in tickets])
        choices, completion, reasoning_total = [], 0, 0
        for i, (out, text, reason) in enumerate(results):
            reasoning, content, calls = _split_chat(text, in_reasoning, tools)
            message: dict[str, Any] = {"role": "assistant", "content": content if content or not calls else None}
            if reasoning:
                message["reasoning_content"] = reasoning
                reasoning_total += len(tok.encode(reasoning))
            if calls:
                message["tool_calls"] = calls
                reason = "tool_calls"
            choices.append({"index": i, "message": message, "finish_reason": reason})
            completion += len(out)
        usage = {"prompt_tokens": len(ids), "completion_tokens": completion, "total_tokens": len(ids) + completion,
                 "prompt_tokens_details": {"cached_tokens": primer.req.prefix_len}, "reasoning_tokens": reasoning_total}
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
                "model": MODEL_ID, "choices": choices, "usage": usage}

    # ---- Anthropic Messages ----
    async def _messages_errors(handler, http: HttpRequest):
        """Run a Messages handler, reporting failures in Anthropic's error shape."""
        def error(status: int, kind: str, message: str) -> JSONResponse:
            return JSONResponse(messages_api.error_body(kind, message), status_code=status)
        try:
            body = await http.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return error(400, "invalid_request_error", "request body must be a JSON object")
        try:
            return await handler(body, http)
        except InvalidRequest as exc:
            return error(400, "invalid_request_error", str(exc))
        except HTTPException as exc:
            return error(exc.status_code, "rate_limit_error" if exc.status_code == 429 else "invalid_request_error", str(exc.detail))
        except EngineUnavailable:
            return error(503, "overloaded_error", "engine unavailable")

    def _messages_prompt(body: dict, choice: str) -> tuple[messages_api.MessagesRequest, list[dict] | None, list[int], bool]:
        req = messages_api.parse_request(body)
        req.messages, req.stopped = _guard(req.messages, body)
        kwargs, thinking = _template_kwargs(body)
        if req.thinking is not None:
            thinking = req.thinking
        tools = None if choice == "none" else req.tools
        try:
            ids, in_reasoning = _chat_prompt(req.messages, tools, thinking, kwargs)
        except TemplateError as exc:
            raise InvalidRequest(f"the conversation cannot be rendered: {exc}")
        return req, tools, ids, in_reasoning

    @app.post("/v1/messages")
    async def messages(http: HttpRequest):
        return await _messages_errors(_messages, http)

    async def _messages(body: dict, http: HttpRequest):
        choice = messages_api.tool_choice(body)
        forced = None
        if choice in ("any", "tool"):
            name = body["tool_choice"].get("name") if choice == "tool" else None
            if choice == "tool" and (not isinstance(name, str) or not name):
                raise InvalidRequest("tool_choice of type 'tool' needs a name")
            forced = ("tool_call", name)
        model = body.get("model", MODEL_ID)
        if not isinstance(model, str):
            raise InvalidRequest("model must be a string")
        max_tokens = body.get("max_tokens")
        if max_tokens is None:
            raise InvalidRequest("max_tokens is required")
        req, tools, ids, in_reasoning = _messages_prompt(body, choice)
        messages_stats["server_tools_ignored"] += req.ignored_tools
        if req.stopped is not None:
            return _messages_stopped(body, model, refusal_guard.stop_message(req.stopped))
        limit = describe.get("context_length", targets.DEFAULT_TARGET.context_length)
        if len(ids) >= limit:
            raise InvalidRequest(f"prompt is too long: {len(ids)} tokens > {limit - 1} maximum")
        if isinstance(max_tokens, (int, float)) and not isinstance(max_tokens, bool):
            max_tokens = min(max_tokens, limit - len(ids))  # clients size max_tokens for a larger window
        sampling = {k: body[k] for k in ("temperature", "top_p", "top_k", "trace", "accept_top_k", "accept_margin") if k in body}
        # Anthropic's default temperature, and both targets' generation_config (top_p 0.95 and top_k 20 are the
        # chat defaults). Claude Code sends no sampling fields; greedy thinking looped to max_tokens in its sessions (#11).
        sampling.setdefault("temperature", 1.0)
        params = _sampling_from({**sampling, "max_tokens": max_tokens})
        if forced is not None:
            if not tools:
                raise InvalidRequest("a tool_choice that requires a call needs tools")
            params.constraint = await asyncio.to_thread(_constraint, forced, tools, in_reasoning)
        stop = _stops({"stop": body.get("stop_sequences")})
        ticket = await _submit(ids, params)
        mid = messages_api.new_id("msg")

        def usage(n_out: int) -> dict:
            cached = ticket.req.prefix_len
            return {"input_tokens": len(ids) - cached, "output_tokens": n_out,
                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached}

        detok = Detokenizer(tok, stop)
        if body.get("stream"):
            sse = messages_api.sse

            async def events():
                # The prefix match is not known before admission; message_delta reports the final split.
                yield sse("message_start", {"type": "message_start", "message": {
                    "id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
                    "stop_reason": None, "stop_sequence": None, "usage": {
                        "input_tokens": len(ids), "output_tokens": 0,
                        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}})
                yield sse("ping", {"type": "ping"})
                writer = messages_api.EventWriter()
                try:
                    async for r_delta, c_delta, calls, finished, reason, n_out in _chat_steps(
                            ticket, detok, http, in_reasoning, tools):
                        for event in writer.delta("thinking", r_delta) + writer.delta("text", c_delta):
                            yield event
                        if finished:
                            for event in writer.finish(calls):
                                yield event
                            stop_reason, stop_sequence = messages_api.stop_reason(reason, detok.stop_hit)
                            yield sse("message_delta", {"type": "message_delta", "delta": {
                                "stop_reason": stop_reason, "stop_sequence": stop_sequence}, "usage": usage(n_out)})
                            yield sse("message_stop", {"type": "message_stop"})
                except EngineUnavailable:
                    yield sse("error", messages_api.error_body("overloaded_error", "engine unavailable"))
            return StreamingResponse(_stream(events(), ticket), media_type="text/event-stream")

        out, text, reason = await _collect(ticket, detok, http)
        reasoning, content, calls = _split_chat(text, in_reasoning, tools)
        stop_reason, stop_sequence = messages_api.stop_reason("tool_calls" if calls else reason, detok.stop_hit)
        return {"id": mid, "type": "message", "role": "assistant", "model": model,
                "content": messages_api.content_blocks(reasoning, content, calls),
                "stop_reason": stop_reason, "stop_sequence": stop_sequence, "usage": usage(len(out))}

    def _messages_stopped(body: dict, model: str, text: str):
        """A Messages turn the server ends itself (the refusal guard): one text block, end_turn, no generation."""
        mid = messages_api.new_id("msg")
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        if body.get("stream"):
            sse = messages_api.sse
            writer = messages_api.EventWriter()
            events = [sse("message_start", {"type": "message_start", "message": {
                "id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
                "stop_reason": None, "stop_sequence": None, "usage": usage}})]
            events += writer.delta("text", text) + writer.finish([])
            events += [sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": usage}),
                       sse("message_stop", {"type": "message_stop"})]
            return StreamingResponse(iter(events), media_type="text/event-stream")
        return {"id": mid, "type": "message", "role": "assistant", "model": model, "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn", "stop_sequence": None, "usage": usage}

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(http: HttpRequest):
        async def count(body: dict, http: HttpRequest):
            # A forced tool_choice does not change the prompt, so it is counted, not rejected.
            _, _, ids, _ = _messages_prompt(body, messages_api.tool_choice(body))
            return {"input_tokens": len(ids)}
        return await _messages_errors(count, http)

    return app


def _text_of(content) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else ""


def fold_system_messages(messages: list[dict]) -> list[dict]:
    """The template accepts one system message, first. Clients send more:
    several leading ones (OpenCode) and instructions later in the
    conversation (Claude Code). Merge the leading run; render a later one in
    place as a user turn in <system-reminder> tags, which keeps the prefix
    before it (and its cache) unchanged. Developer messages have the same
    semantics as system messages."""
    messages = [{**message, "role": "system"} if isinstance(message, dict) and message.get("role") == "developer" else message
                for message in messages]
    lead = 0
    while lead < len(messages) and isinstance(messages[lead], dict) and messages[lead].get("role") == "system":
        lead += 1
    out = []
    if lead > 1:
        out.append({"role": "system", "content": "\n\n".join(t for t in (_text_of(m.get("content")) for m in messages[:lead]) if t)})
    else:
        out += messages[:lead]
    for message in messages[lead:]:
        if isinstance(message, dict) and message.get("role") == "system":
            message = {"role": "user", "content": f"<system-reminder>\n{_text_of(message.get('content'))}\n</system-reminder>"}
        out.append(message)
    return out


def _tool_arguments_as_objects(messages: list[dict]) -> list[dict]:
    """OpenAI clients send earlier tool calls' arguments as a JSON string; the
    template iterates them as a mapping. Returns messages with parsed copies."""
    out = []
    for message in messages:
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if isinstance(calls, list) and any(isinstance((c or {}).get("function", {}).get("arguments"), str) for c in calls):
            parsed = []
            for c in calls:
                args = (c or {}).get("function", {}).get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except ValueError:
                        raise InvalidRequest(f"tool call arguments must be a JSON object, got {args[:80]!r}")
                    if not isinstance(args, dict):
                        raise InvalidRequest("tool call arguments must be a JSON object")
                    c = {**c, "function": {**c["function"], "arguments": args}}
                parsed.append(c)
            message = {**message, "tool_calls": parsed}
        out.append(message)
    return out


def _split_chat(text: str, in_reasoning: bool, tools: list | None) -> tuple[str, str, list[dict]]:
    """Complete chat output as (reasoning, content, tool_calls)."""
    rp = ReasoningParser(in_reasoning, tools=bool(tools))
    rp.feed(text)
    rp.finish()
    content, calls = (parse_tool_calls(rp.content, tools) if tools else (rp.content, []))
    return rp.reasoning, content, calls


def _serve_http(app, host: str, port: int) -> None:
    """uvicorn.run, except that a handled SIGTERM ends here instead of being raised again once the server stops: in
    a container the server is PID 1 and ignores it, but run directly (from the venv, as the launchers do) it would end
    the process before the engine drains and its sessions persist."""
    import contextlib

    import uvicorn

    class Server(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self):
            with super().capture_signals():
                yield
                self._captured_signals.clear()

    Server(uvicorn.Config(app, host=host, port=port, log_level="warning", timeout_graceful_shutdown=30)).run()


def _persist_sessions(ex, engine, log) -> None:
    """Cached sessions survive the restart in the session store, once the engine has stopped: on every
    tensor-parallel rank or on none (parking grows the tiers through collectives)."""
    from sharar.distributed import all_reduce_min
    if all_reduce_min(int(not engine.thread.is_alive()))[0]:
        ex.persist_sessions(log)


def serve(host: str, port: int, log=print, spec: int = 0, proposer: str = "mtp", state_strategy: str = "snapshots", draft_precision: str = 'bf16',
          traces: str | None = None, traces_max_mib: int = 32768, draft_dir: str | None = None,
          accept_default: tuple[int, float] = (1, 0.0), target: str | None = None, max_context: int | None = None,
          accept_default_sampled: bool | tuple[int, float] = False, max_batch: int = 4) -> None:
    from sharar.cache.spec_budget import SpecConfig
    from sharar.api.validation import set_accept_default

    if accept_default != (1, 0.0) and not spec:
        raise ValueError('a relaxed acceptance default requires speculation')
    set_accept_default(*accept_default, sampled=accept_default_sampled)
    from sharar.api.validation import ACCEPT_DEFAULT_SAMPLED as sampled_threshold

    if not spec and draft_precision != 'bf16':
        raise ValueError('draft precision requires speculation')
    if draft_dir is not None and not spec:
        raise ValueError('a draft directory requires speculation')
    served = targets.get(target)
    spec_config = SpecConfig(proposer, spec, state_strategy, draft_precision, draft_dir) if spec else None
    if spec_config is not None:
        from sharar.proposers.factory import draft_snapshot
        if proposer == 'dflash2':
            draft_snapshot(proposer, draft_dir, served)  # fail before loading the target

    from sharar.engine.executor import Executor
    from sharar.engine.scheduler import SchedulerConfig
    snap = served.snapshot_dir()
    tok = SnapshotTokenizer(snap)
    tmpl = ChatTemplate(snap)
    log(f"loading {served.name}: {served.model_id}@{served.model_revision[:8]}")
    model = targets.load_model(served, snap, log=log)
    ex = Executor(model, log=log, spec_config=spec_config, max_context=max_context, max_batch=max_batch)
    ex.capture_graphs(log=log)
    ex.warmup(log=log)
    ex.enable_cold_tier(log=log)  # tensor-parallel ranks agree on every tier decision (cache/cold_tier.py)
    # a burst as wide as the batch must queue, not be refused: the queue holds at least two batches
    cfg = SchedulerConfig(max_running=ex.profile.max_batch, max_queued=max(ex.profile.max_queued, 2 * ex.profile.max_batch),
                          chunk_size=ex.profile.chunk_size, prefill_budget=ex.profile.prefill_budget)
    sched = Scheduler(ex, cfg)
    if not get_tp_info().is_primary():
        log(f"tensor-parallel rank {get_tp_info().rank}: serving the primary rank's requests")
        engine = EngineLoop(sched, Grammars.from_tokenizer(tok._tok, served.vocab_size, SamplingParams().stop_token_ids))
        engine.start()
        engine.thread.join()
        _persist_sessions(ex, engine, log)
        return
    sink = None
    if traces:
        from sharar.engine.traces import TraceSink

        sink = TraceSink(traces, max_bytes=traces_max_mib * 2**20)
        sched.sink = sink
        log(f"recording opt-in request traces under {sink.directory} (cap {traces_max_mib} MiB)")
    from sharar.engine.config import current
    if current().w3_warning:
        log(f"WARNING: {current().w3_warning}")
    describe = {
        "engine": "sharar",
        "target": served.name,
        "model_path": served.model_id,
        "revision": served.model_revision,
        "w3_target_identity": getattr(model, "w3_identity", None),
        "w3_warning": current().w3_warning,
        "context_length": ex.max_context,
        "tp_size": get_tp_info().size,
        "max_running_requests": cfg.max_running,
        "max_queued_requests": cfg.max_queued,
        "chunked_prefill_size": cfg.chunk_size,
        "prefill_budget": cfg.prefill_budget,
        "kv_pages": ex.kv_pool.num_pages,
        "state_slots": ex.state_pool.num_slots,
        "ledger_gib": {k: round(v / 2**30, 2) for k, v in ex.ledger.entries.items()},
        "speculative": {"algorithm": ex.spec.proposer.requirements.name, "num_draft_tokens": spec,
                        "depths_by_batch": ex.spec.depth_policy.depths,
                        "adaptive_step_cost": getattr(ex.spec.depth_policy, "step_cost", None),  # #43; None: fixed depth
                        "draft_precision": draft_precision, "draft": getattr(ex.spec.proposer.model, "source", None)
                        if hasattr(ex.spec.proposer, "model") else None,
                        "accept_default": {"accept_top_k": accept_default[0], "accept_margin": accept_default[1],
                                           "sampled": sampled_threshold != (1, 0.0), "sampled_accept_top_k": sampled_threshold[0],
                                           "sampled_accept_margin": sampled_threshold[1]}} if spec else None,
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
        _serve_http(app, host, port)
    finally:
        engine.drain(timeout=5)
        engine.thread.join(timeout=6)
        _persist_sessions(ex, engine, log)
        if sink is not None:
            sink.close()
