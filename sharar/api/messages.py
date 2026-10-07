"""Anthropic Messages API as a translation onto the chat path.

A request becomes the same chat-template input an OpenAI chat request would:
system text and text blocks become content parts (the template concatenates
them), assistant ``tool_use`` blocks become ``tool_calls`` with the input
object as arguments, ``thinking`` blocks become ``reasoning_content``, and
``tool_result`` blocks become ``tool`` messages in their original order.
Images and documents are rejected (a text-only model), except inside a
``tool_result``: there the agent fetched them, not the user, so each becomes a
short text note and the turn goes on (#121: Claude Code read a PNG and quit on
the 400).
Custom tools become OpenAI function schemas; typed Anthropic tools (server
tools such as web search) have no rendering and are dropped. Unknown
top-level fields are ignored; unknown content blocks are rejected.

Output goes the other way: reasoning becomes a ``thinking`` block, content a
``text`` block, and parsed calls ``tool_use`` blocks. The thinking signature
is a fixed placeholder that clients may send back; it is accepted and never
checked.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field

from sharar.engine.sampler import InvalidRequest

SIGNATURE = "sharar-unsigned"
_UNSUPPORTED_MEDIA = ("image", "document")
MEDIA_NOTE = "[{kind} omitted: this model reads text only and cannot see it]"


@dataclass
class MessagesRequest:
    messages: list[dict]  # chat-template messages
    tools: list[dict] | None  # OpenAI-style function schemas
    thinking: bool | None  # None: the request left thinking to the server default
    # Every cache_control marker as a path into the request: ("system", block),
    # ("tools", index) or ("messages", message, block). The prefix cache does
    # not read them; a later change maps them to recurrent-state snapshot
    # boundaries.
    cache_breakpoints: list[tuple] = field(default_factory=list)
    ignored_tools: int = 0  # typed (server) tools dropped from the schema list
    stopped: object = None  # refusal_guard.Refusals when the server ends this turn itself (#108)


def parse_request(body: dict) -> MessagesRequest:
    """Translate the prompt-shaping fields; raises InvalidRequest."""
    breakpoints: list[tuple] = []
    messages: list[dict] = []
    system = body.get("system")
    if isinstance(system, list):
        parts = _text_parts(system, "system", ("system",), breakpoints)
        if parts:
            messages.append({"role": "system", "content": parts})
    elif isinstance(system, str):
        if system:
            messages.append({"role": "system", "content": system})
    elif system is not None:
        raise InvalidRequest("system must be a string or a list of text blocks")

    turns = body.get("messages")
    if not isinstance(turns, list) or not turns:
        raise InvalidRequest("messages must be a non-empty list")
    for m, turn in enumerate(turns):
        if not isinstance(turn, dict) or turn.get("role") not in ("user", "assistant", "system"):
            raise InvalidRequest(f"messages.{m}: role must be 'user', 'assistant' or 'system'")
        content = turn.get("content")
        if turn["role"] == "system":  # instructions at this point of the conversation (see server.fold_system_messages)
            if isinstance(content, list):
                content = _text_parts(content, f"messages.{m}.content", ("messages", m), breakpoints)
            elif not isinstance(content, str):
                raise InvalidRequest(f"messages.{m}.content must be a string or a list of text blocks")
            messages.append({"role": "system", "content": content})
        elif isinstance(content, str):
            messages.append({"role": turn["role"], "content": content})
        elif isinstance(content, list):
            convert = _user_turn if turn["role"] == "user" else _assistant_turn
            messages += convert(content, m, breakpoints)
        else:
            raise InvalidRequest(f"messages.{m}.content must be a string or a list of content blocks")

    tools, ignored = _tools(body.get("tools"), breakpoints)
    return MessagesRequest(messages, tools, _thinking(body.get("thinking")), breakpoints, ignored)


def tool_choice(body: dict) -> str:
    """'auto', 'none', or the forced kinds 'any' and 'tool'."""
    choice = body.get("tool_choice")
    if choice is None:
        return "auto"
    kind = choice.get("type") if isinstance(choice, dict) else None
    if kind not in ("auto", "none", "any", "tool"):
        raise InvalidRequest("tool_choice.type must be 'auto', 'none', 'any' or 'tool'")
    return kind


def _thinking(value) -> bool | None:
    if value is None:
        return None
    kind = value.get("type") if isinstance(value, dict) else None
    if kind in ("enabled", "adaptive"):  # budget_tokens is accepted, not enforced
        return True
    if kind == "disabled":
        return False
    raise InvalidRequest("thinking.type must be 'enabled', 'adaptive' or 'disabled'")


def _check_block(block, where: str) -> str:
    kind = block.get("type") if isinstance(block, dict) else None
    if kind in _UNSUPPORTED_MEDIA:
        raise InvalidRequest(f"{where}: {kind} content is not supported; this server runs a text-only model")
    if not isinstance(kind, str):
        raise InvalidRequest(f"{where}: content blocks must be objects with a type")
    return kind


def _text(block: dict, where: str) -> dict:
    if not isinstance(block.get("text"), str):
        raise InvalidRequest(f"{where}.text must be a string")
    return {"type": "text", "text": block["text"]}


def _text_parts(blocks: list, where: str, path: tuple, breakpoints: list[tuple]) -> list[dict]:
    parts = []
    for b, block in enumerate(blocks):
        if _check_block(block, f"{where}.{b}") != "text":
            raise InvalidRequest(f"{where}.{b}: only text blocks are supported here, got '{block['type']}'")
        parts.append(_text(block, f"{where}.{b}"))
        if block.get("cache_control") is not None:
            breakpoints.append((*path, b))
    return parts


def _user_turn(blocks: list, m: int, breakpoints: list[tuple]) -> list[dict]:
    """Text runs become user messages and tool results become tool messages, in order."""
    out: list[dict] = []
    text: list[dict] = []
    for b, block in enumerate(blocks):
        where = f"messages.{m}.content.{b}"
        kind = _check_block(block, where)
        if kind == "text":
            text.append(_text(block, where))
        elif kind == "tool_result":
            if text:
                out.append({"role": "user", "content": text})
                text = []
            content = block.get("content")
            if isinstance(content, list):
                content = _text_parts([{"type": "text", "text": MEDIA_NOTE.format(kind=part["type"])}
                                       if isinstance(part, dict) and part.get("type") in _UNSUPPORTED_MEDIA else part
                                       for part in content], f"{where}.content", (), [])
            elif content is None:
                content = ""
            elif not isinstance(content, str):
                raise InvalidRequest(f"{where}.content must be a string or a list of text blocks")
            # is_error has no rendering in the template; clients put the error in the text.
            # is_error is kept for the refusal guard (api/refusal_guard.py); the template does not read it.
            out.append({"role": "tool", "tool_call_id": block.get("tool_use_id"), "content": content,
                        "is_error": bool(block.get("is_error"))})
        else:
            raise InvalidRequest(f"{where}: unsupported content block type '{kind}' in a user turn")
        if block.get("cache_control") is not None:
            breakpoints.append(("messages", m, b))
    if text or not out:
        out.append({"role": "user", "content": text})
    return out


def _assistant_turn(blocks: list, m: int, breakpoints: list[tuple]) -> list[dict]:
    text: list[dict] = []
    reasoning: list[str] = []
    calls: list[dict] = []
    for b, block in enumerate(blocks):
        where = f"messages.{m}.content.{b}"
        kind = _check_block(block, where)
        if kind == "text":
            text.append(_text(block, where))
        elif kind == "thinking":
            if not isinstance(block.get("thinking"), str):
                raise InvalidRequest(f"{where}.thinking must be a string")
            reasoning.append(block["thinking"])  # the signature is not checked
        elif kind == "redacted_thinking":
            pass  # opaque to this server
        elif kind == "tool_use":
            name, args = block.get("name"), block.get("input", {})
            if not isinstance(name, str) or not name or not isinstance(args, dict):
                raise InvalidRequest(f"{where}: tool_use needs a name and an input object")
            calls.append({"id": block.get("id"), "type": "function", "function": {"name": name, "arguments": args}})
        else:
            raise InvalidRequest(f"{where}: unsupported content block type '{kind}' in an assistant turn")
        if block.get("cache_control") is not None:
            breakpoints.append(("messages", m, b))
    message: dict = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning_content"] = "\n".join(reasoning)
    if calls:
        message["tool_calls"] = calls
    return [message]


def _tools(tools, breakpoints: list[tuple]) -> tuple[list[dict] | None, int]:
    if tools is None:
        return None, 0
    if not isinstance(tools, list):
        raise InvalidRequest("tools must be a list")
    out, ignored = [], 0
    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise InvalidRequest(f"tools.{i} must be an object")
        if tool.get("type") not in (None, "custom"):
            ignored += 1  # a server tool: nothing for this model to call
            continue
        name, schema = tool.get("name"), tool.get("input_schema")
        if not isinstance(name, str) or not name or not isinstance(schema, dict):
            raise InvalidRequest(f"tools.{i} needs a name and an input_schema object")
        fn = {"name": name}
        if tool.get("description") is not None:
            fn["description"] = tool["description"]
        fn["parameters"] = schema
        out.append({"type": "function", "function": fn})
        if tool.get("cache_control") is not None:
            breakpoints.append(("tools", i))
    return out or None, ignored


# ---- responses ----
def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def stop_reason(reason: str, stop_hit: str | None) -> tuple[str, str | None]:
    """(stop_reason, stop_sequence) from the chat finish reason."""
    if reason == "tool_calls":
        return "tool_use", None
    if stop_hit is not None:
        return "stop_sequence", stop_hit
    return ("max_tokens" if reason == "length" else "end_turn"), None


def tool_use(call: dict) -> dict:
    return {"type": "tool_use", "id": new_id("toolu"), "name": call["function"]["name"],
            "input": json.loads(call["function"]["arguments"])}


def content_blocks(reasoning: str, text: str, calls: list[dict]) -> list[dict]:
    blocks = []
    if reasoning.strip():
        blocks.append({"type": "thinking", "thinking": reasoning.strip(), "signature": SIGNATURE})
    if text.strip():
        blocks.append({"type": "text", "text": text.strip()})
    return blocks + [tool_use(c) for c in calls]


def error_body(kind: str, message: str) -> dict:
    return {"type": "error", "error": {"type": kind, "message": message}}


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class _Trimmed:
    """Stream a block's text without leading or trailing whitespace: trailing
    whitespace is held until more text follows it."""

    def __init__(self):
        self.started = False
        self.held = ""

    def feed(self, delta: str) -> str:
        text = self.held + delta
        if not self.started:
            text = text.lstrip()
        body = text.rstrip()
        self.held = text[len(body):]
        self.started = self.started or bool(body)
        return body


class EventWriter:
    """Chat stream steps as Messages stream events, one content block at a time."""

    def __init__(self):
        self.index = -1
        self.open: str | None = None  # "thinking" or "text"
        self._trim: _Trimmed | None = None

    def _close(self) -> list[str]:
        events = []
        if self.open == "thinking":
            events.append(sse("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                                      "delta": {"type": "signature_delta", "signature": SIGNATURE}}))
        if self.open is not None:
            events.append(sse("content_block_stop", {"type": "content_block_stop", "index": self.index}))
        self.open = None
        return events

    def _start(self, block: dict) -> list[str]:
        events = self._close()
        self.index += 1
        return events + [sse("content_block_start", {"type": "content_block_start", "index": self.index, "content_block": block})]

    def delta(self, kind: str, text: str) -> list[str]:
        """kind is "thinking" or "text"."""
        events = []
        if self.open != kind:
            if not text.strip():
                return []  # no block for whitespace alone
            self._trim = _Trimmed()
            events = self._start({"type": "thinking", "thinking": "", "signature": ""} if kind == "thinking"
                                 else {"type": "text", "text": ""})
            self.open = kind
        text = self._trim.feed(text)
        if text:
            delta = {"type": "thinking_delta", "thinking": text} if kind == "thinking" else {"type": "text_delta", "text": text}
            events.append(sse("content_block_delta", {"type": "content_block_delta", "index": self.index, "delta": delta}))
        return events

    def finish(self, calls: list[dict]) -> list[str]:
        """Close the open block and emit each tool call as a whole block."""
        events = self._close()
        for call in calls:
            block = tool_use(call)
            events += self._start({**block, "input": {}})
            events.append(sse("content_block_delta", {"type": "content_block_delta", "index": self.index, "delta": {
                "type": "input_json_delta", "partial_json": json.dumps(block["input"], ensure_ascii=False)}}))
            events.append(sse("content_block_stop", {"type": "content_block_stop", "index": self.index}))
        return events
