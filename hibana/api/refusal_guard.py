"""Stop an agent from retrying a tool call its harness keeps refusing.

When a harness refuses a call (Claude Code without permission: "This command requires approval"), a frontier model
reports it and stops. Qwen3.8 sometimes retries with small variations instead: 352 calls in five minutes on the 27B,
until the client gave up. The conversation comes back whole on every turn, so the server sees the
refusals without keeping any state:

- a refusal is a tool result marked as an error (Anthropic `is_error`), or one whose text reads like a permission
  refusal, that repeats exactly for the same tool since the user's last message;
- from NUDGE such refusals on, a reminder is added after them: the call was refused N times, retrying will be refused
  again, finish without it and tell the user what is needed;
- from STOP refusals on (the model kept going despite the reminder), the server ends the turn itself with a short
  message instead of generating.

Requests may set `refusal_guard: false` to turn this off.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NUDGE = 3
STOP = 8
_REFUSAL = re.compile(r"requires? approval|permission|denied|not allowed|rejected|refused|blocked|not permitted", re.I)


@dataclass(frozen=True)
class Refusals:
    tool: str
    text: str  # the refusal, as the harness worded it
    count: int


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content if isinstance(content, str) else ""


def _is_user_turn(message: dict) -> bool:
    """A message from the user (not a tool result, not a harness reminder alone)."""
    if message.get("role") != "user":
        return False
    text = _text(message.get("content")).strip()
    return bool(text) and not (text.startswith("<system-reminder>") and text.endswith("</system-reminder>"))


def refusals(messages: list[dict]) -> Refusals | None:
    """The most repeated refusal of one tool since the user's last message, or None."""
    start = 0
    for i, message in enumerate(messages):
        if isinstance(message, dict) and _is_user_turn(message):
            start = i
    names: dict = {}
    counts: dict[tuple[str, str], int] = {}
    for message in messages[start:]:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            fn = (call or {}).get("function") or {}
            names[(call or {}).get("id")] = fn.get("name", "")
        if message.get("role") != "tool":
            continue
        text = _text(message.get("content")).strip()
        if not text or len(text) > 400 or not (message.get("is_error") or _REFUSAL.search(text)):
            continue
        key = (names.get(message.get("tool_call_id"), ""), text)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    (tool, text), count = max(counts.items(), key=lambda item: item[1])
    return Refusals(tool, text, count) if count >= 2 else None


def reminder(r: Refusals) -> dict:
    """The user-turn reminder added after the refused calls."""
    tool = f"`{r.tool}`" if r.tool else "this tool"
    return {"role": "user", "content": (
        f"<system-reminder>\nThe last {r.count} calls to {tool} were refused with: \"{r.text[:200]}\". Calling it again "
        f"will be refused again. Do not call {tool} again. Finish what you can without it, then tell the user what you "
        f"could not do and what permission or change you need.\n</system-reminder>")}


def stop_message(r: Refusals) -> str:
    tool = f"`{r.tool}`" if r.tool else "a tool"
    return (f"I stopped: {tool} was refused {r.count} times (\"{r.text[:200]}\"), so I can't continue this step. "
            f"Grant the permission or run it yourself, and tell me how to proceed.")


def decide(messages: list[dict], body: dict) -> tuple[str, Refusals | None]:
    """("generate" | "nudge" | "stop", refusals) for one request."""
    if body.get("refusal_guard") is False:
        return "generate", None
    r = refusals(messages)
    if r is None or r.count < NUDGE:
        return "generate", r
    return ("stop" if r.count >= STOP else "nudge"), r
