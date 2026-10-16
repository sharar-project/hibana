"""The refusal guard: repeated refusals of one tool get a reminder, then the server ends the turn."""
import json

import pytest

from sharar.api import refusal_guard
from tests.unit.test_messages_api import post, sse_events

REFUSAL = "This command requires approval"


def call(i, name="Bash", command="python3 test_calc.py"):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps({"command": command})}}]}


def result(i, text=REFUSAL, is_error=None):
    message = {"role": "tool", "tool_call_id": f"c{i}", "content": text}
    if is_error is not None:
        message["is_error"] = is_error
    return message


def refused_turn(n, allowed_between=False):
    messages = [{"role": "user", "content": "Fix calc.py and run the test."}]
    for i in range(n):
        messages += [call(i), result(i, is_error=True)]
        if allowed_between:  # Claude Code lets read-only commands through between refused ones
            messages += [call(100 + i, command="ls"), result(100 + i, "calc.py\ntest_calc.py", is_error=False)]
    return messages


def test_refusals_count_one_tools_identical_refusals_since_the_users_message():
    r = refusal_guard.refusals(refused_turn(4, allowed_between=True))
    assert (r.tool, r.text, r.count) == ("Bash", REFUSAL, 4)
    messages = refused_turn(4) + [{"role": "user", "content": "OK, skip the test then."}, call(9), result(9, is_error=True)]
    assert refusal_guard.refusals(messages) is None  # a new user message starts over; one refusal is not a pattern
    reminder = {"role": "user", "content": "<system-reminder>\nkeep going\n</system-reminder>"}
    assert refusal_guard.refusals(refused_turn(2) + [reminder, call(5), result(5, is_error=True)]).count == 3


def test_refusals_are_errors_or_read_like_refusals_and_repeat():
    plain = [{"role": "user", "content": "go"}]
    for i in range(4):
        plain += [call(i), result(i, "FAILED test_calc.py::test_add - AssertionError")]  # a failing test, not a refusal
    assert refusal_guard.refusals(plain) is None
    worded = [{"role": "user", "content": "go"}]
    for i in range(3):
        worded += [call(i), result(i, "Permission to use Bash has been denied.")]  # OpenAI clients send no is_error
    assert refusal_guard.refusals(worded).count == 3


@pytest.mark.parametrize("n,action", [(2, "generate"), (refusal_guard.NUDGE, "nudge"), (refusal_guard.STOP - 1, "nudge"), (refusal_guard.STOP, "stop")])
def test_decide(n, action):
    assert refusal_guard.decide(refused_turn(n), {})[0] == action
    assert refusal_guard.decide(refused_turn(n), {"refusal_guard": False})[0] == "generate"


def anthropic_turn(n):
    messages = [{"role": "user", "content": "Fix calc.py and run the test."}]
    for i in range(n):
        messages.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": "python3 test_calc.py"}}]})
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": REFUSAL, "is_error": True}]})
    return {"model": "claude-sharar", "max_tokens": 256, "messages": messages,
            "tools": [{"name": "Bash", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}]}


def test_messages_nudge_adds_a_reminder_after_the_refusals():
    response, engine, tmpl = post(anthropic_turn(refusal_guard.NUDGE), "Done.")
    assert response.status_code == 200
    rendered = tmpl.render.call_args.args[0]
    assert rendered[-1]["role"] == "user" and "were refused with" in rendered[-1]["content"] and "`Bash`" in rendered[-1]["content"]
    engine.submit.assert_awaited_once()


@pytest.mark.parametrize("stream", [False, True])
def test_messages_stop_ends_the_turn_without_generating(stream):
    response, engine, _ = post({**anthropic_turn(refusal_guard.STOP), "stream": stream}, "never")
    assert response.status_code == 200
    engine.submit.assert_not_awaited()
    if stream:
        events = sse_events(response.text)
        assert [e["type"] for e in events][0] == "message_start" and events[-1]["type"] == "message_stop"
        assert any(e["type"] == "message_delta" and e["delta"]["stop_reason"] == "end_turn" for e in events)
        text = "".join(e["delta"].get("text", "") for e in events if e["type"] == "content_block_delta")
    else:
        assert response.json()["stop_reason"] == "end_turn"
        text = response.json()["content"][0]["text"]
    assert "refused 8 times" in text


@pytest.mark.parametrize("stream", [False, True])
def test_chat_stop_and_opt_out(stream):
    body = {"messages": refused_turn(refusal_guard.STOP, allowed_between=True), "max_tokens": 64, "stream": stream}
    response, engine, _ = post(body, "never", path="/v1/chat/completions")
    assert response.status_code == 200
    engine.submit.assert_not_awaited()
    if stream:
        chunks = [json.loads(line[6:]) for line in response.text.split("\n") if line.startswith("data: {")]
        assert "refused" in chunks[0]["choices"][0]["delta"]["content"] and chunks[-1]["choices"][0]["finish_reason"] == "stop"
    else:
        assert "refused" in response.json()["choices"][0]["message"]["content"]
    response, engine, _ = post({**body, "refusal_guard": False}, "ok", path="/v1/chat/completions")
    engine.submit.assert_awaited_once()
