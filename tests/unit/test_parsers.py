import pytest

from hibana.api.parsers.reasoning import ReasoningParser
from hibana.api.parsers.tools import parse_tool_calls


def test_reasoning_split_streaming_with_split_tag():
    p = ReasoningParser(in_reasoning=True)
    out = [p.feed(x) for x in ["I think", " so.</thi", "nk>\n\nThe answer", " is 4"]]
    assert p.reasoning == "I think so." and p.content == "\n\nThe answer is 4"
    assert out[0] == ("I think", "") and out[3] == ("", " is 4")


def test_reasoning_disabled_is_plain_content():
    p = ReasoningParser(in_reasoning=False)
    p.feed("Hello <think>not a tag here")  # tag not at start: content
    assert p.content.startswith("Hello") and p.reasoning == ""


def test_reasoning_explicit_start_tag():
    p = ReasoningParser(in_reasoning=False)
    p.feed("<think>hmm</think>ok")
    assert p.reasoning == "hmm" and p.content == "ok"


def test_tool_call_parse_with_schema_casts():
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}}]
    text = 'Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>'
    content, calls = parse_tool_calls(text, tools)
    assert content == "Let me check."
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"
    import json
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris", "days": 3}


@pytest.mark.parametrize("schema,text,want", [
    ({"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]}, '["x@y.z"]', ["x@y.z"]),
    ({"anyOf": [{"type": "array"}, {"type": "null"}]}, "null", None),
    ({"type": ["integer", "null"]}, "3", 3),
    ({"type": ["integer", "null"]}, "null", None),
    ({"oneOf": [{"type": "object"}, {"type": "null"}]}, '{"a": 1}', {"a": 1}),
    ({"anyOf": [{"type": "array"}, {"type": "null"}]}, "x@y.z", "x@y.z"),  # fits no type: the text, for the tool's error
    ({"anyOf": [{"type": "string"}, {"type": "array"}]}, '["a"]', '["a"]'),  # schema order: a string stays a string
    ({"type": "boolean"}, "maybe", "maybe"),
    ({"type": "string"}, "null", "null"),
])
def test_tool_call_casts_optional_and_union_parameters(schema, text, want):
    """Optional[X] (anyOf / oneOf with null) and type lists cast to the first type the text reads as."""
    import json
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"p": schema}}}}]
    _, calls = parse_tool_calls(f"<tool_call>\n<function=f>\n<parameter=p>\n{text}\n</parameter>\n</function>\n</tool_call>", tools)
    assert json.loads(calls[0]["function"]["arguments"]) == {"p": want}


def test_no_tool_call_passthrough():
    content, calls = parse_tool_calls("plain answer", None)
    assert content == "plain answer" and calls == []


CALL = "<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"


def feed_all(p, text, size):
    out = [p.feed(text[i:i + size]) for i in range(0, len(text), size)] + [p.finish()]
    return "".join(r for r, _ in out), "".join(c for _, c in out)


@pytest.mark.parametrize("size", [1, 3, 1000])
def test_tool_call_in_unclosed_reasoning_is_content(size):
    """The turn ends inside reasoning with a complete call; the call is content, the rest stays reasoning."""
    p = ReasoningParser(in_reasoning=True, tools=True)
    assert feed_all(p, "Check the files.\n\n" + CALL, size) == ("Check the files.\n\n", CALL)
    assert (p.reasoning, p.content) == ("Check the files.\n\n", CALL)


@pytest.mark.parametrize("size", [1, 3, 1000])
def test_tool_call_drafted_in_closed_reasoning_stays_reasoning(size):
    p = ReasoningParser(in_reasoning=True, tools=True)
    text = "Maybe " + CALL + " no.</think>\n\nDone."
    assert feed_all(p, text, size) == ("Maybe " + CALL + " no.", "\n\nDone.")


@pytest.mark.parametrize("tools,text", [
    (False, "Thinking " + CALL),                        # no tools offered: stays reasoning
    (True, "Thinking <tool_call>\n<function=Bash>"),    # cut off mid-call (max_tokens): still reasoning
])
def test_unclosed_reasoning_otherwise_stays_reasoning(tools, text):
    p = ReasoningParser(in_reasoning=True, tools=tools)
    assert feed_all(p, text, 2) == (text, "")
