from sharar.api.parsers.reasoning import ReasoningParser
from sharar.api.parsers.tools import parse_tool_calls


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


def test_no_tool_call_passthrough():
    content, calls = parse_tool_calls("plain answer", None)
    assert content == "plain answer" and calls == []
