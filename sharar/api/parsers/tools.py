"""Qwen3.8 tool calls: XML-style blocks emitted by the model.

  <tool_call>
  <function=name>
  <parameter=arg>
  value (may span lines)
  </parameter>
  </function>
  </tool_call>

Parameter values are cast using the tool's JSON schema when available
(integer, number, boolean, object/array parsed as JSON), else kept as strings.
Text before the first block is regular content; nothing after is kept.
"""

from __future__ import annotations

import json
import re
import uuid

BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNC = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL)
PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
TOOL_CALL_START = "<tool_call>"


def has_tool_call(text: str) -> bool:
    return TOOL_CALL_START in text


def _cast(value: str, schema: dict | None):
    if not schema:
        return value
    t = schema.get("type")
    v = value.strip()
    try:
        if t == "integer":
            return int(v)
        if t == "number":
            return float(v)
        if t == "boolean":
            return v.lower() in ("true", "1", "yes")
        if t in ("object", "array"):
            return json.loads(v)
    except (ValueError, json.JSONDecodeError):
        return value
    return value


def parse_tool_calls(text: str, tools: list[dict] | None) -> tuple[str, list[dict]]:
    """Returns (content_before_calls, openai_tool_calls)."""
    if not has_tool_call(text):
        return text, []
    schemas: dict[str, dict] = {}
    for tool in tools or []:
        fn = tool.get("function", tool)
        params = (fn.get("parameters") or {}).get("properties") or {}
        schemas[fn.get("name", "")] = params
    calls = []
    for block in BLOCK.finditer(text):
        m = FUNC.search(block.group(1))
        if not m:
            continue
        name = m.group(1).strip()
        args = {}
        for pm in PARAM.finditer(m.group(2)):
            args[pm.group(1).strip()] = _cast(pm.group(2), schemas.get(name, {}).get(pm.group(1).strip()))
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        })
    content = text[: text.find(TOOL_CALL_START)].rstrip()
    return content, calls
