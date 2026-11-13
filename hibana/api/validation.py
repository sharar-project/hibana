"""HTTP input normalization and supported output-control validation."""
import math
from hibana.engine.sampler import InvalidRequest, SamplingParams

# The relaxed rules requests get when they do not set their own (set once by serve()): greedy requests that set
# neither accept_top_k nor accept_margin get ACCEPT_DEFAULT ((1, 0.0): exact; --accept-default), sampled requests that
# do not set accept_soft get ACCEPT_SOFT_DEFAULT (0: exact; --accept-soft). Sampled requests ignore accept_top_k and
# accept_margin.
ACCEPT_DEFAULT = (1, 0.0)
ACCEPT_SOFT_DEFAULT = 0.0


def set_accept_default(top_k: int, margin: float, soft: float = 0.0) -> None:
    global ACCEPT_DEFAULT, ACCEPT_SOFT_DEFAULT
    SamplingParams(accept_top_k=top_k, accept_margin=margin, accept_soft=soft).validate()
    ACCEPT_DEFAULT, ACCEPT_SOFT_DEFAULT = (top_k, margin), soft


def parse_accept_default(text: str | None) -> tuple[int, float]:
    """'off', '' or None -> exact; 'K:M' -> top-k K within M logits."""
    if text is None or text.strip().lower() in ("", "off", "exact", "0", "1", "1:0", "1:0.0"):
        return (1, 0.0)
    try:
        k, m = text.split(":")
        return (int(k), float(m))
    except ValueError as exc:
        raise ValueError("accept default must be K:M, e.g. 2:1.0, or off") from exc


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
    soft = body.get("accept_soft", sp.get("accept_soft"))
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
        trace=body.get("trace", sp.get("trace", True)),
        accept_top_k=integer(body.get("accept_top_k", sp.get("accept_top_k", 1))),
        accept_margin=body.get("accept_margin", sp.get("accept_margin", 0.0)),
        accept_soft=soft if soft is not None else ACCEPT_SOFT_DEFAULT,
    )
    if (params.greedy and "accept_top_k" not in body and "accept_top_k" not in sp and "accept_margin" not in body
            and "accept_margin" not in sp and ACCEPT_DEFAULT != (1, 0.0)):
        params.accept_top_k, params.accept_margin = ACCEPT_DEFAULT
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


# Client effort levels onto the template's reasoning_effort (xhigh, medium, low).
_EFFORT = {"low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}


def _template_kwargs(body: dict) -> tuple[dict, bool]:
    """chat_template_kwargs without enable_thinking, and the thinking switch (default on).

    A client effort level (OpenAI `reasoning_effort`, Anthropic
    `output_config.effort`) selects the template's reasoning effort unless
    chat_template_kwargs sets it; unknown levels keep the template default."""
    kwargs = dict(body.get("chat_template_kwargs") or {})
    config = body.get("output_config")
    effort = body.get("reasoning_effort") or (config.get("effort") if isinstance(config, dict) else None)
    if "reasoning_effort" not in kwargs and isinstance(effort, str) and effort.lower() in _EFFORT:
        kwargs["reasoning_effort"] = _EFFORT[effort.lower()]
    return kwargs, bool(kwargs.pop("enable_thinking", True))


def _output_spec(body: dict) -> tuple[str, object] | None:
    """The output constraint a chat request asks for, validated before rendering or admission (engine/grammar.py):
    ("tool_call", function name or None for any tool), ("json_object", None), ("json_schema", schema) or None."""
    spec = None
    choice = body.get("tool_choice")
    if choice is not None and choice not in ("auto", "none"):
        name = (choice.get("function") or {}).get("name") if isinstance(choice, dict) and choice.get("type") == "function" else None
        if choice == "required":
            spec = ("tool_call", None)
        elif isinstance(name, str) and name:
            spec = ("tool_call", name)
        else:
            raise InvalidRequest("tool_choice must be null, 'auto', 'none', 'required' or "
                                 "{'type': 'function', 'function': {'name': ...}}")
        if not body.get("tools"):
            raise InvalidRequest("a tool_choice that requires a call needs tools")
    response_format = body.get("response_format")
    if response_format is not None and response_format != {"type": "text"}:
        kind = response_format.get("type") if isinstance(response_format, dict) else None
        schema = (response_format.get("json_schema") or {}).get("schema") if kind == "json_schema" and isinstance(response_format.get("json_schema"), dict) else None
        if kind == "json_object":
            fmt = ("json_object", None)
        elif isinstance(schema, dict):
            fmt = ("json_schema", schema)
        else:
            raise InvalidRequest('response_format must be {"type": "text"}, {"type": "json_object"} or '
                                 '{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}')
        if spec is not None:
            raise InvalidRequest("a tool_choice that requires a call cannot be combined with a JSON response_format")
        spec = fmt
    return spec
