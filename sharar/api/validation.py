"""HTTP input normalization and supported output-control validation."""
import math
from sharar.engine.sampler import InvalidRequest, SamplingParams

# Relaxed acceptance applied to requests that set neither accept_top_k nor
# accept_margin: greedy ones, and sampled ones when ACCEPT_DEFAULT_SAMPLED;
# (1, 0.0) means exact. Set once by serve() from --accept-default(-sampled).
ACCEPT_DEFAULT = (1, 0.0)
ACCEPT_DEFAULT_SAMPLED = False


def set_accept_default(top_k: int, margin: float, sampled: bool = False) -> None:
    global ACCEPT_DEFAULT, ACCEPT_DEFAULT_SAMPLED
    SamplingParams(accept_top_k=top_k, accept_margin=margin).validate()
    ACCEPT_DEFAULT, ACCEPT_DEFAULT_SAMPLED = (top_k, margin), bool(sampled)


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
    )
    if ("accept_top_k" not in body and "accept_top_k" not in sp and "accept_margin" not in body and "accept_margin" not in sp
            and (params.greedy or ACCEPT_DEFAULT_SAMPLED) and ACCEPT_DEFAULT != (1, 0.0)):
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


def _validate_chat_controls(body: dict) -> None:
    """Accept only unconstrained output; fail before rendering or admission."""
    choice = body.get("tool_choice")
    if choice is not None and choice not in ("auto", "none"):
        raise InvalidRequest("tool_choice supports only null, 'auto' or 'none'; forced tool calls are not supported")
    response_format = body.get("response_format")
    if response_format is not None and response_format != {"type": "text"}:
        raise InvalidRequest('response_format supports only null or {"type":"text"}; constrained output is not supported')
