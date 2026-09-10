"""HTTP input normalization and supported output-control validation."""
import math
from sharar.engine.sampler import InvalidRequest, SamplingParams

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
    )
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


def _validate_chat_controls(body: dict) -> None:
    """Accept only unconstrained output; fail before rendering or admission."""
    choice = body.get("tool_choice")
    if choice is not None and choice not in ("auto", "none"):
        raise InvalidRequest("tool_choice supports only null, 'auto' or 'none'; forced tool calls are not supported")
    response_format = body.get("response_format")
    if response_format is not None and response_format != {"type": "text"}:
        raise InvalidRequest('response_format supports only null or {"type":"text"}; constrained output is not supported')
