# HTTP API

The launcher exposes the API at `http://127.0.0.1:30001`; `./sharar-ctl chat` is a terminal client for it. The supported target
context is 32,768 tokens, with at most four running and 16 queued requests.
Admission also checks the output budget and speculative scratch capacity.

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Chat, optional reasoning and parsed tool calls |
| `POST /v1/completions` | Text completions |
| `POST /generate` | Native text or token-ID input, output IDs and request metadata |
| `POST /v1/tokenize` | Token counting/encoding |
| `POST /flush_cache` | Invalidate reusable prefixes; active loans drain safely |
| `POST /reload_draft` | Swap the served draft in place between rounds: `{"draft": DIR}` or `{"draft": "pinned"}`; refuses shape or context-weight changes |
| `GET /v1/models` | Supported model |
| `GET /health` | Engine availability |
| `GET /get_server_info` | Serving profile, scheduler, cache and speculation statistics |
| `GET /metrics` | Metrics |

## Chat controls

| Field | Supported behavior |
|---|---|
| `messages` | Nonempty conversation rendered with the pinned snapshot template |
| `stream` | Server-sent events ending with `[DONE]` |
| `max_tokens` / `max_completion_tokens` | Requested output budget |
| `temperature`, `top_p`, `top_k`, `seed` | Validated sampling parameters |
| `stop` | A nonempty string or list of nonempty stop strings |
| `chat_template_kwargs.enable_thinking` | Thinking defaults to true; false disables it |
| `tools` | Tool schemas passed to the template; generated calls are parsed |
| `tool_choice` omitted/null/`auto` | Unconstrained model choice |
| `tool_choice: "none"` | Omit tool schemas from the prompt and tool-call parsing |
| `response_format` omitted/null/`{"type":"text"}` | Ordinary text output |
| `trace` | `false` excludes this request from the opt-in [trace sink](operations.md#request-traces); accepted on every generation route |
| `accept_top_k`, `accept_margin` | Relaxed speculative acceptance, greedy requests only, off unless both are set: a drafted token is also accepted when the target ranks it within its top `accept_top_k` and within `accept_margin` logits of the argmax (a log-probability ratio; 1.0 means at least 37% of the top probability). Faster, and no longer the target's exact greedy continuation; the per-request `spec_relaxed_tokens` counts such tokens. **Default: the server applies top-2 within 1.0 logit to greedy requests that set neither field** (`--accept-default`, `SHARAR_ACCEPT_DEFAULT`; `off` keeps everything exact); send `accept_top_k: 1` to get the exact greedy continuation for one request. Accepted on every generation route |

`GET /get_server_info` reports `speculative.draft` (the served draft
directory), `speculative.accept_default`, `spec_stats` (rounds, proposed, accepted, emitted, `relaxed_accepts`) and, when
tracing is on, `scheduler.traces` (records, bytes, `dropped`).

Forced/named/required tool choices and constrained JSON formats return HTTP
400 before rendering or admission. `none` does not install a token constraint;
it suppresses supplied tool schemas and returns ordinary generated content.
Sharar does not execute tools. Clients perform calls and submit subsequent turns.

Thinking content is returned separately as `reasoning_content`; tool calls use
`tool_calls`. Streaming chat can return final usage when
`stream_options.include_usage` is true. Chat and completion usage report the
prompt tokens reused from the prefix cache as
`prompt_tokens_details.cached_tokens`. The native `/generate` route accepts
`input_ids` or `text`, and a `sampling_params` object; its metadata includes
cached tokens and speculative work when used. `/generate` and OpenAI routes
have different response schemas.

Queue saturation returns 429. An unavailable engine returns 503; a stream
already in progress reports an SSE error. Client disconnects cancel their
requests. Cache flush acknowledges invalidation; locked storage may remain
until its readers finish, reported by `flush_pending`.

The API's supported controls are checked in `tests/unit/test_chat_controls.py`;
failure, disconnect and drain behavior has separate lifecycle tests. Current
request-level cache modes/priorities are only a [proposal](request-cache-policy.md).
