# HTTP API

The launcher exposes the API at `http://127.0.0.1:30001`; `sharar chat` is a terminal client for it. The supported target
context is 262,144 tokens as launched (`SHARAR_MAX_CONTEXT`; the engine default is 32,768), with at most four running and 16 queued requests.
Admission also checks the output budget and speculative scratch capacity.

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Chat, optional reasoning and parsed tool calls |
| `POST /v1/messages` | Anthropic Messages on the chat path: thinking, text and tool_use blocks |
| `POST /v1/messages/count_tokens` | Rendered prompt length of a Messages request |
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
| `temperature`, `top_p`, `top_k`, `seed` | Validated sampling parameters. An omitted `temperature` is OpenAI's 1.0 (with top_p 0.95 and top_k 20, the models' own generation config); send `temperature: 0` for greedy. OpenCode sends no sampling fields, and greedy thinking looped until `max_tokens` in its runs. `/v1/completions` and `/generate` keep greedy as their default |
| `stop` | A nonempty string or list of nonempty stop strings |
| `n` | 1 to 16 choices (non-streaming). The prompt is prefilled once by a one-token primer and published to the prefix cache; the n samples resume from it (at most 64 tokens re-prefilled each) and decode side by side, four at a time. With a `seed`, choice i uses seed + i; use a temperature above 0 for distinct choices |
| `chat_template_kwargs.enable_thinking` | Thinking defaults to true; false disables it |
| `reasoning_effort` | `low`, `medium`, `high` select the template's reasoning instruction (`low`, `medium`, `xhigh`; `high` and `max` map to `xhigh`, the default); unknown values keep the default; `chat_template_kwargs.reasoning_effort` takes precedence |
| `tools` | Tool schemas passed to the template; generated calls are parsed |
| `tool_choice` omitted/null/`auto` | Unconstrained model choice |
| `tool_choice: "none"` | Omit tool schemas from the prompt and tool-call parsing |
| `tool_choice: "required"` / `{"type":"function","function":{"name":...}}` | Constrained: one or more calls to any tool / exactly one call to that tool ([constrained output](#constrained-output)) |
| `response_format` omitted/null/`{"type":"text"}` | Ordinary text output |
| `response_format: {"type":"json_object"}` / `{"type":"json_schema","json_schema":{"schema":{...}}}` | Constrained: a JSON object / JSON valid against the schema ([constrained output](#constrained-output)) |
| `trace` | `false` excludes this request from the opt-in [trace sink](operations.md#request-traces); accepted on every generation route |
| `accept_top_k`, `accept_margin` | Relaxed speculative acceptance: a drafted token is also accepted when the target ranks it within its top `accept_top_k` and within `accept_margin` logits of the argmax (a log-probability ratio; 1.0 means at least 37% of the top probability). Faster, and no longer the target's exact continuation; the per-request `spec_relaxed_tokens` counts such tokens. **A request that sets neither field gets the server's default**, which the serving profile chooses ([operations](operations.md#serving-profiles); `fast`: top-3 within 2.0 for greedy requests, and for sampled ones top-3 within 2.0 on Flash-Next, top-2 within 1.0 on the 27B; `exact`: off). Send `accept_top_k: 1` for the exact continuation on one request. Accepted on every generation route |

`GET /get_server_info` reports `speculative.draft` (the served draft
directory), `speculative.accept_default`, `spec_stats` (rounds, proposed, accepted, emitted, `relaxed_accepts`) and, when
tracing is on, `scheduler.traces` (records, bytes, `dropped`).

`none` does not install a token constraint;
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

## Refused tool calls

When a harness keeps refusing a tool call (Claude Code without permission answers "This command requires approval"),
the model sometimes retries it with small variations instead of reporting it: on 2026-10-05 the 27B did so on 4 of 4
phrasings of a fix-and-test task in the `fast` profile and 1 of 4 in `exact`, each time for minutes (229-352 calls)
until the client gave up. The server guards against it without keeping state, since the conversation comes back on
every turn ([sharar/api/refusal_guard.py](../sharar/api/refusal_guard.py)):

- A refusal is a tool result marked as an error (Anthropic `is_error`), or reading like a permission refusal, that
  repeats word for word for the same tool since the user's last message.
- From 3 such refusals a reminder follows them in the prompt: the call was refused N times, retrying will be refused
  again, finish without it and tell the user what is needed. With it every looping case above stopped after 3
  refusals with a correct report.
- From 8 the server ends the turn itself with a short message (`end_turn` / `finish_reason: stop`) instead of
  generating.
- `/get_server_info` counts both (`refusal_guard`); a request can send `refusal_guard: false` to turn it off.

## Constrained output

A JSON `response_format` or a `tool_choice` that requires a call (chat; `any`/`tool` on Anthropic Messages) is
enforced token by token with an [xgrammar](https://github.com/mlc-ai/xgrammar) grammar
([sharar/engine/grammar.py](../sharar/engine/grammar.py)), and speculative decoding stays on: after each
verification the drafted tokens are walked through the grammar on the host while the GPU verifies, the target's
verified rows are masked by the grammar state after each drafted prefix, and acceptance (exact, relaxed or rejection
sampling) runs on the masked rows, so a drafted token the grammar forbids is rejected and the correction is always
allowed. The drafted tokens reach the host right after drafting and the masks go up from pinned memory, so the walk
overlaps verification and device-resident acceptance stays on. The first token after prefill and plain decode
steps sample from the masked logits too. A grammar failure finishes that request (reason `grammar`), never the
engine.

- Thinking is free: the grammar applies from the first token after `</think>`. Leading whitespace (at most four
  characters) is allowed; whitespace runs inside JSON are capped at 24 characters.
- JSON schemas compile with xgrammar's `from_json_schema` (properties in the schema's order); a schema it cannot
  compile returns 400. `json_object` is any JSON object.
- Forced tool calls follow Qwen3.8's XML call format with the tools' function and parameter names; parameter values
  are free text (without `</`) and are cast by the usual tool-call parser. The parsed calls come back as `tool_calls`
  (`tool_use` blocks on Messages).
- A forced `tool_choice` cannot be combined with a JSON `response_format`; either needs `tools` / a schema.
- `/get_server_info` counts constrained requests by kind (`constrained`).

Measured 2026-10-05 on the GB10, fast profile: six schemas (weather, contact, order with line items, sentiment,
code review, entities) with thinking off and on, greedy and t=0.7 (four samples each), `json_object`, forced tool
calls on both APIs and a mixed batch of four constrained and four unconstrained requests: 49/49 valid on each target.
Speed at batch one against the same request without the constraint: tokens per round identical, decode tok/s
-1.6% to +1.0% on the 27B and -0.8% to 0% on Flash-Next (within run-to-run noise).

## Anthropic Messages

`POST /v1/messages` translates the request into the chat input above and
runs the same path: template, reasoning and tool-call parsing, sampling
validation, the relaxed-acceptance default, `trace` and disconnect
cancellation. `POST /v1/messages/count_tokens` returns the exact rendered
prompt length as `{"input_tokens": N}`.

| Field | Supported behavior |
|---|---|
| `model` | Any string; the single target serves it and the response echoes it |
| `max_tokens` | Required; capped at the context left after the prompt, since clients size it for a larger window |
| `system` | A string or text blocks |
| `messages` | `user`, `assistant` and `system` turns (see below). Text; in assistant turns `thinking` (rendered as reasoning; signatures are not checked), `redacted_thinking` (dropped) and `tool_use`; in user turns `tool_result` with string or text-block content, in order with the turn's text |
| `tools` | Custom tools become the chat tool schemas; typed server tools (`web_search_*` and similar) are dropped |
| `tool_choice` | `auto` or omitted, and `none`, as in chat |
| `thinking` | `enabled` or `adaptive` turn thinking on (`budget_tokens` is not enforced), `disabled` turns it off; omitted, `chat_template_kwargs.enable_thinking` decides, default on |
| `output_config.effort` | As chat `reasoning_effort`: Claude Code's default `high` is the template's `xhigh`, `/effort low` its `low` (the template's instruction only; on three short prompts on 2026-09-23 `low` did not shorten the answers) |
| `temperature`, `top_p`, `top_k`, `stop_sequences`, `stream` | As the chat fields, except that an omitted `temperature` is Anthropic's 1.0 (with the chat defaults top_p 0.95 and top_k 20, the models' own generation config), not greedy. Claude Code sends no sampling fields, and greedy thinking looped until `max_tokens` in its sessions (#11) |
| `trace`, `accept_top_k`, `accept_margin`, `chat_template_kwargs` | Sharar's extensions, as in chat |

`cache_control` markers are accepted and not acted on; the prefix cache
reuses shared prefixes regardless. `metadata`, other top-level fields,
`anthropic-version`, `anthropic-beta`, `x-api-key` and `?beta=true` are
ignored. HTTP 400 in Anthropic's error shape: `image` and `document` blocks
(text-only model), other content block types, `tool_choice` `any` or `tool`
(forced calls need constrained decoding), a missing `max_tokens`, and a
prompt that fills the context (`prompt is too long: N tokens > M maximum`).
Queue saturation returns 429 `rate_limit_error`, an unavailable engine 503
`overloaded_error`, and a stream already in progress an `error` event.
`/get_server_info` counts rejected forced tool choices and dropped server
tools under `messages_api` (`sharar_messages_*` in `/metrics`).

Content is ordered thinking (signature `sharar-unsigned`), text, then
`tool_use` blocks with parsed `input`. `stop_reason` is `end_turn`,
`max_tokens`, `stop_sequence` (with `stop_sequence`) or `tool_use`. Usage
reports the reused prefix as `cache_read_input_tokens` and the rest as
`input_tokens`; `cache_creation_input_tokens` is 0. A stream sends
`message_start`, `ping`, each block's start, deltas and stop (a thinking
block ends with its `signature_delta`; a tool call arrives whole as one
`input_json_delta`), then `message_delta` with the final usage and
`message_stop`. `message_start` counts the whole prompt as `input_tokens`
because the prefix match happens at admission.

To use Sharar from Claude Code, run `sharar launch claude` (interactive, or
with `claude` arguments such as `-p "..."`; it finds the serving target, `SHARAR_URL` picks another server;
`tools/claude-sharar.sh` is the same command). It
sets the variables below from Claude Code's gateway and model-configuration
docs and keeps its own config directory (`~/.claude-sharar`). Claude Code only accepts model IDs that contain
`claude` or `anthropic`; the server serves any name, so pick one that does:

```bash
export ANTHROPIC_BASE_URL=http://<host>:30001
export ANTHROPIC_AUTH_TOKEN=sharar                  # not checked
export ANTHROPIC_MODEL=claude-sharar-qwen3.8        # must contain "claude"
export ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-sharar-qwen3.8   # background tasks and titles
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144        # the server's --context-length
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1   # no telemetry or update checks
claude
```

Claude Code sends system messages inside `messages` (its environment block
after the first user turn); they are rendered in place as user turns in
`<system-reminder>` tags, since the template accepts one leading system
message and moving them would change the cached prefix. Several leading
system messages (OpenAI clients such as OpenCode) are merged. It asks for
32,000 output tokens; the server caps `max_tokens` at the remaining context
and answers `prompt is too long` when a conversation fills it.
The OpenAI chat endpoint handles role `developer` the same way as `system`.

The API's supported controls are checked in `tests/unit/test_chat_controls.py`
and `tests/unit/test_messages_api.py`;
failure, disconnect and drain behavior has separate lifecycle tests. There are no
request-level cache controls.

### Pi

Configure [Pi's `models.json`](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models.md) with provider `sharar`, API `openai-completions`, base URL `http://127.0.0.1:30001/v1`, and any API key.
For Qwen3.8, use `compat: {"thinkingFormat": "qwen-chat-template"}` and a model with `id: "qwen38-27b-nvfp4"`, `reasoning: true`, and `maxTokens: 8192`.
Match `contextWindow` to the server's `context_length` from `GET /get_server_info`, then run `pi --provider sharar --model qwen38-27b-nvfp4`.
For subagents, follow the [official example](https://github.com/earendil-works/pi/tree/main/packages/coding-agent/examples/extensions/subagent) and pin each child's `model` to `sharar/qwen38-27b-nvfp4`.
