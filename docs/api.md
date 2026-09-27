# HTTP API

The launcher exposes the API at `http://127.0.0.1:30001`; `./sharar-ctl chat` is a terminal client for it. The supported target
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
| `temperature`, `top_p`, `top_k`, `seed` | Validated sampling parameters |
| `stop` | A nonempty string or list of nonempty stop strings |
| `n` | 1 to 16 choices (non-streaming). The prompt is prefilled once by a one-token primer and published to the prefix cache; the n samples resume from it (at most 64 tokens re-prefilled each) and decode side by side, four at a time. With a `seed`, choice i uses seed + i; use a temperature above 0 for distinct choices |
| `chat_template_kwargs.enable_thinking` | Thinking defaults to true; false disables it |
| `reasoning_effort` | `low`, `medium`, `high` select the template's reasoning instruction (`low`, `medium`, `xhigh`; `high` and `max` map to `xhigh`, the default); unknown values keep the default; `chat_template_kwargs.reasoning_effort` takes precedence |
| `tools` | Tool schemas passed to the template; generated calls are parsed |
| `tool_choice` omitted/null/`auto` | Unconstrained model choice |
| `tool_choice: "none"` | Omit tool schemas from the prompt and tool-call parsing |
| `response_format` omitted/null/`{"type":"text"}` | Ordinary text output |
| `trace` | `false` excludes this request from the opt-in [trace sink](operations.md#request-traces); accepted on every generation route |
| `accept_top_k`, `accept_margin` | Relaxed speculative acceptance, off unless both are set (sampled requests may set them too; measured on 2026-09-23 at temperature 0.6 they cost about two GSM8K and three HumanEval points for +4% throughput, so the server default applies to greedy requests only unless `SHARAR_ACCEPT_DEFAULT_SAMPLED=1`): a drafted token is also accepted when the target ranks it within its top `accept_top_k` and within `accept_margin` logits of the argmax (a log-probability ratio; 1.0 means at least 37% of the top probability). Faster, and no longer the target's exact greedy continuation; the per-request `spec_relaxed_tokens` counts such tokens. **Default: the server applies top-2 within 1.0 logit to greedy requests that set neither field** (`--accept-default`, `SHARAR_ACCEPT_DEFAULT`; `off` keeps everything exact); send `accept_top_k: 1` to get the exact greedy continuation for one request. Accepted on every generation route |

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
| `temperature`, `top_p`, `top_k`, `stop_sequences`, `stream` | As the chat fields; omitted sampling fields keep the chat defaults (greedy), not Anthropic's temperature 1 |
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

To use Sharar from Claude Code, run `tools/claude-sharar.sh` (interactive, or
with `claude` arguments such as `-p "..."`; `SHARAR_URL` picks the server). It
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
failure, disconnect and drain behavior has separate lifecycle tests. Current
request-level cache modes/priorities are only a [proposal](request-cache-policy.md).

### Pi

Install the Pi coding agent:

```bash
npm install -g --ignore-scripts @earendil-works/pi-coding-agent
```

Configure `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "sharar": {
      "baseUrl": "http://127.0.0.1:30001/v1",
      "api": "openai-completions",
      "apiKey": "sharar-local",
      "compat": {"thinkingFormat": "qwen-chat-template"},
      "models": [{
        "id": "qwen38-27b-nvfp4",
        "reasoning": true,
        "contextWindow": 262144,
        "maxTokens": 8192
      }]
    }
  }
}
```

`apiKey` can be any string. Set `contextWindow` to the server's
`context_length` from `GET /get_server_info`; it must match the launch
configuration (`SHARAR_MAX_CONTEXT`, default `262144`). Sharar ignores the
request's `model` field on `/v1/chat/completions`.

Run Pi from the directory you want it to work in:

```bash
pi --provider sharar --model qwen38-27b-nvfp4
```

#### Subagents

Pi has no built-in subagents; we verified the official `examples/extensions/subagent` bundled with Pi 0.87.1, loaded automatically from `~/.pi/agent/extensions/subagent`. For a fresh install with an unused backup path:

```bash
pi_sa_pkg="$(npm root -g)/@earendil-works/pi-coding-agent"
cp -a "$HOME/.pi" "$HOME/.pi.bak-sa1"
mkdir -p "$HOME/.pi/agent/extensions/subagent" "$HOME/.pi/agent/agents"
cp "$pi_sa_pkg/examples/extensions/subagent/"{index.ts,agents.ts} "$HOME/.pi/agent/extensions/subagent/"
cp "$pi_sa_pkg/examples/extensions/subagent/agents/"{scout,planner,reviewer,worker}.md "$HOME/.pi/agent/agents/"
```

Using the `sharar` provider above, replace `model:` in each copied agent profile with `model: sharar/qwen38-27b-nvfp4`, merge `"defaultProvider": "sharar"` and `"defaultModel": "qwen38-27b-nvfp4"` into `~/.pi/agent/settings.json`, and keep the default `agentScope: "user"` so parent and children use Sharar.
Verified single, parallel and chained subagents and a real bug-fix task, with 1–8 subagents per call.
Within one parallel call, the extension runs up to four subagents at once; Sharar also serves up to four requests at once, and extra work waits for a free slot.
The model sometimes names an agent that does not exist or uses `python` instead of `python3`, and corrected itself in the recorded runs; these were model behaviours, not server errors.
