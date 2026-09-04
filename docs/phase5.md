# Phase 5 notes: HTTP API

## What exists

- `sharar/api/server.py`: FastAPI application over one `EngineLoop` thread
  that owns the scheduler and steps it whenever work is queued. Handlers submit
  requests and await per-request asyncio queues that the loop fills after every
  step; client disconnects cancel the request. Endpoints:
  `/health`, `/v1/models`, `/get_server_info` (alias `/describe`) with the
  pinned profile, ledger and scheduler statistics, `/metrics` (plain text),
  `/flush_cache`, `/v1/tokenize` (alias `/tokenize`), `/generate` (SGLang
  style, streaming SSE with cumulative text and `meta_info`, or one JSON
  response), `/v1/completions` and `/v1/chat/completions` (streaming and not).
- Chat: the snapshot's Jinja template with `chat_template_kwargs`
  (`enable_thinking` defaults to true as in the reference), `tools`, `stop`
  strings (matched on visible text by `api/detok.py`), `stream_options.include_usage`.
  Reasoning is split from content by `api/parsers/reasoning.py` (the template
  opens `<think>` when thinking is on; `</think>` closes it) and returned as
  `reasoning_content`, with `reasoning_tokens` in usage. Tool calls in the
  model's XML block format are parsed by `api/parsers/tools.py` into OpenAI
  `tool_calls`, casting parameters by the tool's JSON schema; in streaming the
  block is buffered and emitted at the end with `finish_reason` `tool_calls`.
- `sharar serve --host --port` (default 127.0.0.1:30001). `tools/dev-shell.sh
  --serve PORT` publishes that port on loopback; `tools/verify-sharar.sh`
  checks health, model identity, the profile and a chat canary.
- The frozen harness (`bench/benchmark_text_server.py`, `bench/compare_runs.py`)
  runs against Sharar unchanged.

## Results

See the tables below, filled from the controlled matrix run recorded in
`${SHARAR_RESULTS}/controlled/controlled-27b-sharar-phase5-*` and compared
with the frozen reference run
`controlled-27b-sparklang-cli-candidate-20260903T005850Z`.

Controlled matrix through HTTP (256 output tokens, 5 measured batches per
cell, medians), run `controlled-27b-sharar-phase5c-20260904T034427Z`:

| Input | Conc | Sharar tok/s | Reference tok/s | Sharar TTFT p50 / p95 (s) | Reference TTFT p50 / p95 (s) |
|---:|---:|---:|---:|---|---|
| 128 | 1 | 11.94 | 12.74 | 0.19 / 0.19 | 0.13 / 0.13 |
| 128 | 4 | 43.7 | 46.4 | 0.37 / 0.39 | 0.31 / 0.31 |
| 2,048 | 1 | 11.51 | 12.22 | 0.86 / 0.86 | 0.81 / 0.81 |
| 2,048 | 4 | 38.4 | 40.7 | 2.48 / 3.28 | 2.34 / 3.07 |
| 8,192 | 1 | 10.22 | 10.81 | 3.55 / 3.55 | 3.33 / 3.33 |
| 8,192 | 4 | 26.8 | 18.3 | 10.4 / 13.9 | 8.46 / 31.7 |

Quality canaries 4/4. Minimum `MemAvailable` 36.7 GiB (floor 16). The
32,766-token boundary (31,742 in, 1,024 out): 9.62 tok/s, TTFT 16.3 s
(reference 10.18 tok/s), run `controlled-27b-sharar-phase5b-32766-20260904T033626Z`.

`bench/compare_runs.py` against the frozen reference reports FAIL on
throughput (-5.4% to -6.3% in five cells, the Phase 2 decode GEMV gap), on
TTFT for 128 x 1 (+60 ms, HTTP and tokenization overhead on a 130 ms
baseline) and 8,192 x 4 median (+23%, while p95 is -56%), and on serial hash
equality, which section 8 of the roadmap rules out for any second
implementation. These are known and recorded; Phase 6 freezes Sharar's own
run as the baseline so the comparator measures Sharar against itself from
then on.

Two server-side fixes came out of the first matrix run:

- The request handlers originally took the engine thread's lock to submit,
  which blocked the asyncio loop while an unfair lock let the engine
  re-acquire it every step. Concurrency collapsed to one request at a time
  and streamed tokens were delivered only at the end. The engine loop now owns
  the scheduler exclusively; the HTTP side posts to a thread-safe inbox and
  never blocks.
- Tokens were published once per step, after the decode that follows a
  prefill, adding a decode step to every first token. The scheduler now
  reports progress after each sub-step, and requests that arrive during a
  prefill step are admitted and prefilled before the decode step.

## Decisions

- One engine thread, asyncio front. The GPU never waits on the HTTP layer and
  the HTTP layer never blocks on a step; the only shared state is the
  scheduler under one lock.
- `/generate` keeps the SGLang response shape the harness and the parity
  oracles use (`text`, `output_ids`, `meta_info` with `prompt_tokens`,
  `completion_tokens`, `cached_tokens`, `finish_reason`, `e2e_latency`).
  Logprobs are not implemented yet.
- Tool calls stream as one delta at the end rather than incrementally; the
  model's XML format is not JSON-prefix parsable while incomplete.
- Repetition and presence penalties are still absent from the sampler.
