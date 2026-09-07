# Sharar

Sharar is a thin GB10 inference engine. Its first and currently only validated
model profile is:

- One NVIDIA GB10 (SM 12.1) on `aarch64`, one DGX Spark-class machine with
  unified LPDDR5X memory.
- `RadixArk/Qwen3.8-27B-NVFP4` at revision
  `319f741cce68d7914884900c138a1fbb70a42f30`.
- Local, offline, OpenAI-compatible text generation with speculative decoding.

It is written from scratch beside SGLang rather than carved out of it. The
`sparklang` repository holds the frozen SGLang-based reference container; Sharar
uses that container only as a correctness and performance oracle and never
imports the `sglang` package.

Target size is about 6K lines of engine code plus about 5K lines of vendored,
attributed Triton kernels, on top of `torch`, `flashinfer`, `triton`,
`safetensors` and `tokenizers`.

## Try it

From a DGX Spark with the model snapshot and the reference image present
(`config/gb10.env` points at them):

```bash
tools/dev-shell.sh --tty -- python3 -u -m sharar chat
```

Or serve the OpenAI-compatible API on loopback port 30001 and verify it:

```bash
tools/dev-shell.sh --serve 30001 -- python3 -u -m sharar serve --host 0.0.0.0 --port 30001
tools/verify-sharar.sh
curl -s http://127.0.0.1:30001/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"RadixArk/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"Hi"}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

Production: build the runtime image once (14 GB, CUDA 13 base plus the pinned
packages, no SGLang) and launch it with the preflight and the memory guard:

```bash
docker/build.sh                 # -> sharar:qwen38-27b
./run-sharar.sh --check         # host, GPU, driver, memory and snapshot preflight
./run-sharar.sh                 # container sharar-qwen38 on 127.0.0.1:30001 + guard; DFlash2 speculative decoding (15 drafts, 16 verify rows)
SHARAR_PROPOSER=mtp SHARAR_SPEC=3 ./run-sharar.sh   # the checkpoint's MTP head instead (no external draft needed)
tools/verify-sharar.sh
./run-sharar.sh --stop
```

Streaming (`"stream": true`), thinking (`enable_thinking` true, returned as
`reasoning_content`), tools (`tools`, returned as `tool_calls`), `stop`
strings, `/v1/completions`, `/v1/tokenize`, `/flush_cache` and the SGLang-style
`/generate` are supported. The frozen benchmark harness runs against it with
`python3 bench/benchmark_text_server.py --base-url http://127.0.0.1:30001 --container <name>`.
Chat accepts omitted/null `tool_choice` or `"auto"`, and omitted/null
`response_format` or `{"type":"text"}`. Other values return HTTP 400 before
prompt rendering or engine admission. Forced/disabled tool calls and constrained
JSON output require enforcement that is still tracked in the roadmap.

This loads the model (about 90 s), captures the decode CUDA graphs, and gives
an interactive `you>` prompt. Replies stream token by token and every turn ends
with a stats line: prompt tokens, time to first token, tokens generated and
decode tokens per second. `/reset` clears the history, `/exit` quits. Useful
flags: `--temperature 0.7` to sample, `--thinking` to enable the model's
reasoning mode, `--system "..."`, and `--prompt "..."` for a one-shot answer.

Other commands: `python3 -m sharar canaries --reference /results/parity/reference`
(quality canaries against the reference), `python3 -m sharar bench-decode
--prompts /results/parity/prompts.json` (decode throughput and determinism),
`python3 -m sharar parity --reference /results/parity/reference` (prefill
parity), `python3 -m sharar profile --prompts /results/parity/prompts.json`
(kernel time breakdown), `python3 -m sharar cache-check --prompts
/results/parity/prompts.json` (prefix cache correctness) and `python3 -m sharar
soak --prompts /results/parity/prompts.json` (memory soak), `python3 -m sharar
sched-bench` (the six matrix cells through the scheduler). All run inside
`tools/dev-shell.sh`.

## Layout

Contributors can run CPU checks without a GPU or model:
`bash tools/check.sh setup-cpu`. Start with [CONTRIBUTING.md](CONTRIBUTING.md)
and [the execution/ownership interfaces](docs/interfaces.md). The MTP and
n-gram, DSpark and DFlash2 adapters share target verification and state commit; n-gram is selectable
with `sharar serve --spec 3 --proposer ngram` as a contract/reference adapter.
Hardware research directions and decision criteria are in
[the GB10 experiment plan](docs/gb10-experiments.md).

| Path | Purpose |
|---|---|
| `sharar/` | The engine package (see the roadmap for the module plan) |
| `bench/` | Controlled six-cell benchmark and comparator, shared with the reference |
| `tools/` | Reference-container verification and the host memory guard |
| `tests/` | Unit tests and parity tests against the reference container |
| `config/` | Host-local paths (`gb10.env` is ignored by git) |
| `docs/` | Design notes that outlive a single phase |

## Status

The [two-path follow-up](docs/two-path-study.md) measures only 2–3 ms extra
target cost for a second candidate path with shared immutable history. Full
512/1,024-token continuations still fail the 10% completion-improvement screen,
so no two-path serving policy is added. Unsupported chat output controls now
return HTTP 400 explicitly; constrained-output enforcement remains separate work.

The [agent workload study](docs/agent-workloads.md) compares matched DFlash2
profiles at 8K/16K/near-32K with growing turns, shared-history branches and cache
churn. For 512-output cold requests, sustained decode is 34.24 / 30.74 / 28.00
tokens/s per request versus SGLang's 29.08 / 28.22 / 27.32. Cached retries benefit
more, but generation has no universal lead: SGLang wins the 16K/1,024-output cell.
Corrected task-format checks pass 8/12 in Sharar and 6/12 in SGLang; neither has
an all-pass agent episode. This is measurement work, with unchanged serving code
and no DFlash2 default promotion.

The [long-session cache work](docs/agent-context.md) passes 584 clean-image
CPU/GPU checks. With the same 12 state slots, a near-32K repeated prompt through
the serving scheduler falls from 14.48 s to 0.145 s TTFT; the fourth appended
agent turn now completes without exhausting snapshots. The 512-output study
uses a repeated paragraph and measures roughly 68–71 decode tokens/s per request,
with little generation-rate change. It is distinct from the agent-shaped review
workload above. Phase 8 prioritizes sustained generation before physical-memory
reclamation and final packaging.

The earlier [clean-image repeat review](docs/promotion.md) records:
414 measured requests, all 40 performance gates against repaired MTP passed,
and DFlash2 promotion withheld by the frozen quality rubric. Qualifying natural
answers finish 4.7–13.3% sooner than native SGLang DFlash2 in two windows; the
short four-request cell has no stable lead. Its controlled prefill partitions
change first tokens and acceptance tails. The new trace locates early
recurrent-projection rounding sensitivity; a numerical control remains open.
Production remains on the verified original image.

Phases 0–7D include lifecycle repairs, extension contracts, frozen local
workloads, native DSpark/DFlash2, draft-aware caching and experimental accepted-state
reconstruction. [Phase 7D evidence and remaining gates](docs/phase7d.md) record
the GB10 tests, 73% smaller verify buffers, lower measured load peak and pilot
workload comparisons. External drafts improve several tested workloads but
have not established a universal lead or passed every promotion gate. Historical
measurements below predate the repairs. The engine loads the checkpoint, prefills, and
decodes with CUDA graphs at 12.1 tok/s single-stream and 44.6 aggregate at
batch 4 (reference 12.7 and 46.4), passes the four quality canaries and the
documented numerical comparisons for the tested configurations, and keeps a
radix prefix cache with GDN state snapshots sized from a unified-memory ledger
(a re-sent 9.6K-token prompt goes from 4.4 s to 0.16 s to first token), and a
continuous-batching scheduler with chunked, batched prefill (8K prompts at
concurrency 4: 27 tok/s and 13.9 s p95 TTFT against the reference's 18 tok/s
and 31.7 s). See [docs/parity.md](docs/parity.md),
[docs/phase2.md](docs/phase2.md), [docs/phase3.md](docs/phase3.md),
[docs/phase4.md](docs/phase4.md), [docs/phase5.md](docs/phase5.md),
[docs/phase6.md](docs/phase6.md), [docs/phase7.md](docs/phase7.md). Sharar is
the production engine on the machine (`run-sharar.sh`, its own 14 GB image).
Since 2026-09-07 it serves with the pinned DFlash2 draft (15 proposals per
round, 16 verify rows) and accepted-state reconstruction by default, about
8x the plain decode rate on easy text and 2.5x on hard text, plus a
unified-memory cold tier that parks evicted sessions so a 30K-token history
resumes in 0.15 s instead of 18 s, and restarts in about 30 s (weights
load in seconds; [docs/restart-and-syncs.md](docs/restart-and-syncs.md));
the checkpoint's MTP head remains available (`SHARAR_PROPOSER=mtp`). [docs/default-dflash2.md](docs/default-dflash2.md)
records the promotion, [docs/cold-tier.md](docs/cold-tier.md) the wide block and the tier; [docs/comparison.md](docs/comparison.md) is the older
MTP comparison against SGLang, including GSM8K accuracy.
Phase 7A fixes terminal cleanup, speculative stop/state alignment, active
cache flush, streamed-history retraction, RNG/accounting, request bounds and
engine failure handling. [Fixes and proof](docs/phase7a.md): 103 CPU tests pass;
the GB10 run passes 194 tests, including 84 real-model lifecycle cases.
The source repairs have not been promoted to the serving image.
Phase 7B extracts the request-state ownership and proposer interfaces, adds
n-gram/feature-lifetime checks and contributor setup, and preserves measured
performance. The [mixed target experiment](docs/mixed-forward.md) failed arrival
first-token latency protection; its runtime hooks and failed scheduling policies
are archived in Git history. [One GDN prefill plan per forward](docs/prefill-plan.md) now removes
repeated metadata work across 48 layers: measured ordinary TTFT falls 18–29%
with identical output tokens. The [fresh plain/DFlash2 comparison](docs/competitive.md)
uses the same GB10, pinned weights, workloads and memory budget for both engines.
The [controlled verification study](docs/verify-cost.md) screens out compact
convolution integration: its exact prototype projects only 1.0–1.6% round-time
savings. The [NVFP4 follow-up](docs/nvfp4-cost.md) measures real packed inputs,
rejects narrow-tile kernels, and finds only 1.3–1.6% natural-round benefit from
a dispatch change, below its frozen gate. Serving remains unchanged.
The [runtime cleanup and MTP
repair](docs/promotion.md) remove failed serving hooks, externalize raw evidence
and reduce natural MTP B=4 round cost 2.76–3.75% with exact state comparisons.
Broader cache-pressure and quality gates remain open; unified-memory capacity
mechanisms (8) and remaining kernels/AOT (9) follow.
Read [roadmap.md](roadmap.md) before doing anything else.

## Attribution and license

Sharar is licensed under the Apache License 2.0. Vendored kernels and ported
code keep their original headers and are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
