# Phase 7C: local workload pilot and first GB10 experiment

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

This completes the initial 7C investigation, workload freeze and experiment
selection. Full competitive comparison, cache-pressure serving runs, physical
DRAM counters and a production workload comparator remain open 7C gates.

## Repository and observability fixes

The first remote CPU workflow failed because `.gitignore` contained `parity/`.
Git applied that rule to `sharar/parity/` as well as root-level artifacts, so
the locally present Python package never reached a clean checkout. Anchoring
the artifact rule to `/parity/` and tracking the source fixes the missing import.
The repair was tested from an exported Git tree, with no host configuration or
ignored package files: 129 passed, 9 skipped. It is commit `2ee8694`. The restored
historical diagnostic scripts are not all rerun by this CPU import check.
In particular, the old `parity.blocks` sub-operation probe still refers to
the former fused `in_qkvz`/`norm` members. Its migration remains a cleanup task;
it is not used as evidence here. The new cache trace and its GPU regression
use the current split projections and gated-normalization path.

`Scheduler.Metrics` previously retained every completed request's TTFT and
sorted the entire history on each statistics update. It now retains the most
recent 2,048 values. Lifetime request counters remain cumulative; the API
reports the TTFT window count and capacity so percentiles have an explicit
scope. The regression checks eviction of old samples and filtering of NaNs.

## Cold/split/cache diagnosis

`sharar.parity.cache_trace` observes the normal in-place prefill path, cloning
only the last row at each layer and selected projection/core outputs. These
instrumented forwards are correctness evidence, not timing samples.

For the fixed 129-token encoding of `hello ` repeated 128 times, compare:

1. A single cold prefill.
2. A live 128-token prefill followed by its last token, with no cache match.
3. The same live split with the intermediate FP32 recurrent state rounded
   through BF16 before continuation.
4. Manual restoration of the cold run's exact prefix KV, recurrent checkpoint
   and convolution window, followed by the last token.
5. Radix-cache restoration of those same tensors, followed by the last token.

The cold run emits token 109266; all four split/restored runs emit 248046.
Manual and radix restoration produce **bit-identical outputs at all 64
layers**, and restored recurrent/convolution tensors match the saved tensors
bit for bit. The live split alone reproduces the mismatch. This rules out the
radix copy/loan machinery as the cause of this reproducer.

The first differing operation is layer 0's FP8 QKV projection: M=129 uses
`torch._scaled_mm`, whereas M=1 uses the custom streaming GEMV. The output
difference is at most 0.0625, relative RMS 0.000171. The complete layer streams
remain equal through layer 1; the first differing post-block stream is layer 2
(maximum 0.00244140625). Later layers amplify the difference. The diagnostic
also holds the quantized input row, weights and scales fixed and compares both
projection shapes with a dequantized FP32 matmul.

Against FP32, wide/narrow relative RMS errors are 0.001704155 and 0.001704153,
respectively, with the same maximum error 0.1187668. Against BF16-rounded FP32,
the custom one-row GEMV differs by at most 0.00024414, while the wide GEMM
differs by 0.0625. This is consistent with accumulation/rounding variation at
the first differing operation; this probe does not show an accuracy defect
in the custom GEMV. No kernel dispatch or model numerics were changed to force
the two token continuations to agree.

Intermediate chunk checkpoints still round FP32 recurrent state through BF16
`h` storage, while a live split keeps its final state in FP32. That precision
loss is separately visible in the saved per-layer state comparisons. Full
attention also uses BF16 ragged KV on a cold prefill versus FP8 paged KV after
a prefix. Neither fact should be mistaken for an ownership error or erased
from the quality requirements. The result isolates the first numerical-path
difference; it does not establish universal cold/cache token equality or
prove that all future cache discrepancies are harmless.

The GPU regression asserts exact manual/radix restoration across all layers.
It does not assert that the old mismatching output tokens must remain forever.

## Frozen local workloads

`bench/workloads/local_v1.json` owns seven tasks: chat, a bounded Python clamp
function, a weather tool call, JSON, a reasoning answer, seeded sampled chat,
and a multi-turn review of real request-state code. The code fixture is frozen
at `7aebbe6`, with a checksum; future source edits do not rewrite the prompt.
Every run materializes exact token IDs using the pinned tokenizer/template.

The matrix includes cold/repeated/session prefixes, staggered arrivals at
0/50/100/150 ms, streaming-client cancellation after eight consumed tokens,
and output budgets 1–5. Natural EOS remains enabled. A short JSON repeat is
an explicit expected miss: its 46-token prompt has no eligible interior
64-token checkpoint. The initial pilot incorrectly required that case to hit;
its failed harness output is preserved separately, and the corrected pilot
reruns the unchanged prompts with the corrected cache expectation.

Each execution mode has one warmup and three measured repetitions for the
serial cases, three mixed-arrival groups, cancellation and five budget checks.
The HTTP client records TTFT, completion time, actual output IDs/text, cache
hit length, per-request speculation metadata and cumulative token-delivery
events. Gaps are measured between delivered SSE bursts. Several speculative
tokens in one burst do not receive invented individual timestamps.

The automatic checks exercise clamp behavior on seven inputs, exact JSON,
tool name/arguments and the final reasoning answer. Chat and session outputs
have explicit review rubrics. They are quality smoke tests, not broad quality
or stochastic-speculation validation. Sampled requests still use ordinary
decoding. Three repetitions do not support p95/p99 or statistical superiority.

The manifest freezes a 10% selected-workload completion improvement, at most
5% protected-workload regression, the 16 GiB host floor and quality checks.
An alternative capacity gate requires at least 25% lower total verification
snapshot allocation at fixed batch/width, exact FP32 committed state and the
same protected latency requirement. Confirmatory independent runs and a
workload comparator are required before promotion.

The corrected pilot completed **156 HTTP requests**, including primers,
warmups, partial-output checks and measured requests: 78 in each execution
mode. There were **zero request failures and 60/60 automatic quality checks
passed**. The audit verifies the exact expected request-ID set, both successful
client disconnects and every output budget, not just an aggregate count.
The 52 free-form rubric samples had 16 unique outputs, read by the assistant;
one chat answer imprecisely described reuse "for every new token". The saved
rubric review retains that caveat and is not an independent human evaluation.

Some measured medians below illustrate the baseline, with three repetitions
per row. These are the existing plain/MTP modes, not a new optimization result.
Different token counts explicitly show why a completion ratio alone cannot
isolate engine speed. The sampled case uses ordinary decoding in both modes.

| Workload | Plain output tokens | MTP output tokens | Plain completion (s) | MTP completion (s) |
|---|---:|---:|---:|---:|
| Chat, cold | 78 | 73 | 6.445 | 3.781 |
| Coding, cold | 35 | 35 | 2.925 | 1.135 |
| Tool call, cold | 39 | 39 | 3.336 | 1.305 |
| JSON, cold | 30 | 30 | 2.535 | 1.019 |
| Reasoning, cold | 48 | 82 | 4.059 | 2.881 |
| Sampled chat, cold | 64 | 64 | 5.360 | 5.326 |
| Session, cold | 69 | 64 | 6.652 | 3.753 |
| Session, repeat | 71 | 62 | 5.935 | 2.904 |
| Session, prior-turn prefix | 64 | 68 | 5.387 | 3.226 |

The session prompt contains 2,172 tokens. Repeated-request hits reuse 2,112;
the frozen prior-turn seed reuses 2,136. In MTP mode, median TTFT is 0.993 s
cold, 0.150 s repeated and 0.146 s with the prior-turn seed. Draft-prefix
validity/restoration remains 7D work; these examples do not certify it for
all sessions or establish deterministic draft acceptance across fresh runs.
The frozen manifest SHA-256 is
`1f5df2ca0c2bc88e42ecfb17d9b3bdb5aeaa6f2d77c5f146f79167c87828fdcf`.

## Profiling and experiment selection

`bench/gb10_profile.py` records plain and MTP k=3 at batches 1/4 and prefixes
128/8,192. Each cell has three independent fixed-prefix four-step timings,
then a separate four-step CPU/CUDA trace with proposer, verifier and acceptance
scopes. It saves every emitted sequence so differing acceptance/work is visible.
Profiles use the union of device kernel/copy intervals; overlapping kernel
durations are not added as if they were serial work. Instrumentation and the
explicit end-of-step synchronization affect trace gaps, so unprofiled timing
is reported separately.

| Mode | Prefix | Batch | Unprofiled ms/step, median | Trace device activity |
|---|---:|---:|---:|---:|
| Plain | 128 | 1 | 82.79 | 96.3% |
| Plain | 128 | 4 | 89.38 | 97.0% |
| Plain | 8,192 | 1 | 85.05 | 95.9% |
| Plain | 8,192 | 4 | 94.89 | 95.9% |
| MTP k=3 | 128 | 1 | 109.40 | 97.5% |
| MTP k=3 | 128 | 4 | 129.89 | 98.1% |
| MTP k=3 | 8,192 | 1 | 112.60 | 96.3% |
| MTP k=3 | 8,192 | 4 | 137.34 | 97.6% |

CPU phase scopes include waiting for previously queued device work. For
example, the acceptance scope spans the synchronous result download; its
duration is not pure acceptance bookkeeping or removable CPU time. The idle
intervals and device events, rather than that scope duration, drive the choice.

The full FP32 verification snapshot allocation has shape
`[48 layers, 4 rows, 4 inputs, 48 heads, 128, 128]`: **2.25 GiB** for recurrent
state, plus **45 MiB** of convolution windows. These are measured tensor
allocations, not physical DRAM traffic counters. The first experiment should
test storing compact verified recurrence inputs and reconstructing only the
accepted recurrent state, preserving the current full-snapshot path as the
reference. The detailed keep/remove decision is in
[gb10-experiments.md](../gb10-experiments-2026-09.md).

## Reproduction and evidence

Run `python3 bench/phase7c.py --output /path/to/new-proof` inside the prepared
GB10 environment with exclusive model memory and a host-floor guard. See
[bench/README.md](../../../bench/README.md) for the standalone HTTP client and limits.

Raw evidence lives under
`$SHARAR_RESULTS/reviews/2026-09-05/phase7c/`: command/source identities,
cache/layer comparisons, exact workload manifests, all stream events, quality
outputs, compressed Chrome traces and host memory/power samples. The validation
image is `sha256:b893cdc2e3171bc2b977949305535e1cc62ebb41cee7d607638d628588d50b1b`,
with source mounted read-only, 35,000 KV pages, 16 state slots, batch capacity 4,
graphs enabled and the pinned Qwen revision. The test container has no external
network; HTTP traffic stays on its loopback interface. GPU clocks are observed,
not locked. No new SGLang/DSpark/DFlash2 comparison is claimed here.

`measurements/audit.json` checks full request/profile coverage;
`measurements/{plain,mtp}/requests.jsonl` contains all outputs and delivery
events; `measurements/profiles/` has all eight traces and raw timing samples;
`measurements/cache-trace.json` includes the FP32 projection comparison.
`initial-measurements/` and `initial-measure*` retain the first pilot with the
incorrect short-repeat expectation. Minimum sampled `MemAvailable` during
the corrected workload/profile run was **73.72 GiB**. The source CI repair
also passed [GitHub CPU CI](https://github.com/abdelfattah-lab/sharar/actions/runs/33940611492).

Final source validation: **135 CPU tests passed, 10 skipped** from the exported
candidate Git tree; **235 tests passed in the GPU environment**, including
**90 real-checkpoint cases**. The latter includes the new 64-layer manual/radix
comparison, existing stop/commit/continuation checks, eager/graph coverage and
both 32K boundary cases. GPU-suite minimum `MemAvailable` was **73.57 GiB**.
The only serving-path change is bounded metric history; the new diagnosis and
profiling code is opt-in. The same serving container and memory guard are
restored after exclusive validation; this work does not promote a new image.
