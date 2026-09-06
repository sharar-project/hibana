# Agent workloads and sustained generation

This Phase 8 study compares the current Sharar DFlash2 candidate with the pinned
SGLang DFlash2 profile on GB10. It adds measurement code; serving source remains
`ba77a5f`. The previously validated runtime image is used with a read-only source
mount for the new benchmark. Runtime file hashes are checked against that image.
This study does not promote DFlash2 or replace the failed frozen promotion rubric.

## Fixed-work result

Four samples per original cell, from two independently loaded windows in
reversed engine order. The following single-request table reports **sustained
post-prefill decode TPS per request**; completion includes prefill and transport.
These are equal-token-count reviews with EOS ignored, not quality-qualified
agent-task speedups.

| Input tokens | Output/request | Cache | Sharar TTFT | SGLang TTFT | Sharar decode TPS/request | SGLang decode TPS/request | Sharar completion | SGLang completion |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 8,192 | 512 | cold | 3.059 s | 3.363 s | 34.24 | 29.08 | 17.988 s | 20.940 s |
| 8,192 | 512 | retry | 0.148 s | 0.915 s | 34.34 | 29.18 | 15.026 s | 18.430 s |
| 8,192 | 1,024 | cold | 3.068 s | 3.366 s | 34.02 | 33.42 | 33.126 s | 33.983 s |
| 16,384 | 512 | cold | 6.626 s | 7.191 s | 30.74 | 28.22 | 23.251 s | 25.320 s |
| 16,384 | 512 | retry | 0.155 s | 1.035 s | 37.18 | 28.21 | 13.901 s | 19.148 s |
| 16,384 | 1,024 | cold | 6.624 s | 7.198 s | 30.62 | 32.07 | 40.040 s | 39.093 s |
| 31,744 | 512 | cold | 14.623 s | 15.690 s | 28.00 | 27.32 | 32.876 s | 34.403 s |
| 31,744 | 512 | retry | 0.159 s | 0.667 s | 29.92 | 27.26 | 17.230 s | 19.417 s |

| Shared-history input/request | Output/request | Concurrent requests | Sharar total TPS | SGLang total TPS | Sharar batch completion | SGLang batch completion |
|---:|---:|---:|---:|---:|---:|---:|
| 8,063–8,064 | 512 | 4 | 93.23 | 87.83 | 21.967 s | 23.338 s |
| 16,255–16,256 | 512 | 4 | 93.81 | 83.84 | 21.831 s | 24.434 s |
| 31,615–31,616 | 512 | 4 | 87.82 | 83.59 | 23.321 s | 24.506 s |

The second table reports **total batch throughput**, with 512 output tokens per
request (2,048 per batch). Sharar's per-request median decode TPS is not higher
in every batched cell: it is 25.84 / 26.10 / 23.38 versus SGLang's 27.36 / 23.31 /
25.63. Batch completion and individual-request decode are different metrics.

The 16K/1,024-output cold cell favors SGLang. The original near-32K/1,024-output
comparison is excluded because SGLang clips its output count, as detailed below.
The generation lead is modest and workload-dependent; this does not establish
that Sharar is broadly much faster than SGLang. Identical retries reuse 8,128 /
16,320 / 31,680 tokens in Sharar versus 6,144 / 14,336 / 30,720 in this SGLang
profile. Their generated continuations can differ numerically, so a retry's
higher decode rate is not attributed solely to skipping prefill.

## Pinned execution

- Target: `RadixArk/Qwen3.8-27B-NVFP4`, revision `319f741cce68d7914884900c138a1fbb70a42f30`.
- Draft: `incoai/Qwen3.8-27B-DFlash2`, revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`.
- Sharar serving source: `ba77a5f`; runtime image `sha256:4a7206b66ead662a1dd2a5de811ef270f4dd7a27926273b2e5b812bbe0bade9b`.
- SGLang source: `5f55db35e926d50676f75b812640ea2410b0fe0e`; image `sha256:60166a99b6610e2b4b3febbeb6411e4cbc9338e996ab6c80d3339f86a7b5b9c3`.
- Benchmark commits: `186c6b4`, with profiler admission/priming corrected in `03b2ebf`. Earlier harness versions and every frozen input file remain in the external bundle.

## Frozen protocol

The workload and automatic quality rules are frozen before any model outputs.
Inputs combine the existing, checksum-pinned request-state source fixture with
varied deterministic test-log records and a repository review task. Historical
logs are truncated at a token boundary to set exact lengths; these are simulated
agent workloads, not a real repository repair or an external agent benchmark.

Two exclusive GPU windows reverse engine order. Each window measures two repeats
of each fixed-work cell and one growing episode per context. Both engines use
the same target/draft revisions, DFlash2 with seven draft tokens and eight target
verification rows, FP8 target KV, FP32 recurrent state, graph execution, 35,000
KV-token entries, eight CPU cores of quota and four OpenMP threads. The host
memory-drop/cgroup budget is 64 GiB with a 16 GiB `MemAvailable` floor.

Sharar uses 12 state slots and accepted-state reconstruction. SGLang uses its
validated `extra_buffer` strategy and 24 Mamba-cache entries, needed to admit
four running requests. Engine internals and cache implementations differ; the
physical budget and submitted work are matched. Graphs and JIT caches are warm;
this is not a cold process-start comparison.

Fixed work covers:

- 8,192 / 16,384 / 31,744 input tokens, 512 outputs, cold and paused identical retry.
- The same input lengths with 1,024 outputs, cold.
- Four branches from a primed shared history, 512 outputs per request. Actual
  branch input lengths are 8,063–8,064 / 16,255–16,256 / 31,615–31,616. Priming is
  recorded separately and excluded from branch latency.

The branches share 8,039 / 16,231 / 31,591 input tokens. Four independent 32K
histories do not fit this KV capacity. Fixed work ignores EOS; equal token count
is not equal answer quality. Generated token IDs and acceptance counts remain in
the raw evidence, including numerical differences between engines.

Growing episodes start at 7,168 / 15,360 / 30,720 input tokens and have a 256-token
output ceiling per turn, allowing room for actual answers and new tool data.
They request a strict JSON run summary, one exact `rerun_suite` tool call, updated
test totals, and an alternate branch from the original answer. The final branch
follows three unrelated 16K sessions to exercise cache eviction. Pauses are
250 ms; tool results are deterministic fixture messages. There is no external
tool execution, CPU contention or independent-process memory-pressure test.
Later inputs include actual prior output IDs and are saved per engine. These
are bounded tool-format/state-update checks, not comprehensive coding quality or
long-history retrieval evaluation. No failed quality case qualifies for a task
latency win.

## Timing and proof rules

HTTP TTFT and completion are per request. Sustained decode tokens/s excludes the
entire first SSE burst and its arrival time. It uses later observed bursts;
the harness never invents arrival times for individual speculative tokens.
Aggregate throughput divides all output tokens by batch completion time,
including prefill. Tokens per round is `(output_tokens - 1) / spec_verify_ct`,
which normalizes the engines' different `spec_accept_length` conventions.

Four fixed-work samples and two episodes per engine/context constitute a pilot.
Retain both windows and all failures; do not derive tail percentiles or a general
speedup claim. Growing episodes can have different input lengths if correct
answers differ in formatting, so compare task completion separately from fixed
identical-input work.

The direct generation profile runs 32 uninstrumented native rounds, eight rounds
with phase events and four with CUDA/CPU tracing at each context and batch 1/4.
These are successive parts of one growing continuation. It uses native proposals
and acceptance, without forced counts. Event spans overlap; parent spans include
their children. Trace kernel-duration sums are attribution, not physical DRAM
traffic. Direct prefill partitions can differ from the serving scheduler, so
these need not be the identical HTTP continuations. Instrumented measurements
never enter HTTP TPS.

The first diagnostic attempt stopped at 16K × 4: its one-shot primer retained
many interior recurrent snapshots, and direct admission pinned that cache path
before all four private state slots could be acquired. The HTTP matrix used
chunked priming and was unaffected. The corrected profiler uses the serving
2,048-token priming chunks and checks `can_admit` before admission, with the same
12 state slots and 35K KV entries. Preserve the partial attempt and rerun all
six diagnostic cells in `generation-profile-v2`; no serving capacity change is
made to obtain the profile.

After all timing, sixteen further native rounds per cell record a separate
candidate-coverage screen. Before target verification, select one alternative
token by its draft path probability. A hypothetical eight-row verifier would
keep the anchor, six greedy chain nodes and this alternative. Credit one possible
additional token when the first target correction reaches that alternative;
debit one when shortening the chain loses an eight-token native round. The
frozen screen requires at least 10% more potential tokens at both batch sizes
and no context/batch below -5% before implementing a standalone tree verifier.
This predicts neither tree arithmetic nor subsequent rollout, and measures no
tree-verification speedup. Draft predictions and emitted tokens are retained.

## Tool-template correction

The original `agent-v1` fixture supplied a schema in a user message but omitted
the snapshot template's native tool instructions. Sharar emitted a JSON
description of the requested call and failed the strict XML tool-call parser.
Preserve that result as a setup failure; do not diagnose it as a speculative
decoding regression or relax the parser to turn it into a pass.

The prospective `agent-v2` fixture renders the initial history with `tools`
through the snapshot template. Its strict quality rules are unchanged. A
separate natural-only pass runs once per context/profile in the second window.
Its outputs are not pooled with the original fixture. All fixed-work input IDs
remain identical between fixture versions. Neither fixture substitutes for
independent review or the outstanding production promotion gates.

## Context-headroom correction

The original 31,744-input/1,024-output cell reaches the declared 32,768-token
context exactly. This SGLang profile reserves two positions and returns 1,022
tokens. Those samples fail the output-count check and are excluded from speed
ratios; Sharar fulfills the requested 1,024. Preserve the clipped samples.

A prospective `agent-v3` follow-up uses 31,616 input tokens and 1,024 outputs
on both engines, with two measured repeats in the second window. It is a
separate prompt/cell and is not pooled with the original boundary case. The
current freezer uses this headroom and the corrected native tool template.
Reproduce the original failures using their frozen inputs from the evidence
bundle. A shared context limit does not imply identical output reservations.

## Task quality and corrected long-generation result

The canonical tool-template pass uses one episode per context/profile. Both
engines produce all requested outputs without transport/count errors and pass
all three requested tool-call checks. Sharar passes 8/12 turn checks and SGLang
6/12. Sharar emits unrequested tool calls for the three initial JSON summaries
and the near-32K alternate branch. SGLang has three unrequested calls and three
Markdown-fenced JSON failures. Neither engine has a fully passing four-turn
episode, so **no quality-qualified whole-agent completion-speed claim is made**.
These are unconstrained `/generate` requests with simulated tool responses.

Both engines begin the failed initial summary format with the target prefill
token, before speculative verification. This does not establish a rejection
algorithm bug. A separate code audit finds that Sharar's chat endpoint currently
does not consume `tool_choice` or `response_format`; the benchmark did not send
these fields. Reject unsupported controls explicitly before clients rely on
them, then validate enforcement with speculative state/acceptance semantics.

The original noncanonical fixture remains in the evidence: Sharar passes
18/24 turn checks across the two windows; SGLang passes 12/24. Do not pool
these with the corrected template or reinterpret failed formats as passes.

The output-headroom follow-up has two repeats in the second window:

| Input | Output/request | Concurrency | Sharar TTFT | SGLang TTFT | Sharar decode TPS/request | SGLang decode TPS/request | Sharar completion | SGLang completion |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 31,616 | 1,024 | 1 | 14.586 s | 15.610 s | 31.20 | 29.97 | 47.374 s | 49.745 s |

The original four SGLang 31,744/1,024 samples return 1,022 tokens and are
excluded from ratios. All original Sharar fixed-work samples fulfill their
requested counts. The corrected headroom cell fulfills 1,024 in both engines.
All failures, actual growing inputs, output IDs and cache counts are retained.

## Generation costs and candidate-path decisions

The corrected direct profile completes six context/batch cells and 60 native
rounds per request. It confirms that target verification dominates the round;
this is where reusing a weight pass for more useful tokens has the most scope.
The table pairs uninstrumented round medians with separate diagnostic event
spans. They describe different successive rounds and must not be summed or
treated as a precise elapsed-time decomposition.

| Context | Batch | Whole round | Draft span | Target verification | Recurrent reconstruction | Useful tokens/request/round (32 uninstrumented rounds) |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 1 | 113.30 ms | 19.07 ms | 87.68 ms | 1.38 ms | 2.81 |
| 8,192 | 4 | 132.54 ms | 21.42 ms | 99.70 ms | 5.34 ms | 3.19 |
| 16,384 | 1 | 115.10 ms | 20.23 ms | 89.19 ms | 1.41 ms | 3.09 |
| 16,384 | 4 | 135.67 ms | 21.70 ms | 103.41 ms | 5.35 ms | 2.90 |
| 31,744 | 1 | 119.20 ms | 20.35 ms | 93.06 ms | 1.43 ms | 2.91 |
| 31,744 | 4 | 142.31 ms | 22.22 ms | 108.97 ms | 5.41 ms | 3.47 |

NVFP4 matrix-multiplication kernel durations total 49.86–55.14 ms per traced
round. Kernel sums are attribution, not bandwidth counters. Increasing context
from 8K to near-32K adds about 5.9 ms to the single-request round and 9.8 ms at
batch four in this diagnostic. Direct continuations differ from the HTTP run;
the HTTP tables remain the speed comparison.

Three policies were fixed before any profiler outputs. The first reserves one
of the seven draft nodes for an alternative, shortening the main chain. The
second uses dynamic programming to maximize expected accepted-prefix length
under the existing DFlash2 conditional lattice. The third retains two complete
seven-token chains; its hypothetical target work grows from eight to sixteen
rows. None changes actual proposals or runs an alternative verifier in this study.

| Offline screen | B=1 potential token change | B=4 potential token change | Frozen general screen |
|---|---:|---:|---|
| One alternative within eight target rows | +10.83% | -2.81% | Fail |
| Expected-prefix path selection | +2.78% | -0.25% | Fail |
| Two complete paths | +23.15% | +7.19% | Fail |

The frozen general screen requires +10% at both batch sizes and no context/batch
below -5%. The single-alternative screen has 240 request-round observations.
The two path screens use 216 observations with eight known subsequent target
tokens; 24 terminal observations lack that lookahead and are excluded by the
frozen rule. Native greedy-path coverage matches native emissions on all 216.
The expected-prefix calculation also passes 24 exhaustive CPU optimality checks.

All three general screens fail. In particular, the two-path result is **coverage,
not measured speed**: it ignores the extra target-row cost, subsequent rollout
and possible target rounding changes. No runtime policy or serving hook is added.

The +23.15% single-agent coverage result motivates a **new, narrower prospective
study**, not a retroactive pass. First measure target verification at physical batch sizes one and two,
then repeat coverage over full 512/1,024-token
held-out single-agent continuations at 8K/16K/near-32K. Keep the complete greedy
chain and one conditional alternative, share immutable history, and account for
target KV/recurrent-state handling and winner-only commit. Existing accepted-state
reconstruction is the relevant building block. Do not implement a new serving
mechanism unless fresh cost/coverage evidence supports at least 10% useful-turn
completion reduction, all quality/state/memory checks and protected concurrency-four
behavior. Four running agents keep their existing verification capacity.

## Validation and memory

The HTTP study records 244 measured requests plus 84 warmup/priming/churn requests.
Sharar fulfills every requested fixed count. Four original SGLang boundary
requests are clipped; both corrected headroom runs fulfill their counts. The
95 serving-source hashes match the prior clean runtime image. CPU checks pass
268 tests with six GPU-only skips; the profiler-setup correction is separately
validated by all six real-GPU cells and clean pool accounting. The unchanged
runtime retains its earlier 584-test clean-image proof; that suite is not rerun
for measurement-only code.

The main HTTP windows show peak host `MemAvailable` drops of about 31.2 GiB
for Sharar and 43.7 GiB for SGLang. The lowest main-window `MemAvailable` is
69.62 GiB. These are host-level changes including load/graph startup and file-cache
effects, not reconciled per-process footprints. All jobs enforce the 64 GiB
drop/cgroup budget and 16 GiB floor. Clock/power/temperature samples are retained;
clocks are not locked. Physical reclamation, external tool/CPU pressure and
64K/128K context profiles remain unvalidated.

Production is restored to its original image with the memory guard active and
health/model/canary checks passing. DFlash2 remains an unpromoted candidate.

## Reproduction

Freeze inputs with the pinned tokenizer before starting either server:

```bash
python3 -m bench.agent_workloads --freeze /path/to/inputs.json
python3 -m bench.agent_workloads --inputs /path/to/inputs.json \
  --url http://127.0.0.1:PORT --profile sharar-dflash2 --output /path/to/new-run
python3 -m bench.agent_profile --inputs /path/to/inputs.json --output /path/to/new-profile
```

Use the exclusive GPU guard and restore production after each window. Commands,
inputs, answers, traces, resource samples and source hashes belong outside Git.
The [compact evidence manifest](agent-workloads-results.json) identifies the
sealed external bundle, including corrected and failed attempts.
