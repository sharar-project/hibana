# Runtime cleanup and clean-image promotion review

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

This review keeps the shipped scope to the pinned Qwen model on GB10, removes
rejected serving experiments, and repairs MTP feedback before evaluating the
DFlash2 candidate. Default promotion depends on the measured gates below.

## Changes

- Remove mixed forwarding from the scheduler, executor, StateManager and model,
  along with the failed width and latency policies and their public options.
  Their source and benchmark runner remain reproducible at `56605f0`.
- Move checkpoint construction to `proposers/factory.py`. Generic execution
  validates declared features and maximum depth. Normal startup captures only
  the selected verification width; controlled benchmarks explicitly opt into
  additional widths. Preserve draft-aware cache ownership, sampled verification,
  accepted-state reconstruction and the shared GDN prefill plan.
- MTP accepted feedback uploads all pending/accepted input tokens once through
  bounded pinned staging. A copy-completion event protects staging reuse for
  direct callers. Per-request KV correction preserves the original
  row shape, BF16 arithmetic and the required target-conditioned KV
  for accepted inputs, including the fully accepted bonus input. A captured-KV
  prototype adds no consistent gain beyond pinning and is archived in the
  benchmark instead of becoming another runtime path.
- Replace ten large evidence bundles in the current Git tree with compact
  manifests: **7,232,106 bytes to 15,925 bytes**. Every original bundle is preserved
  outside Git and verified by SHA-256. Markdown reports retain the results;
  manifests locate the full evidence. CI enforces a 32 KiB manifest limit.
  This does not rewrite or shrink existing Git history.
- The runtime shrinks from **15,340 to 15,100 Python lines**; `executor.py`
  shrinks from **570 to 493 lines**. These counts include vendored runtime kernels
  and exclude benchmarks/tests. The engine directory shrinks from **2,484 to
  2,268 lines** (8.7%).
- Hold additional dense/MoE models and generic target abstractions until the
  current default is proven. Update the roadmap and current interface guide;
  mark obsolete reproduction commands as historical.

## Validation

The CPU suite passes **251 tests**, with six GPU-only skips. The initial clean
image, including the subsequently archived KV graph prototype, passes **573
CPU/GPU tests**, including real state/cache continuations,
stop and output-budget boundaries, retraction, stochastic verification, API
validation, shutdown draining and canaries. Tests for removed experiments were
removed with those experiments; applicable reservation coverage was retained.

The Dockerfile dependency stages were rebuilt with `--no-cache` from the CUDA
13.0.3 development base. The pinned cu130 dependency install, `pip check` and
runtime imports all succeed. Final benchmark/test updates reuse those freshly
built dependency stages. The final pinned-feedback image is:

`sha256:2addcbfc71a422dffc30a51ebc2cc0cb1a91b4f5592ae09edd87843e19bfca3d`

The initial graph-prototype image was
`sha256:225273d394c467f6f76365f1e6a1ef2dcae74e05423d09a9c8018f0933d30823`.
Removing that MTP-only path leaves the validated target and DFlash2 code unchanged.
The final image passes **269 focused checks**: the complete CPU/API suite plus
cached MTP EOS/budget tests at batches 1, 3 and 4 under both state strategies.

The final image contains all 95 current runtime Python files, checked against the build
context. Clean-image validation and Sharar performance runs use the image's
runtime directly, without a source overlay. Independent measurement/control
scripts and output directories are mounted separately.

An earlier source-overlay test launch omitted the Python main guard, causing
multiprocessing spawn to re-enter pytest during the SIGTERM test. It recorded
564 passes and that launcher failure. The corrected clean-image invocation
passes all 573 checks, including SIGTERM; no serving change was required.

## MTP repair results

Twenty cells cover 128/8,192 context tokens, batch one/four, native acceptance
and forced accepted counts 1–4. Each cell runs legacy/pinned/graphs/graphs/pinned/
legacy in one warmup block and three measured blocks: **360 timed rounds plus
120 warmup rounds**. All **480 round comparisons** have identical emitted IDs,
committed lengths, full target recurrent/convolution state, corrected draft KV
and pending target hidden rows. Forced counts measure cost and state alignment;
they are not answer-throughput measurements.

Natural whole-round medians at MTP k=3, including proposal, verification and
acceptance, with the same prepared context:

| Context tokens | Concurrent requests | Blocking uploads (ms) | Pinned batch upload (ms) | Round reduction |
|---:|---:|---:|---:|---:|
| 128 | 1 | 110.760 | 109.383 | 1.24% |
| 128 | 4 | 128.559 | 123.743 | 3.75% |
| 8,192 | 1 | 112.837 | 111.969 | 0.77% |
| 8,192 | 4 | 132.099 | 128.455 | 2.76% |

Pinning supplies the material gain: **3.64–4.82 ms per natural B=4 round**.
The additional KV graph prototype is 0.51% slower than pinning at 128/B=4 and
0.11% faster at 8K/B=4; results across forced counts are also mixed. It stays in
`bench/mtp_feedback.py`, with no runtime field, mode or extra graph capture.
The required target-conditioned KV correction remains asynchronous on the
executor's stream. These costs do not directly predict DFlash2 acceleration.

## DFlash2 cache and lifecycle evidence

Additional real-model checks pass at 8,192 and 32,766 input tokens with two
outputs, both cold and cached. Missing/evicted draft features safely fall back
to ordinary target decoding and match its output IDs for greedy and sampled
requests. Four active prefix readers survive eviction and cache flush, then
retraction preserves delivered output without duplication. Cancellation releases
its request. After flushing, all 34,999 usable KV pages, 11 state slots and four
draft slots are returned.

The 32,766-token repeat reuses only a **20,480-token checkpoint** with the fixed
12-slot pool. It is correct, but must redo over 12K prompt tokens. Endpoint versus
interior checkpoint retention under tight slot budgets is a concrete later
cache-efficiency task; this run does not establish a long-session capacity win.

All **16 historical parity gates pass**. The full clean-image validation window
stays above **75.67 GiB MemAvailable**, within the declared budget. Production
restoration passes its health/model/canary check after the window.

## Frozen comparison protocol

Two independent windows reverse profile order. Each compares repaired Sharar
MTP k=3, Sharar DFlash2 k=7 and native SGLang DFlash2 with eight-token blocks.
Both Sharar controls use accepted-state reconstruction. The target, external
draft, tokenizer inputs and natural workload definitions remain pinned.

Each fixed-work cell uses 128/2,048/8,192 input tokens, **256 output tokens per
request**, concurrency one/four, one warmup and three measured batches per
window. EOS is ignored in these diagnostic cells. Aggregate output throughput
is total emitted tokens divided by client batch makespan, including prefill.
Natural tasks use their original stopping, sampling and quality checks.

Both engines get eight CPU cores, four OpenMP threads, 35,000 target KV tokens,
FP8 target KV, FP32 recurrent state, 32,768 context capacity and four running
requests. Sharar uses 12 state slots; SGLang uses 24 to provide actual running
capacity four. The fresh SGLang comparison uses 2K prefill chunks; the prior
8K results remain historical, so this is not a per-cell best of both settings.
SGLang's default sampler ignores per-request seeds without its deterministic
mode; sampled answers are qualitative comparisons, not matched-seed timings.

Each exclusive run enforces a 64 GiB cgroup cap and host `MemAvailable`-drop
budget, plus the 16 GiB absolute floor. These are common enforced budgets, not
identical allocations or a sum of overlapping CUDA/RSS measures. Restore the
original production image and memory guard after each validation window.

Promotion requires passing correctness and frozen quality checks. Relative to
repaired MTP, no fixed cell may lose over 3% throughput or regress TTFT over 15%;
protected natural completion may regress at most 5%. A selected natural workload
must improve at least 10%. Failed quality cells cannot supply speed wins. Two
windows with six measured batches per cell remain limited evidence for tails;
they do not establish production p95/p99 or a universal speed lead.

## Repeated fixed-work comparison

Both engines use native DFlash2 with seven proposals per verification block.
Each value below is a **window median of three measured batches**, in aggregate
output tokens/s, including prefill and the final request's tail. W1/W2 preserve
the two independent loads instead of hiding their disagreement in one median.

| Input tokens/request | Output tokens/request | Concurrent requests | Sharar DFlash2 W1 / W2 | SGLang DFlash2 W1 / W2 |
|---:|---:|---:|---:|---:|
| 128 | 256 | 1 | 63.71 / 63.74 | 19.57 / 19.69 |
| 128 | 256 | 4 | 215.76 / 97.53 | 55.62 / 133.23 |
| 2,048 | 256 | 1 | 56.34 / 56.38 | 51.91 / 51.51 |
| 2,048 | 256 | 4 | 113.49 / 91.29 | 79.78 / 60.17 |
| 8,192 | 256 | 1 | 29.82 / 30.01 | 23.16 / 23.15 |
| 8,192 | 256 | 4 | 50.24 / 46.98 | 39.35 / 39.28 |

Sharar wins five cells in both windows. The short four-request cell reverses:
Sharar is 3.88 times SGLang's throughput in W1, but **26.8% lower in W2**.
Its individual measured batches span **76.98–217.97 tokens/s**; SGLang spans
**54.99–179.81**. This is not a stable short-batch speed lead.

Request-level evidence identifies low-acceptance tails. Sharar's slow W2
batches have three requests completing in 35 verification rounds and a fourth
taking 85, 86 or 111. SGLang's slow batches include a request taking 139–142
rounds, while its common fast requests take about 35. Output continuations also
change. These fixed-work cells ignore EOS and contain repetitive synthetic
prompts; their large throughput ratios do not isolate kernel speed or establish
equivalent improvements on useful answers.

Throughput also hides a long-prompt latency tradeoff: at 8,192 inputs and four
requests, Sharar's median TTFT is **10.32 / 10.31 s**, versus SGLang's
**8.75 / 8.75 s**. Sharar has lower median TTFT in the other five cells in both
windows. The full evidence retains TTFT, individual completion, output IDs and
verification counts for every measured request.

## Natural answers and quality

For the six natural cells meeting their automatic checks or manual rubric in
this review, DFlash2 answer completion is **4.7–13.3% shorter than SGLang** across
the two windows. All use concurrency one and normal stopping. These are complete
answer latencies; output lengths can differ even when the task succeeds.

| Task | Input tokens | Output limit | Actual output tokens Sharar / SGLang | Sharar seconds W1 / W2 | SGLang seconds W1 / W2 |
|---|---:|---:|---:|---:|---:|
| Chat | 28 | 96 | 70 / 79 | 3.668 / 3.677 | 4.179 / 4.244 |
| Coding | 67 | 128 | 35 / 35 | 0.678 / 0.678 | 0.748 / 0.748 |
| Tool call | 313 | 128 | 39 / 39 | 0.833 / 0.831 | 0.916 / 0.919 |
| Reasoning | 87 | 256 | 49 / 48 | 1.122 / 1.129 | 1.241 / 1.242 |
| Session, cold | 2,172 | 96 | 68 / 69 | 3.093 / 3.086 | 3.396 / 3.398 |
| Session, primed | 2,172 | 96 | 71 / 70 | 2.370 / 2.370 | 2.494 / 2.486 |

The frozen sampled-chat rubric fails all six DFlash2 candidate responses. It
demands that the answer mention running computation, while its prompt asks
about a generic cache miss. The responses instead explain a slower data lookup;
the MTP control also misses this requirement. This exposes a prompt/rubric
mismatch, not evidence of an engine-state error. Preserve the failed result;
reconcile the task and rubric prospectively and validate on new held-out outputs
before claiming that gate passes.

SGLang additionally fails all six strict JSON checks by adding Markdown fences.
Sharar's automatic coding, tool, JSON and reasoning checks all pass. Sampled-chat
and failed SGLang JSON timings remain in the full evidence but do not supply
quality-qualified speed claims. Manual rubrics here were reviewed by the coding
agent; independent human sign-off was not obtained.

## Promotion decision

The two windows complete **414 measured requests**, plus 138 warmup requests.
All **40 predeclared performance checks against repaired MTP pass**, evaluated
separately in each window. Coding, tool, strict JSON, reasoning and both session
variants each reduce answer completion by at least 10% in both windows while
meeting their checks/rubrics. Chat stays within the protected 5% regression
limit. The quality failure above still blocks the combined gate.

**DFlash2 is not promoted.** The candidate remains explicit; source defaults and
the deployed production image remain unchanged. Reconcile the sampled-chat
prompt/rubric before a new held-out quality evaluation and independent review.
Do not relabel this frozen run as passing. The SGLang short-batch reversal and
8K/B=4 TTFT deficit are additional competitive-speed work, distinct from the
performance gates against MTP that passed here.

All six comparison runs and the attribution job exit successfully within their
budgets. The comparison's minimum observed `MemAvailable` is **70.75 GiB**.
After testing, the original `b893cdc` production image is running with its
memory guard active, and health/model/canary verification passes. The exact
image digest, final UTC check and per-run budget samples are in the evidence.

## Short-batch attribution and next speed work

Nine additional measured HTTP batches, plus one warmup, reproduce both paths:
**71.79–213.82 aggregate tokens/s**. Fast batches spend about 4.45 s in 35
four-request rounds. Slow batches add **51–84 single-request rounds**, taking
about 10.1–13.9 s in rounds overall. Median host-observed round time is roughly
126–127 ms at B=4 and 111–113 ms during the B=1 tail. These include orchestration
and GPU waiting; they are not kernel timings. Instrumented runs are excluded
from the promotion comparison.

The direct diagnostic keeps the same ordered prompts, no cache hits, 256
outputs per request, and four requests present before the first decode. Only
the admission/prefill partition changes. Every repeated partition produces
identical output IDs within that partition:

| Prefill groups | Repetitions | Median batch completion (s) | Verify rounds for requests 0 / 1 / 2 / 3 |
|---|---:|---:|---|
| 4 | 3 | 13.217 | 35 / 35 / 35 / 111 |
| 1 + 3 | 2 | 4.792 | 35 / 35 / 35 / 34 |
| 2 + 2 | 2 | 14.236 | 35 / 35 / 35 / 119 |
| 3 + 1 | 2 | 10.414 | 85 / 35 / 35 / 35 |

Some differences occur at the **first output token**, emitted by target prefill
before the first speculative round. Thus admission/prefill partitioning changes
the target continuation and subsequent acceptance cost. The experiment does
not identify the responsible projection, recurrent computation or numerical
margin. It does rule out fixing this with a simple wait-for-all-four policy:
the fully batched control consistently retains a long tail.

The first speed task is a fixed-token target-prefill trace across these four
partitions: compare initial logits and margins, locate the first divergent
layer/state, and inspect the associated draft features. Distinguish an
implementation error from expected quantized numerical sensitivity before
changing serving. Do not force the fastest synthetic continuation or add a
batching-delay/width knob.

Then screen a bounded way to produce more useful tokens per target weight
traversal on the remaining request. One candidate is spending otherwise unused
verification rows on alternative draft candidates at low active concurrency;
measure candidate coverage and incremental target/state cost in a standalone
benchmark before extending contracts. This is a proposed experiment, not an
implemented speedup. Keep it only with state/sampling correctness and measured
natural-workload gains. Long-session checkpoint endpoint retention and the
8K/B=4 prefill-latency deficit remain separate measured opportunities.

## Reproduction

Export `sharar/proposers/mtp.py` from `56605f0` as the frozen control, then run
in the exclusive guarded runtime:

```bash
python3 -m bench.mtp_feedback --reference /path/to/legacy_mtp.py --output /path/to/new-proof
python3 -m bench.competitive --inputs /path/to/inputs.json --url http://127.0.0.1:30000 \
  --profile sharar-dflash2 --output /path/to/new-run
```

Full sources, build logs, commands, runtime hashes and measurements live under
`$SHARAR_RESULTS/reviews/2026-09-06/promotion/`. The archived older evidence is
in its `archive/` directory. Exact host launch/restoration commands are retained
there rather than embedded in committed JSON. The compact
[evidence manifest](promotion-results.json) records the full report's location,
size and SHA-256; the full report hashes the underlying artifacts.
