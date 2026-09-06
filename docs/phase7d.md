# Phase 7D: native drafts and GB10 state retention

Historical report: the width/latency policies and their commands below are
archived at commit `56605f0`. They were subsequently removed from the shipped
runtime; the draft, cache and accepted-state interfaces remain supported.

The subsequent [DFlash2 profiling and selection optimization](draft-selection.md)
records the next measured speed change; the measurements below remain the 7D baseline.

7D adds native DSpark/DFlash2 adapters, draft-aware prefix state, exact sampled
verification, prospective memory planning and experimental scheduling policies.
It also implements the accepted-state reconstruction experiment selected in
7C. These are independent contribution boundaries, not a second engine.

MTP/full snapshots remain the defaults. The external drafts and reconstruction
are selectable. No universal speed lead over SGLang or production promotion is
claimed. The evidence below separates implementation correctness, numerical
oracle checks, workload quality and speed.

## Necessary repairs

- **Load memory:** release each layer's CPU checkpoint mappings after copying
  its final weights to CUDA. On GB10 those mappings competed with CUDA weights
  in the same physical DRAM. The 20 ms host trace fell from **39.40 GiB** peak
  net `MemAvailable` drop to **21.36 GiB**, confirmed at **21.33 GiB** in the
  DSpark run. Load time was 94.47 versus 94.06 seconds in the isolated A/B.
  CUDA peaks remained 19.577 GiB allocated / 19.994 GiB reserved. The loader
  now reserves 24 GiB plus the independent 16 GiB floor: **40 GiB**, replacing
  the unmeasured 64 GiB gate. Pool/draft startup has its own prospective check.
  Host samples measure observed available-memory pressure, not physical traffic
  counters or a proof that every possible system workload has this peak.
- **MTP cache validity:** identity-keyed endpoint features distinguish usable
  draft KV from target-only prefix hits. Missing metadata uses ordinary decode.
  Full acceptance now writes the extra committed draft-KV input associated with
  the bonus output. Previously the next round could attend to unwritten memory.
  A poisoned-KV regression covers both failures; the control measurements include
  these repairs. The old faster, incorrect MTP path is not an optimization control.
- **Shutdown and failures:** accepted work drains on normal shutdown with a
  deadline; ASGI lifespan owns final engine cleanup before Uvicorn re-raises
  SIGTERM. The engine owner thread enters PyTorch inference mode explicitly.
  Invalid requests/queue exhaustion stay isolated. Unexpected partially mutating
  GPU/engine failures still fail the engine: rollback cannot undo arbitrary
  in-place state writes, so blindly continuing would risk corrupt output/cache.
- **Compatibility:** finite integral JSON numbers such as `max_tokens: 128.0`
  are accepted at the HTTP boundary. Internal integer contracts remain strict.
  `/generate` restores SGLang-compatible `spec_accept_length`, including the
  prefill token, and exposes `spec_tokens_per_round` for actual speculative output.
- **Arbitrary prefill chunks:** the interactive workload exposed an absolute
  checkpoint-alignment assumption. A chunk starting at token 1,910 attempted
  to snapshot 2,048, which is not a local 64-token recurrent boundary. Snapshot
  selection now uses boundaries relative to that forward (2,102/2,166 in the
  reproduced case). Single and batched unaligned-prefix regressions retain
  cache/draft restoration coverage. The failed run is preserved separately.
- **Image:** the pinned Docker tree was built. `pip check` exposed an upstream
  `nvidia-cusparselt-cu13==0.8.1` wheel tagged `manylinux2014_sbsa`. The build
  verifies its AArch64 ELF payload and repairs only that exact WHEEL tag and
  RECORD entry to `manylinux2014_aarch64`; library bytes and pins are unchanged.
  The standalone image passes the dependency check. No Transformers or SGLang
  dependency was added to the runtime.

## What contributors can extend

| Owner | Implemented behavior |
|---|---|
| `model/parallel_draft.py` | Five-layer checkpoint backbone, requested target features, RoPE/YaRN, DSpark Markov/confidence, DFlash2 local convolution and sparse selector |
| `proposers/parallel.py`, `parallel_graphs.py` | Private draft KV rings, bounded projected-feature LRU, exact conditional proposals, fixed CUDA graphs |
| `engine/rejection.py` | Shared greedy/point-mass and dense/sparse stochastic acceptance, transformed p/q, request-local RNG |
| `engine/spec.py`, `spec_graphs.py` | Target verification, B=1/2/4 and depth=1/3/7 buckets, stop-aware accepted state |
| `memory/request_state.py`, `radix.py` | Sole target page/slot ownership and identity-keyed endpoint metadata |
| `memory/spec_budget.py` | Header-derived weight sizes, draft KV/features, total verify bytes and explicit graph/activation reserves before optional cache allocation |
| `kernels/fla/accepted_state.py` | One all-layer accepted-prefix reconstruction launch, FP32 recurrence, existing commit boundary |
| `engine/width_policy.py`, `latency_policy.py` | Opt-in measured-cost width policy and bounded prefill feedback |

DSpark is pinned to `RadixArk/Qwen3.8-27B-DSpark` revision
`b9a5dbdf03bc999c6c73c426b19c2d9041cea393`; DFlash2 to
`incoai/Qwen3.8-27B-DFlash2` revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`. Both use target post-block features
5/19/33/47/61 read from their configs. DSpark retains the supported target
context; DFlash2 retains a 2,048-token draft window. Their prefixes carry bounded
feature-cache handles and rebuild draft K/V only. Missing handles safely lose
speculation for that request; target prefix reuse remains available.

The test-only publisher oracle uses Z Lab DFlash revision
`07ebd93db9f472af339b644bb70221ad8428328a` and Transformers 5.15.0 in an isolated
proof directory. Runtime graph/backbone tests agreed on every tested target-head
top-1 prediction. Relative hidden-state RMS error was 1.12% for DSpark and 1.44%
for DFlash2, with cosine 0.99994/0.99990. This is a pinned one-prompt backend
comparison, not a broad quality benchmark or bitwise draft parity claim.

See [interfaces.md](interfaces.md) for tensor lifetime and ownership contracts.
Sampled rejection remains a clear reference path with scalar synchronization;
optimizing it should preserve that oracle.

## Reconstruction experiment

At B=4, MTP k=3, full snapshots retain **2,463,105,024 bytes (2,349 MiB)**.
Reconstruction retains **664,043,520 bytes (633.28 MiB)**, including compact
recurrence inputs, final FP32 scratch and convolution windows: **73.04% less**.
Both leave committed state untouched during verification and truncate accepted
input count for EOS/output budget before publishing state. The new kernel is
bitwise equal to the snapshot recurrence at every tested accepted count.

The initial independent-process pilot completed 156 HTTP requests per strategy.
Most MTP serial completion medians improved by 0–4%; mixed cells varied between
about 6% faster and 2.4% slower. No 10% latency improvement is established.
The capacity result is useful, but it does not imply that the same percentage
of DRAM traffic or end-to-end time disappeared. Reconstruction techniques also
exist upstream; this implementation is not claimed as a new algorithm.

The first reconstruction profiling stage failed on obsolete benchmark metadata
that assumed the snapshot tensor existed. Its HTTP workload had completed;
the benchmark was repaired and the independent confirmation uses the built
image. Failed runs and source hashes remain in the proof directory.

The second independent HTTP comparison also completed 156 requests per
strategy with no request or automatic-check failures. MTP serial medians were
mostly 1–4% lower, while mixed MTP cells ranged from 4.7% lower to 4.3% higher.
One protected plain mixed-JSON cell was **5.06% higher**, just outside the frozen
5% threshold. Three repetitions do not resolve that uncertainty. The strict
promotion gate is therefore not cleanly passed; reconstruction stays opt-in.
The following interactive experiment failed at the unaligned checkpoint and
was repaired separately; it does not invalidate the completed ordinary workloads.

## External-draft workload pilot

Frozen `local_v1.json`, target revision
`319f741cce68d7914884900c138a1fbb70a42f30`, GB10, graphs, B≤4, 35,000 target
pages, 12 draft-test state slots. The repaired MTP control uses 16 slots. These
are **not equal-total-memory profiles**: external weights/features/rings add
capacity cost. Medians below use three measured repetitions, natural stopping,
and actual returned outputs; different continuations can change output work.

| Completion seconds | Repaired MTP k=3 | DSpark k=7 + reconstruction | DFlash2 k=7 + reconstruction |
|---|---:|---:|---:|
| Coding, cold | 1.165 | 0.719 | 0.722 |
| Tool call | 1.347 | 1.006 | 0.893 |
| JSON, cold | 1.055 | 0.602 | 0.613 |
| Reasoning | 2.628 | 1.112 | 1.192 |
| Chat | 3.743 | 4.225 | 3.840 |
| Session, cold | 3.829 | 3.928 | 3.361 |
| Session, reused | 2.856 | 3.044 | 2.730 |
| Mixed chat | 5.301 | 5.559 | 4.405 |
| Mixed session | 4.865 | 4.183 | 3.605 |

Coding/tool/JSON have the same output counts across these modes (35/39/30
tokens). The reasoning continuations differ: MTP emits 82 tokens, external
drafts 49, both reaching the expected answer. Its completion-time improvement
therefore includes shorter output and is not a same-token engine-speed claim.

Each external adapter completed 78 HTTP requests with no request failures and
no automatic coding/tool/JSON/reasoning failures. Its width/cache smoke recorded
59 valid draft-prefix restores and no invalid-state fallbacks. These are local
pilot comparisons, not a cross-engine claim, sustained soak or tail-latency study.

The repaired standalone image also passed **16 extended boundary cases per
adapter**: T=8,192/32,766, eager/graphs, greedy/sampled and cold/reused prefixes.
Both retained valid draft state through the two-token near-32K output budget.
Minimum sampled host availability was 67.75 GiB for the DSpark extended/policy
run and 74.18 GiB for DFlash2. These are lifecycle/numerical smoke gates, not
a long-context quality evaluation under cache pressure.

Assistant review examined all 26 rubric-scored responses per external adapter
(nine distinct texts each): 22 meet the frozen rubric. Four seeded sampled-chat
responses per adapter give coherent two-sentence data-retrieval analogies but
omit the rubric's explicit recomputation requirement. The plain/MTP controls
have the same rubric gap. Preserve the original records and gate; do not
quietly reword the frozen rubric to report a pass. A separately versioned
prompt/rubric and broader sampled evaluation are needed before promotion.

## Policy decisions

Width selection has measured costs and a reproducible calibration fitter.
The first DSpark short-context pilot measured B=1 rounds at 112.4/113.5/115.6 ms
for depth 1/3/7; B=4 at 146.6/147.3/154.6 ms. Draft work stays at maximum width,
so shorter verification saves relatively little in this profile. Current-round
confidence calibration uses only the accepted prefix and first rejection;
serving uses previous-round confidence to avoid a new mid-round host barrier.
That predictor requires independent validation. No guessed cost curve ships
as an enabled default.

The independent DSpark policy replay completed 78 HTTP requests with no
request or automatic-check failures and selected depth 3 for 250 request-rounds
and depth 7 for 625. Completion changes varied by workload (roughly 12% lower
to 5% higher); continuations can change with verify width. These are the same
prompts used for calibration, not a held-out quality/generalization gate. The
policy remains opt-in and is not a promoted speed profile.

The optional interactive policy uses bounded observations to cap total prefill
work while requests are decoding, avoids the second prefill forward in that
scheduler step, and retains a minimum chunk so queued work can progress. A
250 ms target is best effort, not a hard deadline. Measure visible SSE gaps and
quality; smaller chunks alone are not proof of a better local experience.
Width and prefill policies currently operate independently. Joint policy
selection and broad promotion remain explicit roadmap items.

The repaired **250 ms prefill profile is rejected for default use**. It
completed 78 requests without request or automatic-check failures, but mixed
chat completion grew from 5.23 to 8.18 seconds and the arriving session from
4.77 to 11.29 seconds. Session TTFT grew from 1.31 to 8.28 seconds. Worst observed
delivery gaps for the already-running coding/chat/JSON requests fell from about
1.18 to 0.28 seconds. This is a real tradeoff, not a successful general speedup,
and three repetitions do not establish population tail latency. Keep the flag
experimental. The next scheduling experiment should measure shared target
projection/weight work across prefill and verification, rather than assuming
ever-smaller separate prefills improve local inference.

Repaired reconstruction profiling completed all eight B=1/4, T=128/8,192 cells.
Every fixed-work emitted-token sequence matched the snapshot control. MTP
rounds changed from 112.66/139.45 ms to 110.04/131.89 ms at T=128, and from
116.58/148.46 ms to 113.93/140.58 ms at T=8,192 (B=1/4 respectively). Plain
rounds stayed within 0.5%. These isolated round gains do not override the
protected HTTP promotion miss above.

## Reproduce and evidence

The standalone GB10 image test run passed **401 tests**, including both state
retention strategies, 180 real-checkpoint lifecycle cases and 48 reconstruction
kernel cases. The latter cover B=1/2/3/4, steps=1/2/4/5/8/16, eager/graphs and
every accepted prefix count. The lifecycle reconstruction fixture records the
original kernel's full snapshots as test-only reference storage and checks
committed state bitwise. Minimum sampled host availability in that run was
50.18 GiB. After the checkpoint repair, **10 targeted GPU cases passed**
(single/batched unaligned checkpoints, cache restoration and 32K boundaries,
both state strategies). The final CPU suite passed **171 tests**, with six GPU-only skips,
including an actual 128-token HTTP stream drained across SIGTERM.

The full-suite image is
`sha256:f2b24fe20c666c3ee321bebc31a1451961b661f798524f044498087d3f4e4683`.
The checkpoint-repair image is
`sha256:bff092535eccb82cb1f41354fbda8b3291eca5b15f8ee2c6fd9053ea5fc6db09`.
The source differences and separate test runs are retained in the result JSON.
The core Python runtime is about 5.9K lines excluding kernels/parity tools; the shared external draft
backbone, adapter and graph driver total 444 lines.

With the pinned snapshots in the offline cache, select an experiment explicitly:

```bash
python3 -m sharar serve --spec 7 --proposer dflash2 --verify-state reconstruct
python3 -m sharar serve --spec 7 --proposer dspark --verify-state reconstruct
python3 -m sharar serve --spec 3 --verify-state reconstruct --interactive-target-ms 250
```

The host launcher accepts `SHARAR_PROPOSER`, `SHARAR_SPEC` and
`SHARAR_VERIFY_STATE`. Runtime startup plans draft costs before optional target
cache sizing. It does not automatically download checkpoints.

Maintain CPU tests with `bash tools/check.sh setup-cpu`; exclusive GB10 tests
with `bash tools/check.sh gpu`. `bench/draft_validate.py` runs the independent
oracle, width/sampling/cache smoke, optional HTTP workload and extended 8K/32K
boundaries. Its oracle packages live outside the runtime. `bench/phase7d.py`
runs independently loaded strategy controls and records allocator/host traces.

Raw evidence is under
`/home/yke/sharar-results/reviews/2026-09-05/phase7d/`: source hashes, command
arrays, image build logs, JUnit, 20 ms memory traces, raw SSE/tokens, profiler
traces and failed-attempt records. The running historical container is restored
after exclusive GPU jobs; building/testing a candidate does not deploy it.
The original `sharar-qwen38` container and `sharar-memory-guard.service` were
restored; health, model identity and canary checks passed after the final job.
The compact, portable measurements and artifact hashes are in
[phase7d-results.json](phase7d-results.json).

## Next speed work

1. DFlash2 is the strongest broad candidate in this pilot. Run equal-memory
   long-session/eviction and fresh SGLang comparisons, plus the unresolved
   sampled quality gate, before changing the default.
2. Profile external draft projection/selector and accepted feature-to-KV
   ingestion against target verification. Fuse whichever repeated work actually
   dominates on GB10, with the current implementation as its numerical oracle.
3. Prototype accepted-state reconstruction directly into validated committed
   slots, removing final scratch/copy while preserving StateManager ownership.
   Keep it only for measured capacity or client-visible gains.
4. Extend streaming load into a versioned prepacked immutable weight artifact,
   then test mapped/registered allocation paths and CPU contention. Separately
   test reclaimable optional cache slabs with graph-safe lifetimes. Neither
   shared physical memory nor returned cache IDs proves zero-copy execution or
   physical memory reclamation.
