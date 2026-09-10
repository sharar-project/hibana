# GB10 optimization implementation and measurements

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

2026-09-08. Follow-up to [the 10k/1k audit](gb10-10k1k-audit.md), using
`bef1456` as the arithmetic control. Raw evidence is under
`${SHARAR_RESULTS}/reviews/2026-09-08-gb10-optimization/`.

The later [SGLang acceptance investigation](gb10-acceptance-parity.md) supersedes
the inference that the generation-round gap establishes a much weaker draft.
On 128 identical reference histories, original Sharar matches 481 draft prefix
tokens versus 489 for SGLang (609 versus 617 including correction tokens).
The engines' different generated continuations explain much of the apparent
gap. Reference norm/weight arithmetic trials do not yield a promoted speedup.

The implementation adds a fused BF16 draft convolution and an explicit FP8
draft precision. The paired sixteen-row HTTP experiment reduces code B1
completion time by **4.50%** and code concurrency-four batch time by **5.83%**.
Design improves **6.52%**; reasoning regresses **2.51%**. This is a modest,
workload-dependent improvement. The primary B1 result misses the prospectively
recorded 5% promotion threshold, so BF16 remains the default.

## Integrated changes

- `kernels/draft_conv.py` combines the two-tap draft convolution into one
  Triton kernel. It preserves the original BF16 multiply/add rounding points
  and fused `addcmul` behavior, including each request's block boundary.
  The first full code continuation is token-identical to the unfused control.
- `model/draft_linear.py` packs the five draft layers' q/o and gate/up/down
  matrices into E4M3 with explicit weight scales. These 20 matrices execute
  through the installed GB10 `torch._scaled_mm` backend. The original BF16
  matrices are released after conversion.
- `kernels/draft_fp8.py` computes the dynamic activation maximum and packs
  E4M3 in two kernels. It avoids full-size FP32 activation temporaries and
  matches the eager prototype's packed bytes and scales, including graph
  replay with changing activation ranges.
- `SpecConfig`, proposer construction, `Executor`, CLI and server metadata
  carry `draft_precision`. Unsupported configurations fail before model
  loading. The memory planner retains its conservative BF16 load reserve.

The FP8 option leaves the target checkpoint, target execution, feature
projection, draft context-KV feedback and selector weights in their existing
formats. The draft model allocation falls from **3.600 to 2.160 GiB**;
post-warmup CUDA allocation falls from **29.589 to 28.148 GiB** in the matched
configuration. These are allocation measurements, not DRAM traffic counters.

For a source build:

```bash
sharar serve --proposer dflash2 --spec 15 --verify-state reconstruct --draft-precision fp8
```

FP8 is an explicit numerical profile. Its long generated continuations differ
from BF16 even with greedy sampling. The tests establish kernel behavior,
ownership and acceptance mechanics; the four canaries provide limited answer
checks, not broad model-quality certification.

## Paired HTTP results

Each request has the same frozen **10,000 input IDs and exactly 1,000 output
tokens**, greedy sampling and ignored EOS. Caches are flushed before every
sample. B1 code is primary; B4 code measures the completion of four simultaneous
client requests. Design and reasoning are separately frozen synthetic held-out
task families. Startup, compilation and graph capture are excluded.

The load order is control / FP8 / FP8 / control, with one measurement of each
cell per load. Both engines use the same Sharar HTTP scheduler, capacity four,
60,000 KV pages, 24 state slots, 2,048-token prefill chunks and reconstruction
of committed GDN state. There is no full-generation warmup in this paired run;
engine warmup runs before timing. Two samples do not establish tail latency or
statistical significance. B4 scheduling changes output trajectories between
repeats; full request metadata and output hashes are retained.

| Workload | Control seconds, both samples | FP8 + fusion seconds, both samples | Mean time reduction | Throughput ratio |
|---|---|---|---:|---:|
| Code, B1 | 35.144, 35.566 | 33.814, 33.712 | 4.50% | 1.047× |
| Code, four clients | 59.525, 57.872 | 55.110, 55.438 | 5.83% | 1.062× |
| Design, B1 | 34.623, 34.915 | 32.621, 32.383 | 6.52% | 1.070× |
| Reasoning, B1 | 24.127, 24.218 | 24.749, 24.806 | −2.51% | 0.976× |

B1 first-token latency stays around 3.75–3.80 seconds. Code verification
rounds rise from 264 to 266 with FP8; design changes from 258 to 257. Reasoning
rises from 170 to 187: cheaper rounds lose their advantage when acceptance
drops. Each B1 profile repeats its own output hash exactly in both loads;
BF16 and FP8 hashes differ for all three tasks. All four fixed canaries pass
in all four loads (16/16 checks).

## Native ablations and failed hypotheses

These are separate, single-sample native-engine measurements with serial
prefill. They are useful ablations and must not be substituted for the paired
HTTP results above.

| Code B1 profile | Complete time | Verify rounds | Interpretation |
|---|---:|---:|---|
| Original BF16 arithmetic | 35.106 s | 264 | Control |
| Fused convolution | 34.598 s | 264 | 1.45% less time; identical output IDs |
| Fusion + FP32 selector scores | 36.018 s | 277 | Regresses; selector default unchanged |
| Fusion + draft FP8 | 33.469 s | 266 | 4.66% less time in this sample |
| Fusion + FP8 + FP32 selector | 33.018 s | 266 | Same code output IDs as FP8 alone; follow-up below |

The follow-up measures two HTTP samples per cell in each independently loaded
profile. These are sequential profile screens, rather than another ABBA run.
All twelve canary checks pass (four per profile).

| Profile | Code B1 mean | Code four-client mean | Design B1 mean | Reasoning B1 mean |
|---|---:|---:|---:|---:|
| Original BF16, eight verify rows | 34.518 s | 54.912 s | 33.057 s | 27.675 s |
| FP8 + fusion, eight verify rows | 34.041 s | 53.612 s | 28.886 s | 24.901 s |
| FP8 + fusion + FP32 selector, sixteen rows | 33.661 s | 56.558 s | 31.911 s | 26.088 s |

Compared with the original sixteen-row paired control, FP8 at eight rows cuts
four-client code time **8.67%** and design time **16.92%**, but cuts code B1 only
**3.72%** and increases reasoning time **3.01%**. It is a useful explicit width
choice (`--spec 7`), not a universal replacement for sixteen rows. BF16 at eight
rows increases reasoning time **14.49%**, showing why the depth control must
include multiple tasks.

Adding FP32 selector scores to the sixteen-row FP8 profile changes no output
IDs in the complete code continuation. Its code timing difference is about
0.3%, and reasoning regresses **7.93%** against the original control. The
selector arithmetic remains unchanged in serving.

The installed Torch backend also supports per-output-channel FP8 weight
scales. An additional native screen (`fp8_rowwise`, keeping the same global
activation scale) takes 33.174 / 32.082 / 24.554 seconds on code / design /
reasoning. The code difference from per-tensor FP8 is under 1% in single
samples. It remains a benchmark-only candidate without paired HTTP or quality
validation; these numbers are not an additional achieved serving speedup.

A final rounding ablation computes draft RMSNorm's weight multiplication and
the complete two-tap convolution in FP32 before the final BF16 cast. This
tests one difference from the reference's fused operations; it does not claim
complete SGLang parity and leaves RoPE and GEMM partitioning unchanged. The
BF16-weight case takes **32.622 seconds / 246 rounds**, versus 35.106 / 264
for its native control. The FP8-weight combination regresses to **34.380
seconds / 272 rounds**, versus 33.469 / 266. Both miss this experiment's
predeclared requirement of 10% fewer rounds before advancing to held-out
validation. These profiles remain benchmark-only; the serving convolution
continues to preserve the original BF16 rounding points.

A fixed-history diagnosis records the first 96 native rounds of each task,
then runs the alternative selector on those same hidden states. Its code
output remains identical to the uninstrumented native control.

| Task | Actual first rejections observed | Correct token present in top 16 | Missing from top 16 | FP32 paths better / worse / unchanged |
|---|---:|---:|---:|---|
| Code | 96 | 73 | 23 | 1 / 1 / 94 |
| Design | 96 | 76 | 20 | 0 / 0 / 96 |
| Reasoning | 82 | 74 | 8 | 1 / 1 / 94 |

Most sampled failures are selection failures within available support. FP32
score addition alone barely changes those selections. Fixed-history prefix
agreement totals are 238→239, 291→291 and 595→587; these are diagnostic counts,
not measured acceptance under a counterfactual generated trajectory. The
result identifies a possible training direction within Sharar's own histories;
it does not establish a selector disadvantage against SGLang. The later
[same-history comparison](gb10-acceptance-parity.md) finds little aggregate
deficit and does not support prioritizing retraining on this evidence alone.

An eight-token history lookup screen finds only two matching rounds out of
264 on code. Those matches would emit five tokens instead of the native
seven. No lookup serving path is added from this negative coverage result.

## Target NVFP4 sensitivity and cost

`bench/gb10_target_fp4.py` reconstructs selected FP8 checkpoint matrices,
calibrates four scale/smoothing candidates on 64 actual prefill rows, and
validates the chosen candidate on a later 16-row verify block. The four
sampled layers cover early, middle and final GDN/attention projections.
The calibration grid and 5% normalized RMS error screen are recorded before
execution. It is a small layerwise sensitivity screen, not checkpoint-level
quantization calibration or an end-to-end speedup.

Both captured graphs are replayed before checking their outputs. Regenerating
the FP8 control inputs gives zero error in all ten sampled projections.
Timing includes the candidate's smoothing, BF16 conversion, NVFP4 packing and
GEMM. A 128 MiB flush precedes each of seven separately timed graph replays.

| Projection | Held-out normalized RMS error | Local FP8 / NVFP4 time ratio |
|---|---:|---:|
| Layer 0 GDN qkv | 3.52% | 1.49× |
| Layer 0 GDN z | 4.95% | 1.51× |
| Layer 0 GDN out | 1.63% | 1.43× |
| Layer 3 attention qkv | 2.86% | 1.71× |
| Layer 3 attention o | 2.49% | 1.44× |
| Layer 32 GDN qkv | 8.11% | 1.48× |
| Layer 32 GDN z | 5.67% | 1.41× |
| Layer 32 GDN out | 13.62% | 1.45× |
| Layer 63 attention qkv | 5.59% | 1.72× |
| Layer 63 attention o | 11.14% | 1.43× |

Five of ten projections fail the numerical screen. Target conversion remains
disabled. A viable target profile needs more representative calibration,
layerwise sensitivity selection and model-level quality validation. The
local 1.4–1.7× GEMM ratios are not whole-engine gains.

The first screen attempt read graph-capture output storage before replay,
making its validation errors invalid. It is retained with `INVALID.txt`;
the table uses `target-fp4-screen-v2`. No serving decision uses the invalid
attempt.

## Validation and evidence

- 344 CPU tests pass; seven GPU/snapshot-dependent unit cases skip.
- 54 GPU tests pass: exact FP8 pack/scale checks, BF16 convolution equivalence,
  and real-model lifecycle tests in both precisions across eager/graph paths,
  widths 1/3/7, batches 1–4, mixed greedy/sampled requests, cache reuse and
  long batched prefill.
- The convolution checks additionally exercise strided base coefficients.
- The guarded model windows retain substantially more than the 16 GiB host
  available-memory floor. The original server and memory guard are restored
  after each window. The production image is not rebuilt by these experiments.

Commands, input IDs, native output IDs, HTTP output hashes, source hashes,
image identities, resource samples and unsuccessful trials remain in the
dated evidence directory. `bench/gb10_optimize.py` reconstructs the original
selector from the frozen baseline source and selects arithmetic profiles in
isolated processes. `bench/gb10_acceptance.py` is deliberately separate from
performance timing.

## Refreshed SGLang comparison

The same pinned SGLang image and native sixteen-row DFlash2 configuration
complete the exact code B1 task in **31.013 and 31.257 seconds**, mean
**31.135 seconds**, with 205 verification rounds. One full-generation warmup
(31.084 seconds) is excluded. Model loading, compilation and graph capture
are also excluded. The frozen input-file hash matches the Sharar follow-up.
Target/draft revisions, launch command and complete runtime configuration are
retained in `http-sglang-refresh-k15/`.

Sharar's paired per-tensor FP8 mean is **33.763 seconds**, still **8.44% more
time** than this SGLang reference. The optimization closes part of the gap;
it does not establish a wide-margin win. This is a complete configuration
comparison with a BF16 SGLang draft versus Sharar's explicit FP8 draft.
The prior B4 delayed-admission caveat remains unresolved, so no saturated-B4
engine advantage is claimed from those historical reference timings.

The fresh SGLang window preserves at least **63.40 GiB** host available memory
and restores the original Sharar container and memory guard afterward.

[Machine-readable results and the evidence archive checksum](gb10-optimization-results.json).
