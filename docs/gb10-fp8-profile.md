**GB10 FP8 profiling and next experiments — 2026-09-09**

The optimized Sharar profile reproduces **44.15 / 34.16 / 56.31 decode TPS**
for code / design / reasoning at 10K input and 1K output. Target verification
dominates the roughly 107 ms round. The small trained selector failed its
held-out screen, and expanded calibration still rejects half the sampled
FP8-to-FP4 projections. These results do not demonstrate a new serving
speedup or reach 100 TPS.

The launcher, benchmark tools, README priorities, roadmap and stale depth
documentation are updated. The experiments use the existing FP8 draft;
target arithmetic, weights and serving policy are unchanged. The complete
investigation and architecture context are in the
[100 TPS review](dgx-spark-100tps-review.md).

**What was run.** One NVIDIA GB10, maximum batch four but one active request;
the same frozen 10,000 input IDs per task and exactly 1,000 output IDs;
greedy decoding, seed 42, EOS ignored, cold prefix cache. Sharar uses DFlash2
with an FP8 draft, depth 15, 16-row target verification, FP8 target KV, FP32
recurrent state, accepted-state reconstruction, 60,000 KV pages, 24 state
slots and 2,048-token prefill chunks. Loading, compilation and full-generation
warmups are excluded from throughput.

The target is `RadixArk/Qwen3.8-27B-NVFP4` at
`319f741cce68d7914884900c138a1fbb70a42f30`; the draft is
`incoai/Qwen3.8-27B-DFlash2` at
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`. Engine source matches Sharar
`b8dc773`; the evidence also hashes individual source files. The GPU windows
use pinned image digests with current source mounted read-only. The original
service and memory guard are restored after each window.

Native measurements use one excluded warmup and two timed full generations
per task. A separate continuation collects four CUDA-event rounds followed
by four CUPTI rounds near output positions 32, 384 and 768. Actual positions
are retained because a natural round can cross the requested position.
Another full continuation captures candidate IDs, conditional scores and
proposals, downloading the extra diagnostic tensors after generation.
Instrumented continuations reproduce every native output ID on all tasks.
No profiler or candidate-capture timings are used as serving TPS.

**Fresh serving measurements.** AIPerf 0.12.0 uses one excluded full warmup
and three timed requests per task and engine. Both servers use the same
loopback/Unix relay, frozen IDs and cold-cache protocol in separate GPU
windows. The client audits raw streamed prompt/output IDs, usage, successful
termination and zero cached input tokens. End-to-end TPS is `1000 / latency`;
decode TPS is `999 / (latency - TTFT)`.

| Task | Sharar decode TPS | Sharar end-to-end TPS | TTFT, s | Total latency, s | Emitted tokens/round |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 44.15 | 37.61 | 3.961 | 26.588 | 4.735 |
| Design | 34.16 | 30.11 | 3.963 | 33.210 | 3.673 |
| Reasoning | 56.31 | 46.06 | 3.969 | 21.710 | 6.055 |

Sharar outputs are identical across all four requests per task, the native
profile run and the preceding FP8 process. The new measurements reproduce
the [previous baseline](gb10-tps.md), rather than establish another optimization.

The pinned SGLang reference retains its BF16 DFlash2 draft and its existing
compiled 16-row profile. Its fresh results are:

| Task | SGLang decode TPS | SGLang end-to-end TPS | TTFT, s | Rounds | Sharar end-to-end gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 25.08 | 22.70 | 4.214 | 304 | +65.67% |
| Design | 33.45 | 29.35 | 4.212 | 229 | +2.61% |
| Reasoning | 44.73 | 37.66 | 4.224 | 171 | +22.32% |

SGLang's effective round costs are 130.4–131.0 ms. Each task repeats exactly
within this process, but all three continuations differ from the preceding
SGLang process and from Sharar. Cross-engine first differing output IDs are
at offsets 103 / 41 / 208. The draft precisions also differ. These are measured
deployment-profile comparisons, not an identical-work engine-only speedup.

Using the fastest task mean across this and the three preceding pinned
SGLang loads gives a more conservative end-to-end comparison:

| Task | Fastest SGLang task mean | Current Sharar gain |
| --- | ---: | ---: |
| Code | 31.19 TPS | +20.58% |
| Design | 29.35 TPS | +2.61% |
| Reasoning | 40.62 TPS | +13.39% |

All 24 fresh AIPerf requests pass the raw-stream audit. Three timed samples
per task establish this repeated baseline, not latency-tail estimates or
broad task-quality coverage.

**Where the round goes.** Native full-generation averages are 106.82 ms
(code), 107.50 ms (design) and 107.53 ms (reasoning). There are 211, 272 and
165 rounds respectively after the first prefill-produced token. The task
speed difference principally follows acceptance, while round costs are similar.

| Component | CUDA-event duration across nine sampled windows |
| --- | ---: |
| Draft proposal | 12.27–12.32 ms |
| Target verification | 89.90–90.32 ms |
| Accepted recurrent-state reconstruction | 1.40–1.51 ms |
| Commit including reconstruction | 1.63–1.81 ms |
| Accepted-feature/draft-KV feedback | 1.77–1.80 ms |

These are ranges of four-round medians. Nested phases overlap: reconstruction
is inside commit, and commit/feedback are inside acceptance. The roughly
104 ms CPU span around acceptance mostly waits for previously queued GPU
work; it is not 104 ms of removable Python work.

Kernel ownership is attributed through CUPTI launch correlations and the
innermost CPU phase scope, including CUDA driver launches. A graph's delayed
GPU timestamps cannot be assigned by CPU entry/exit timestamps alone.
Across 36 traced rounds, mean kernel-duration sums per round are:

| Kernel group | Time, ms |
| --- | ---: |
| Target NVFP4 GEMMs | 48.01 |
| Target other GEMMs, predominantly FP8 | 36.53 |
| Target attention | 2.26 |
| Target remaining kernels | 4.41 |
| Draft other GEMMs | 8.55 |
| Draft NVFP4 vocabulary head | 3.16 |

The other-GEMM category includes small BF16 projections. GPU activity covers
95.4–97.5% of instrumented decode spans; GEMMs account for 90.3–91.8% of
summed kernel durations. These are diagnostics, not physical bandwidth or
occupancy measurements. Nsight Compute is installed, but the driver has
`RmProfilingAdminOnly=1`; these non-root benchmark windows did not collect
hardware counters. Physical DRAM, L2 and occupancy attribution remains open.

The separate code-prefill trace differs: matrix operations occupy about
54.7% of 3.871 s summed kernel time. The NVFP4 packing kernel alone sums to
268 ms (6.9%) over the full 10K prefill. Prefill fusion and GDN/attention work
therefore deserve their own cost model; the decode GEMM share cannot be
applied to cold TTFT.

**Full-generation candidate support and the selector screen.** A support
oracle asks how long the actual future remains in each position's existing
top-16 candidates. It is evaluated at native round starts and includes the
target correction/bonus token. End-of-output windows without a complete
16-token horizon are excluded from this diagnostic, accounting for its
slightly different native means from the serving table.

| Task | Uncensored rounds | Native emitted mean | Oracle support mean | Missing target at first rejection |
| --- | ---: | ---: | ---: | ---: |
| Code | 210 | 4.743 | 7.852 | 61 / 210 |
| Design | 268 | 3.683 | 7.295 | 46 / 268 |
| Reasoning | 160 | 6.163 | 9.656 | 27 / 146 |

Most first rejections are selection errors within available candidate support,
but the support itself also limits long prefixes. Reasoning varies strongly:
its early/middle/late native means are 6.80 / 10.58 / 4.03. A favorable middle
window is not representative of the whole answer. The oracle uses future
tokens and fixed native histories; a changed policy visits different histories
and round starts. Multiplying these means by current round frequency does
not establish achievable generation speed.

The prospective feasibility test trains an eight-feature residual ranker on
code/design only: score gaps, rank, position/previous-rank interactions and
two quadratic terms. It uses 3,126 training rows, 200 full-batch Adam steps,
learning rate 0.03, L2 0.01 and position discount 0.9. Reasoning is evaluated
after the fixed training run. Zero weights reproduce every uncensored native
lattice walk; learned inference features contain no target-future tokens.

| Evaluation | Change in fixed-history emitted-prefix coverage |
| --- | ---: |
| Code, training prompt | -0.90% |
| Design, training prompt | -0.41% |
| Reasoning, held-out prompt | -1.62% |

The frozen gate was at least +10% on held-out coverage before implementing a
native policy experiment. It failed, so there is no selector serving change.
This rejects this cheap score/rank model. It does not reject hidden-state
selector training or draft-backbone distillation. One held-out prompt also
cannot establish generalization across agent tasks.

**Expanded target FP4 screen.** Ten existing FP8 projections at layers 0, 3,
32 and 63 are screened. Code/design calibration captures 16 spread rows per
prefill chunk and complete 16-row verify blocks near positions 32/384/768:
256 calibration rows per projection. Held-out reasoning supplies three
16-row blocks, 48 validation rows. Four candidates combine no smoothing or
alpha 0.5 with weight-scale factors 1 or 2. Selection uses calibration error;
every held-out block must stay below 5% normalized RMS output error.

| Projection | Held-out NRMSE | Worst block NRMSE | Screen | FP8 / FP4 median, ms |
| --- | ---: | ---: | --- | ---: |
| Layer 0, GDN QKV | 3.25% | 3.48% | Pass | 0.293 / 0.191 |
| Layer 0, GDN Z | 4.16% | 4.85% | Pass | 0.199 / 0.137 |
| Layer 0, GDN output | 1.64% | 1.65% | Pass | 0.199 / 0.140 |
| Layer 3, attention QKV | 2.29% | 2.35% | Pass | 0.443 / 0.247 |
| Layer 3, attention output | 2.30% | 2.39% | Pass | 0.200 / 0.138 |
| Layer 32, GDN QKV | 8.21% | 8.68% | Fail | 0.298 / 0.195 |
| Layer 32, GDN Z | 5.75% | 5.99% | Fail | 0.198 / 0.137 |
| Layer 32, GDN output | 12.00% | 12.52% | Fail | 0.197 / 0.140 |
| Layer 63, attention QKV | 5.84% | 6.12% | Fail | 0.434 / 0.245 |
| Layer 63, attention output | 9.31% | 9.97% | Fail | 0.195 / 0.137 |

Timings use the real 16-row shape, seven samples with a 128 MiB cache-flush
buffer, and include activation packing plus smoothing for FP4.
The reconstructed FP8 input reproduces its control outputs exactly. The five
passing projections show 1.43–1.79x local speedups, but their isolated savings
sum to only about 0.48 ms. That sum is not a measured whole-round improvement
and cannot be extrapolated over untested layers.

The weights are reconstructed from the current quantized checkpoint; this
is not fresh quantization from original BF16 weights. No candidate is installed
in the model, and no whole-model quality or compressed-target continuation is
claimed. Calibration improves the evidence but does not justify uniform FP4.

**What to do next.** At 107 ms/round, 100 decode TPS requires about **10.7
emitted tokens per round**. At approximately 3.96 s TTFT, 100 end-to-end TPS
requires **165 decode TPS**, or **17.7 tokens per current round**. Current
depth permits at most 16. A useful intermediate budget is 8 emitted tokens
in 80 ms: 100 decode TPS, approximately 72 cold end-to-end TPS. These are
arithmetic targets, not forecasts from the measured oracle.

1. **Make target-specific draft quality the main speed experiment.** Collect
   independent code/design/reasoning and tool-use histories, split by repository
   or task before training, and reserve a final test set. Capture draft/target
   hidden features as well as the candidate lattice. Train for expected accepted
   prefix length; compare a richer selector with backbone distillation, since
   captured support means are below the 10.7-token budget on all three tasks.
   One concrete variable: the pinned draft config declares `block_size=8`,
   while these runs use 16 verify rows. Test supervision matched to deployment
   width; this metadata difference alone does not establish a defect.
   Advance only after held-out native completions improve at least 10%, including
   selector cost, and task-quality/regression gates pass. The current ranker
   provides a negative control and reusable capture/evaluation machinery.
2. **Profile the expensive target matrices before another kernel sweep.** Use
   the 16-row target shapes and actual matrices identified above. Collect memory
   and stall counters to distinguish transfer, tensor-core utilization and
   instruction overhead. Protect rounding with common-history replay, then
   measure complete generations. Tiny reconstruction or top-k changes have
   insufficient measured budget for a twofold speed increase.
3. **Develop mixed target precision through sensitivity and quality checks.**
   Extend activation coverage across layers, held-out prompts and longer
   contexts. Retain FP8 where sensitive. Evaluate isolated and combined passing
   replacements on logits, committed state, perplexity/task quality and long
   continuations before serving comparisons. If original-precision weights
   become available, calibrate from them. Compare the resulting artifact in
   both engines where supported.
4. **Measure the agent loop separately from cold 10K/1K.** Keep this cold
   benchmark, and add repeated turns, common system/tool prefixes, branches,
   tool-result appends and session pressure. Report cache-hit TTFT, whole-turn
   latency and per-request decode speed. Prefix reuse can remove repeated
   prefill work; it does not raise sustained decode TPS. Revisit prefill packing
   and GDN/attention fusion for the cold-request 100 TPS goal.

**Validation and evidence.** The CPU suite passes 397 tests with 8 skipped.
The four Sharar quality canaries pass; they are a small smoke check, not broad
quality validation. All engine source hashes match the measured native run.
The launcher passes FP8 preflight and accepts
`SHARAR_DRAFT_PRECISION=fp8 ./run-sharar.sh` when using an image built from
current source. Its BF16 default and the original running image are retained.
The final health check returns HTTP 200, the original container/image identity
matches, the memory guard is active and the temporary GPU container is gone.
Minimum host `MemAvailable` was 79.86 GiB during native profiling, 80.04 GiB
during compression, 79.14 GiB during Sharar AIPerf and 61.78 GiB during SGLang
AIPerf, above the 16 GiB guard floor throughout.

Raw traces, candidate captures, calibration/timing grids, AIPerf streams,
commands, pins, resource samples and executable analysis scripts live under
`$SHARAR_RESULTS/reviews/2026-09-09-fp8-profile/` (this host:
`/home/yke/sharar-results/reviews/2026-09-09-fp8-profile/`).
The compact [evidence manifest](gb10-fp8-profile-results.json) identifies
`results.json` and its checksum. Reproduction commands are in the
[benchmark guide](../bench/README.md#detailed-profiling-and-historical-studies).
