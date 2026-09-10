**Sharar on DGX Spark: what reaching 100 tokens/s requires**

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

Follow-up: the [FP8 profiling and experiment report](gb10-fp8-profile.md)
executes the profiling, full candidate capture, selector feasibility and
expanded target-compression screens proposed below. Historical measurements
in this review remain dated; use the follow-up for current results.

Review date: 2026-09-09. Source inspected: Sharar `b8dc773` and the local
SparkLang/SGLang checkout `6b36f9d0a2`. The primary interpretation is one active
agent using the existing Qwen3.8-27B target, with cold 10,000-token inputs and
1,000 output tokens. Decode and end-to-end targets are evaluated separately.
This review reads source and existing evidence; it adds no GPU measurements
and does not change the running service.

The evidence supports investing in better proposals and lower target matrix
cost. It does not establish a route to 100 TPS through framework overhead
removal alone. A useful experimental milestone is 8 emitted tokens per
80 ms round: 100 decode TPS, but only about 72 end-to-end TPS with a 3.96 s
TTFT. Neither component of that milestone has been demonstrated together.

The latest [AIPerf results](gb10-tps.md) use one excluded full warmup and five
timed requests per task, temperature zero, identical frozen input IDs, exactly
1,000 output IDs, and a cache flush before every request. These are three
distinct prompts, not fifteen independent tasks. The code prompt asks for an
allocator implementation; design and reasoning use repeated reference notes.
They are useful fixed-work regression fixtures, with thinking disabled and
EOS ignored, rather than a complete agent-quality benchmark.

| Task | Sharar FP8 draft decode TPS | Sharar end-to-end TPS | Latest SGLang decode TPS | Latest SGLang end-to-end TPS | Sharar emitted tokens/round | Sharar effective round time |
|---|---:|---:|---:|---:|---:|---:|
| Code | 44.20 | 37.65 | 27.62 | 24.77 | 4.735 | 107.12 ms |
| Design | 34.20 | 30.14 | 33.23 | 29.18 | 3.673 | 107.39 ms |
| Reasoning | 56.37 | 46.09 | 47.09 | 39.34 | 6.055 | 107.41 ms |

The last two columns are calculated from the recorded means:
`999 / verification_rounds` and `(latency - TTFT) / verification_rounds`.
They describe useful emitted tokens, including correction/bonus tokens, and
effective serving round cost. They are not isolated GPU-event measurements.
All three Sharar tasks have approximately 3.96–3.97 s TTFT.

The latest SGLang load takes approximately 130.1 ms per round. Its design
continuation emits more tokens per round than Sharar's, which largely cancels
Sharar's cheaper rounds. Its reasoning continuation has similar acceptance,
so the execution-cost advantage is more visible. Cross-engine continuations
differ, and SGLang's continuations also differ between fresh processes.
Against the fastest recorded SGLang task mean across three recent loads,
Sharar's end-to-end throughput gains are 20.71%, 3.30%, and 13.46%.
These remain complete-configuration results with different draft precision
and output trajectories, not a controlled attribution of engine-only speedup.

I verified the linked `reviews/2026-09-09-tps/results.json` evidence against
its manifest: 40,931 bytes, SHA-256
`108af80b5363d1a3b9705a6b6af02ba5316f2b48e6561fb2a4e31ae2fdf973af`.
The reported throughput arithmetic also checks against the recorded means.
This is an integrity/arithmetic check, not a new audit of every raw stream or
a rerun of the historical correctness suites.

**The execution path is already specialized.** Sharar owns its serving,
scheduling, model execution, speculative verification, and memory management.
It does not import SGLang. SparkLang is a separate SGLang-based reference;
its adapter translates a GB10 profile into SGLang CLI arguments.

| Source | Responsibility and relevant constraint |
|---|---|
| [`api/server.py`](../../../sharar/api/server.py) | FastAPI/OpenAI and native API handling; one `EngineLoop` thread owns scheduler/GPU work and publishes progress to HTTP handlers. |
| [`engine/scheduler.py`](../../../sharar/engine/scheduler.py) | Continuous admission, bounded prefill, decode, cancellation, release, and retraction. It can execute two prefill forwards before decode. |
| [`engine/executor.py`](../../../sharar/engine/executor.py) | Pools, attention plans, prefill assembly, graph setup, and proposer installation. |
| [`model/qwen38.py`](../../../sharar/model/qwen38.py), [`model/gdn.py`](../../../sharar/model/gdn.py) | The 64-layer, hidden-size-5120 dense hybrid target: 48 Gated DeltaNet layers and 16 full-attention layers. No MoE path. |
| [`weights/modelopt.py`](../../../sharar/weights/modelopt.py), [`kernels/ops.py`](../../../sharar/kernels/ops.py) | Mixed FP4/FP8 weight loading and numerical boundaries; explicit FlashInfer/CUTLASS dispatch. Small ordinary-decode FP8 matrices use custom GEMV; 16-row verification uses scaled GEMM. |
| [`model/parallel_draft.py`](../../../sharar/model/parallel_draft.py), [`proposers/parallel.py`](../../../sharar/proposers/parallel.py) | Five-layer DFlash2 backbone, sparse conditional candidate selector, private draft KV rings, feature feedback, and cache restoration. |
| [`engine/spec.py`](../../../sharar/engine/spec.py), [`engine/spec_graphs.py`](../../../sharar/engine/spec_graphs.py) | Draft, target verification, acceptance, stop/budget truncation, state commit, and feedback. Target verification has CUDA graph batch buckets. |
| [`memory/request_state.py`](../../../sharar/memory/request_state.py), [`memory/radix.py`](../../../sharar/memory/radix.py) | Target pages, private recurrent slots, reservations, processed-prefix publication, cache loans, and rollback ownership. |
| [`kernels/fla/accepted_state.py`](../../../sharar/kernels/fla/accepted_state.py) | Reconstructs the accepted recurrent prefix directly into validated private FP32 slots, avoiding full speculative state snapshots. |
| [`memory/cold_tier.py`](../../../sharar/memory/cold_tier.py) | Managed-memory slabs park inactive sessions; restored state is copied back into the hot pools. Restore still includes a stream synchronization before arena reuse. |
| [`benchmark.py`](../../../sharar/benchmark.py), [`benchmark_aiperf.py`](../../../sharar/benchmark_aiperf.py), [`benchmark_workloads.py`](../../../sharar/benchmark_workloads.py) | Frozen workload construction, AIPerf coordination, exact-token auditing, and comparison. |

In a DFlash2 round, the pending target token anchors a 16-row draft block.
The draft discards its anchor prediction and proposes 15 tokens. Target
verification consumes the pending token plus those proposals. Greedy prefix
acceptance downloads one packed result; stop/budget truncation occurs before
state commit. The final emitted token remains pending for the next round.
Committed target state describes `prompt_ids + output_ids[:-1]`.
Optimizations must preserve that invariant through cancellation, prefix reuse,
rejection, and output-budget boundaries.

The draft consumes post-block target features at layers 5, 19, 33, 47, and 61,
projects them into its own context, and keeps a 2,048-token draft window.
Its optional FP8 mode compresses q/o and MLP compute matrices; feature
projection, context-KV feedback, and selector weights retain their existing
precision. Greedy lattice traversal already runs in a small GPU kernel.
The sampled path already avoids per-position host scalar downloads and uses
vectorized rejection sampling per request, though it has additional proposal
and probability work beyond the greedy benchmark.

The target MLP and vocabulary matrices are NVFP4; large attention/GDN
projections remain FP8. Target KV is FP8 and recurrent state is FP32.
Derived directly from the shapes, one complete recurrent snapshot is 144 MiB
before approximately 2.81 MiB of convolution state. Full-attention KV costs
32 KiB per token, or 312.5 MiB at 10K context. This explains why recurrent
snapshot ownership and draft restoration matter so much for session capacity.
Neither allocation size is a direct measurement of memory traffic.

**The Spark-specific limit is moving and reusing the hot weights.** NVIDIA
specifies 128 GB coherent memory and 273 GB/s memory bandwidth; its advertised
1 PFLOP FP4 figure uses sparsity. The repo's device query records SM 12.1,
48 SMs, 24 MiB L2, and 100 KiB shared memory per SM.
[NVIDIA specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/),
[local device audit](gb10-10k1k-audit.md).

The historical target projection inventory is 17.555 GB, including 7.214 GB
of FP8 matrices. Reading that inventory once at the specified bandwidth takes
64.3 ms, before draft, activation, attention, recurrent, and launch costs.
This is a conditional ideal traffic floor, not proof of physical DRAM
saturation. The [earlier counters](nvfp4-cost.md) found roughly one weight
footprint in system-memory fills for sampled NVFP4 GEMMs. The
[later target trace](gb10-target-path.md) was 96.82% GPU-active, with GEMMs
accounting for approximately 92.65% of that activity. Both studies predate
the final FP8-draft configuration, so its component profile needs refreshing.

Use SM12x kernel assumptions. CUTLASS explicitly supports block-scaled NVFP4
warp MMA on `sm_121a`; SM100/B200-specific execution and tile assumptions
should not be transferred without checking support and storage limits.
[CUTLASS warp operations](https://docs.nvidia.com/cutlass/4.6.0/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warp.html).

**The 100 TPS budgets are materially different.** For this B1 benchmark:

```text
L = TTFT + 999 * R / A
decode TPS = A / R
end-to-end output TPS = 1000 / L

A = mean emitted tokens per round, including correction/bonus
R = complete round time in seconds
```

At 3.96 s TTFT, 100 end-to-end TPS allows 6.04 s for the remaining 999 tokens:
approximately 165.4 decode TPS. Holding round cost fixed gives:

| Hypothetical complete round cost | Tokens/round needed for 100 decode TPS | Tokens/round needed for 100 end-to-end TPS |
|---|---:|---:|
| 107 ms | 10.70 | 17.70 |
| 95 ms | 9.50 | 15.71 |
| 90 ms | 9.00 | 14.89 |
| 80 ms | 8.00 | 13.23 |

The current block emits at most 16 tokens. At 107 ms and perfect acceptance,
the optimistic estimate is about 94 end-to-end TPS. Actual round cost can
also change with acceptance because feedback and reconstruction process more
rows. Therefore 100 end-to-end TPS needs a faster round, faster cold prefill,
or a wider block with sufficient acceptance. Simply increasing width is not
a demonstrated solution and requires extending validated limits.

At unchanged current acceptance, 100 decode TPS would require approximately
47.3 ms rounds for code, 36.7 ms for design, and 60.5 ms for reasoning. All
are below the historical ideal target projection-read floor. Better acceptance
or a change to the weight traffic is therefore central to this target.

**I would rank the existing optimization ideas as follows.**

| Priority | Experiment | Why and what would justify continuing |
|---|---|---|
| 1 | Profile the actual FP8-draft configuration and stabilize comparison inputs/trajectories | Record natural early/middle/late rounds; split target FP4, target FP8, draft, feedback, reconstruction, and exposed host gaps. Keep uninstrumented AIPerf for latency. This identifies the current budget before more kernels are written. |
| 2 | Train a target-specific selector; expand to draft distillation if necessary | Optimize emitted prefix length per complete round, using the actual quantized target and deployment draft precision. Split by task/repository before training; measure whole held-out generations and quality. This is the main plausible mechanism for a large gain. |
| 3 | Calibrated mixed FP4/FP8 target compression | Reduce the remaining FP8 traffic selectively. Start with representative prefill and decode activation calibration and long-rollout sensitivity; retain FP8 where necessary. Evaluate the resulting checkpoint in both engines. |
| 4 | SM121 matrix tuning, guided by the new trace | Target the actual 16-row verifier and 15/16-row draft shapes, then protect B4 and prefill. Sweep layouts, tile shape, stages, and registers only with evidence of a relevant bottleneck. Preserve required rounding or label a changed numerical profile. |
| 5 | Remaining fusion / device progress work | Useful incremental work if the refreshed trace exposes a meaningful cost. Direct recurrent reconstruction and numerous norm/conv/feedback fusions already exist. Device-owned progress requires changes to page/state commit and cancellation, for a historically small exposed-host budget. |
| 6 | Performance-core placement and CPU contention | Measure idle and realistic agent-side CPU load. Pin the scheduling/launch thread and separate tokenizer/tool activity where useful; report tails and shared-memory contention. A practical tuning experiment, with no measured large B1 upside yet. |
| 7 | Cache reclamation, event-based restore, prepared weights, SSD sessions | Valuable for repeated turns, memory pressure, and startup. Measure saved agent-session time and first-token latency separately; these do not remove target traversals for a fresh 10K/1K request. |

The selection opportunity needs a stronger qualification than “train a better
selector.” The [existing oracle-support study](gb10-next-speedups.md) found
perfect-selection ceilings of 6.58, 6.81, and 10.41 emitted tokens per round
on 96 early histories per task. At a hypothetical 107 ms round those imply
about 61.5, 63.7, and 97.3 decode TPS. These are future-aware ceilings on
historical fixed candidate support, not attainable predictions or current
full-generation estimates. For code/design, those samples suggest selector
training alone may be insufficient for 100 TPS: candidate support/backbone
quality and/or target execution cost may also need improvement.

The earlier FP8-to-FP4 projection experiment found 1.4–1.7x local speedups,
but five of ten projections failed its numerical screen. Converting every
remaining FP8 operand would reduce the logical projection inventory from
about 17.56 GB to 14.40 GB after scales, approximately 18% fewer bytes.
Selective conversion saves less. This can complement better drafting but
does not supply a standalone twofold gain. NVIDIA Model Optimizer supports
calibrated NVFP4 and mixed-precision layer selection; compatibility and quality
for this checkpoint remain experimental.
[Projection results](gb10-optimization-results.md),
[NVIDIA quantization workflow](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/README.md).

**Additional experiments worth considering are bounded by the same budget.**

1. A smaller or vocabulary-restricted draft head could reduce the draft's
   vocabulary projection traffic. Sharar currently scores all 248,320 tokens
   before choosing the top 16. Keep the target's verification head complete;
   measure draft support recall and accepted-prefix loss against milliseconds
   saved. For sampling, report the actual restricted proposal law to rejection
   sampling. SGLang's FR-Spec provides a precedent for vocabulary restriction
   in EAGLE, not a ready-made DFlash2 feature. Historical local top-k cost was
   only 0.14 ms and the two vocabulary GEMMs together about 6.2 ms, so this is
   incremental unless the new profile changes that assessment.
   [SGLang FR-Spec](https://docs.sglang.io/docs/advanced_features/speculative_decoding),
   [local draft profile](draft-selection.md), [matrix profile](nvfp4-cost.md).
2. Train proposal confidence for the quantity that matters: the probability
   that the entire prefix through position j survives. Use measured marginal
   verifier/draft cost to choose whether a longer block is worthwhile. This
   differs from raising a local token-probability threshold. Revisit adaptive
   width only with a newly successful offline screen, since previous policies
   were archived and reducing target rows often saves little weight traffic.
3. Build a task-level agent replay alongside the 10K/1K regression fixture:
   repository edits with executable tests, schema-validated tool calls,
   tool-result continuations, shared-prefix branches, and sampled/thinking
   configurations actually used by the agent. Measure successful task time,
   retries, TTFT, and delivery gaps. A faster failed tool call is not useful
   agent acceleration; the existing four canaries do not cover this.
4. If model choice becomes flexible, evaluate a smaller dense target or a
   low-active-parameter MoE with a compatible draft using that same replay.
   Fewer active weight bytes change the hardware budget more substantially.
   Sharar currently has no MoE or second-target implementation, so assess
   quality and throughput in an existing supported engine before expanding
   its architecture. No specific alternative is asserted to achieve 100 TPS.

**SGLang remains a serious comparator.** The local SparkLang launcher and
`python/sparklang/adapters/sglang.py` describe the older plain-decode reference.
The latest performance evidence instead uses a native DFlash2 SGLang image
with recorded source `5f55db35e` and digest
`sha256:60166a99b6610e2b4b3febbeb6411e4cbc9338e996ab6c80d3339f86a7b5b9c3`.
Do not substitute the old launcher, the local checkout HEAD, or a generic
SGLang image and call it the same reference.

The matched recent comparison uses FP8 target KV, FP32 recurrent state,
16-row verification, maximum batch four, 60,000 KV tokens, 24 state slots,
2048-token prefill chunks, compilation and CUDA graphs. SGLang's draft is
BF16. Its hybrid state capacity can reduce actual admission below the
requested concurrency: an earlier four-client study saw only three decode
requests together. Check resolved limits and actual concurrent execution
before claiming aggregate throughput advantages. New draft training or
compression can also help SGLang; compare identical resulting artifacts
where supported.

Numerical correctness needs an explicit control. The target repairs matched
65 common-history verification blocks, 307 accepted rows, their top-10 logits,
and committed recurrent states. Complete free-running outputs still diverge.
Use common-history, common-boundary replay to isolate execution changes,
then independent generations and task-quality checks to assess deployment.
Also test Sharar's own ordinary/speculative and width-dependent behavior;
cross-engine equality is not the only useful correctness question.
Floating-point batch/reduction variation is a recognized issue in SGLang;
its deterministic-inference documentation does not establish that its flags
solve this specific pinned GB10/hybrid/DFlash2 profile.
[Target repair evidence](gb10-target-fix.md),
[SGLang deterministic inference](https://docs.sglang.io/docs/advanced_features/deterministic_inference).

**Several source/documentation details can otherwise misdirect the next run.**

- `run-sharar.sh` selects DFlash2, depth 15, and reconstruction. It now accepts
  `SHARAR_DRAFT_PRECISION=fp8`; the launcher and CLI/server default is BF16. The benchmark
  client measures whichever server is running; it does not enable FP8 or
  rebuild an old image. The optimized serving configuration is explicitly
  `sharar serve --proposer dflash2 --spec 15 --verify-state reconstruct --draft-precision fp8`.
- `bench/gb10_audit.py` now defaults to an explicit FP8 draft and accepts
  frozen benchmark inputs, all three task types, full candidate captures,
  and separate early/middle/late profiles. Older audit results used BF16.
- `docs/interfaces.md` and `CONTRIBUTING.md` now describe the DFlash2 depth
  correctly. The generic `ServingProfile` still declares depth seven; the actual DFlash2
  adapter declares `max_depth=15`, and `Executor.set_proposer` validates
  against the adapter. Some older documents also describe sampled host syncs
  that the current implementation has already removed.
- Statements in dated measurement documents about dirty trees, production
  images, or opt-in defaults describe those experiments. The inspected
  Sharar checkout was clean before this review; the benchmark and FP8 work
  have merged. The follow-up records the inspected live image and original
  command, and restores that original service after isolated GPU experiments.

**Do not repeat the failed screens without a changed mechanism.** Narrow
NVFP4 tiles regressed B1; earlier FP8 verifier tuning saved only 1.97–2.90%
per round; QKV/Z fusion saved 0.58%; cheap lattice lookahead improved coverage
only 0.688%; two-path/tree screens missed their full-completion gates; and
mixed prefill/verify forwarding hurt arrival TTFT. History lookup was also
screened and had insufficient coverage on these tasks. Existing evidence
does not rule out improved versions, but each needs a specific reason its
outcome should differ.
[Tile study](nvfp4-cost.md), [FP8 study](atlas-verifier-study.md),
[fusion study](gb10-target-path.md), [selector screen](gb10-next-speedups.md),
[tree study](tree-coverage-study.md), [mixed forwarding](mixed-forward.md),
[lookup screen](gb10-target-fix.md).

The [follow-up](gb10-fp8-profile.md) now completes the current FP8 profile,
full candidate capture, small selector fit and expanded compression screen.
It reproduces target-matrix dominance and rejects the cheap selector and
uniform target conversion. The next training experiment needs independent
task/repository splits, hidden-state supervision and a prefix-length objective.
Use 60, 80 and 100 decode TPS as measured milestones, preserving the separate
cold end-to-end target and reporting regressions on the hardest tasks.
