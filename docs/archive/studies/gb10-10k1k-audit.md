# GB10 audit for 10,000 input / 1,000 output tokens

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

2026-09-08. Engine source: `bef1456`; production image and experiment commands
are recorded under `${SHARAR_RESULTS}/reviews/2026-09-08-gb10/`.

The latest [TPS validation and push review](gb10-tps.md) reruns AIPerf with five
timed requests per task and engine. The FP8 draft configuration measures
37.65 / 30.14 / 46.09 output tokens/s including prefill for code / design /
reasoning. The report includes decode TPS, fresh and preceding SGLang loads,
token audits and repository checks. Full cross-engine text equality remains
unresolved; the recommendation is a review PR with FP8 kept opt-in.

The preceding [target arithmetic repairs and benchmark iteration](gb10-target-fix.md)
fix prefill and compiled verification differences. On 65 identical SGLang
verification blocks, all 307 accepted rows, their top-10 logits and every
committed recurrent state match exactly. Full independent greedy generations
still diverge. With the explicit FP8 draft option, final AIPerf measures
26.561 / 33.111 / 21.660 seconds for code / design / reasoning: 17.16% / 4.69% /
12.01% lower than the faster SGLang mean per task across two fresh loads.
These are complete-configuration timings with different cross-engine text,
not a wide matched-output win. See the report for both draft configurations,
reference reload variation and validation; production retains its original image.

The earlier [target-path investigation](gb10-target-path.md) repairs incorrect
FP4 quarter-tie rounding and finds a separate SiLU/BF16 quantization mismatch.
Fresh AIPerf measures 34.427 s / 269 rounds for the tie-corrected code run.
A reference-like SiLU candidate reaches 31.236 s but regresses design and
reasoning, so it is not promoted. QKV/Z fusion saves only 0.58%. The new trace
is 96.82% GPU-active; GEMMs account for about 92.65% of that activity. A wide
matched-output win remains unproven, and production retains its original image.

The earlier [exact-arithmetic engine changes](gb10-engine-speedups.md) reduce
code latency from **35.144 to 33.803 s (3.82%)** in a fresh paired AIPerf run.
Design improves 3.82% and reasoning 3.36%. All 24 requests, including warmups,
retain the previous Sharar baseline's exact output IDs and text. Code is still
9.38% slower than the earlier SGLang measurement; cross-engine greedy parity
remains unresolved. Those measurements precede the FP4 tie repair. These
changes are validated in source; production retains its original image.

The [baseline Sharar–SGLang AIPerf comparison](gb10-aiperf-results.md) verifies identical
10,000-token inputs and exactly 1,000 output IDs for both engines. Sharar takes
34.969 s versus SGLang's 30.904 s on code, and less time on design and reasoning.
Cross-engine text equality fails on all three prompts. The code gap comes from
264 versus 205 verification rounds; Sharar's effective decode round is cheaper
(118.3 versus 130.3 ms). First-token scores localize the code divergence to
target prefill computation, before draft acceptance.

The [implementation follow-up and achieved speedups](gb10-optimization-results.md)
cover the subsequent engine changes. This audit retains the original baseline
measurements and optimization hypotheses.

The [next speed priorities](gb10-next-speedups.md) combine the measured cost
budget with a new offline selector-support ceiling and a failed cheap-lookahead
screen. After the exact fusions, they prioritize target-execution parity,
selector training feasibility and calibrated target compression.

The subsequent [acceptance parity investigation](gb10-acceptance-parity.md)
finds that the 264-versus-205 round comparison follows different generated
continuations. On 128 identical SGLang histories, original Sharar's matched
draft-prefix-plus-correction total is 609 versus 617, a 1.3% deficit. The raw
latency measurements below remain valid, but they do not establish a large
draft porting defect. Reference arithmetic variants fail performance or quality
gates, so none changes the serving defaults.

The principal opportunity is **more accepted tokens per target traversal**.
The current code task produces about 3.8 tokens per 117–120 ms single-request
round. Most of that interval is GPU work. A substantial lead over a tuned
SGLang therefore needs better proposals, fewer bytes in the hot matrices, or
both. Another small normalization kernel cannot produce a twofold gain.

This audit adds `bench/gb10_audit.py` and raw experiments, without changing
serving execution or the production image. It distinguishes full native
generation, independent event/trace instrumentation, and unimplemented
performance hypotheses. The experiments use the pinned Qwen3.8-27B NVFP4
target and pinned DFlash2 draft, temperature zero, and exact token budgets.
EOS is ignored for fixed work; these runs do not establish answer quality.

The request did not specify concurrency. Single-request latency is the
primary interpretation, with batch/concurrency four examined separately.
Here 10k means 10,000 tokens and 1k means 1,000, not 10,240 and 1,024.

The execution path is small and already specialized. `api/server.py` owns
the HTTP/event-loop boundary; one `EngineLoop` thread calls the scheduler.
The scheduler admits requests, performs bounded prefill, decodes, then
publishes output. It can execute two prefill forwards before one decode
round. `engine/executor.py` assembles tensors and FlashInfer plans, and
`model/qwen38.py` runs 64 dense decoder layers. Forty-eight use recurrent
Gated DeltaNet; sixteen use full attention. This is not a conventional
64-layer full-attention transformer and it is not an MoE model.

`weights/modelopt.py` loads mixed precision: the 64 MLP gate/up and down
pairs and vocabulary head are NVFP4 W4A4; attention and GDN main projections
are FP8 W8A8; embeddings, smaller gates and draft weights are BF16. Each
GDN layer splits qkv/z into two contiguous projections, executes the causal
convolution and recurrence, then gated normalization and an output projection.
The MLP already fuses SiLU, multiplication and NVFP4 activation packing.
Input residual normalization already emits the first projection's FP8 input.

In a speculative round, `proposers/parallel.py` runs the five-layer DFlash2
backbone over the anchor/noise block, then scores a sparse conditional
candidate lattice with the shared target vocabulary head. Its context comes
from target post-block features at layers 5, 19, 33, 47 and 61, projected into
a private 2,048-token draft KV ring. `engine/spec.py` verifies the pending
input plus proposals; the target and draft forwards use CUDA graphs.
Greedy acceptance performs one packed device-to-host download. Stop and
output-budget truncation precede the committed-state update. The final
emitted token remains the next unprocessed input.

`memory/request_state.py` owns target KV reservations, recurrent slots,
rollback and prefix loans. Verification leaves committed recurrent state
untouched. `kernels/fla/accepted_state.py` reconstructs the accepted prefix
directly into validated private FP32 slots with one kernel across all GDN
layers. The proposer then projects accepted features and updates its own
ring. These ownership boundaries must survive any fusion or graph capture.
The cold tier and prefix cache can avoid repeated prefill, but they do not
avoid computation for a new cold 10k prompt.

**Hardware facts that change the implementation choices.** NVIDIA specifies
128 GB coherent LPDDR5X and 273 GB/s. Its advertised 1 PFLOP FP4 rate uses
sparsity; it is not the dense throughput of this checkpoint. The 20 CPU
cores are ten Cortex-X925 plus ten Cortex-A725.
[NVIDIA specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/).

The actual device query in this audit reports:

| Property | This GB10 |
|---|---:|
| CUDA capability | 12.1 |
| SMs | 48 |
| L2 | 24 MiB |
| Shared memory per SM | 100 KiB |
| Maximum opt-in shared memory per block | 99 KiB |
| Registers per SM | 65,536 32-bit registers |
| Threads resident per SM | 1,536 |
| Total CUDA-visible memory | 128,524,255,232 bytes |
| Driver / CUDA runtime | 580.95.05 / 13.0 |

Use the query, not a B200 tuning table, for occupancy and storage budgets.
For native block-scaled kernels use the SM12x warp MMA path. CUTLASS explicitly
supports `sm_120f` and `sm_121a` for its NVFP4 warp operations; the SM100
`tcgen05`/tensor-memory programming model is a different path.
The CUDA guide lists `compute_120f` as compatible with 12.0 and 12.1.
[CUTLASS warp MMA](https://docs.nvidia.com/cutlass/4.6.0/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warp.html),
[CUDA compiler feature targets](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).
Current traces already contain SM121 TMA matrix kernels and SM120
block-scaled GEMMs. Tensor Core and TMA support are already being used.

The shared physical pool does not mean every CUDA pointer is safely
CPU-dereferenceable. NVIDIA specifically documents restrictions on coherent
CPU/I/O access to device allocations on Spark. Keep hot operands in their
validated device allocations, and manage cold/mapped buffers through explicit
allocation, ownership and synchronization contracts.
[Spark CUDA porting guidance](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html).
The existing host `MemAvailable` ledger is appropriate for UMA; CUDA allocation
figures alone omit reclaimable OS memory.
[NVIDIA UMA memory accounting](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html).

**Fresh production baseline.** The existing HTTP benchmark ran one warmup
and three measured batches at each concurrency, with cache flushes and the
production memory guard. All 20 requests completed their exact 1,000-token
budget; four quality canaries passed. Minimum host available memory was
55.96 GiB. The measured filler requests intentionally have different unique
prefixes, so their spread is not a repeated-identical-input noise estimate.

| Concurrency | Measured aggregate output tok/s | Median aggregate tok/s | Median TTFT |
|---:|---|---:|---:|
| 1 | 80.68, 27.16, 24.22 | 27.16 | 3.747 s |
| 4 | 60.77, 58.80, 52.01 | 58.80 | about 12.70 s |

At concurrency one the fast request averaged 13.51 emitted tokens per
verification round; the other measured requests averaged 3.56 and 3.19.
Their end-to-end times were 12.39, 36.82 and 41.28 seconds. Length alone
therefore does not define this speculative benchmark. Preserve exact inputs,
outputs and acceptance distributions; do not choose a favorable filler prefix
and describe that as general Qwen speed.

**Matched HTTP comparison.** A separate comparison reuses the exact frozen
code input IDs, the same pinned target/draft, cold caches, greedy generation,
and 1,000 output tokens. It runs one warmup and two measured repetitions per
cell, excluding model loading, compilation and graph capture. Sharar uses
the audit's bounded pools; the reference image is
`lmsysorg/sglang:dev-qwen38-27b-dflash2`, SGLang source `5f55db35e`, image
digest `sha256:60166a99b6610e2b4b3febbeb6411e4cbc9338e996ab6c80d3339f86a7b5b9c3`.
SGLang uses native DFlash2, Torch compilation, CUDA graphs, FlashInfer,
2,048-token prefill chunks, FP8 KV, FP32 recurrent state, 60,000 target KV
tokens and a requested maximum of four running requests.

| Engine/configuration | B1 measured completion times | B1 end-to-end output tok/s | Emitted tokens / verify round |
|---|---|---|---:|
| Sharar, 16 verify rows | 35.531, 35.293 s | 28.14, 28.33 | 3.788 |
| SGLang, 8 verify rows | 34.178, 34.357 s | 29.26, 29.11 | 4.082 |
| SGLang, 16 verify rows | 31.008, 30.774 s | 32.25, 32.50 | 4.878 |

Sharar is about **14.6% slower in B1 completion time than the faster SGLang
configuration** on this task (35.412 versus 30.891 seconds, means of two
samples). There is no measured wide-margin single-request win. Sharar needs
264 verification rounds versus SGLang-16's 205. Using the observed decode
durations, Sharar's round is roughly 120 ms versus SGLang's roughly 130 ms;
SGLang's greater acceptance outweighs that cost advantage. Output hashes are
stable across B1 repetitions within each configuration, but differ between
engines and widths. These are free-running system results, not an identical
trajectory comparison or an answer-quality equivalence proof.

Sharar's B1 TTFT is 3.735/3.750 seconds, versus 4.174/4.211 seconds for
SGLang-16. The first-token advantage does not make up for the extra rounds.

| Engine/configuration | Four-request measured makespans | Aggregate output tok/s |
|---|---|---|
| Sharar, 16 verify rows | 59.996, 56.591 s | 66.67, 70.68 |
| SGLang, 8 verify rows | 72.997, 83.225 s | 54.80, 48.06 |
| SGLang, 16 verify rows | 100.170, 95.437 s | 39.93, 41.91 |

The B4 logs reveal an admission caveat despite SGLang's reported capacity
four: only three requests decode together and the fourth remains queued
until one completes. For the 8-row configuration, that request's TTFT is
45.42 and 49.23 seconds. Its 24 recurrent slots and 60,000 KV-token capacity
are explicitly recorded, and the default minimum-free-slots delay is disabled
at this request limit. The precise remaining admission gate is unresolved;
do not promote the apparent B4 throughput advantage as a fair saturated-B4
kernel result. A follow-up should instrument the admission decision and test
additional request/state headroom while keeping the four-request workload.
The same delayed-fourth-request pattern occurs with sixteen verify rows;
its delayed TTFTs are 59.73 and 60.55 seconds. All completed measured and
warmup requests meet the exact token budget and report zero cached tokens.
Concurrent output hashes vary within both engines as scheduling changes.

The component, Sharar HTTP, SGLang-8 and SGLang-16 windows keep at least
79.86, 80.20, 66.48 and 62.28 GiB of host available memory respectively.
Among resource samples reporting at least 90% GPU utilization, median SM
clocks are approximately 2.48–2.50 GHz. Two initial SGLang startups failed
on unwritable compiler/JIT cache paths before measurement; their logs are
retained. Explicit writable cache directories resolve those setup errors.
Neither failed startup is counted as a reference request failure or latency.

**Fresh component profile.** The audit freezes a filler prompt and a source
review/code-generation prompt at exactly 10,000 input IDs, with four distinct
row prefixes. Full 1,000-token continuations run without the profiler, followed
by a fresh continuation with eight event-instrumented rounds and four traced
rounds after at least 256 outputs. Pools are bounded to 60,000 KV entries and
24 recurrent slots. Each request's prefill uses 2,048-token chunks; the
component study prefills requests serially before batch decode, so its B4
time is not a serving-scheduler benchmark.

| Workload | B | Native decode tok/s, k=15 | Target verify | Draft | Acceptance incl. state/feedback |
|---|---:|---:|---:|---:|---:|
| Filler | 1 | 33.78 | 92.20 ms | 20.23 ms | 6.25 ms |
| Filler | 4 | 96.30 | 114.10 ms | 22.52 ms | 9.24 ms |
| Code | 1 | 31.43 | 92.24 ms | 19.48 ms | 4.79 ms |
| Code | 4 | 94.92 | 114.85 ms | 22.85 ms | 9.23 ms |

Decode tok/s is aggregate for B4 and excludes the first prefill token.
Phase values are medians of CUDA-event spans from a different part of the
continuation; nested phases overlap and must not be added indiscriminately.
In particular, `_accept`'s roughly 114 ms host duration includes waiting for
previously queued draft/target work. It is not 114 ms of Python overhead.

On the code trace, B1 averages 120.30 ms wall and 116.18 ms device activity;
B4 averages 150.47 and 145.51 ms. Instrumentation affects both, but these
figures do not support removing 16 ms of launch gaps from every 10k round.
The older short-context estimate in `batch4-and-sharegpt.md` should not be
copied into this workload's speedup prediction.

The dominant B1 kernels are approximately 51.3 ms of NVFP4 GEMMs (including
both vocabulary projections), 36.3 ms of target FP8 GEMMs, and 17 ms of BF16
matrix products across draft/feedback and smaller target work. Target full
attention costs about 2.15 ms, GDN verification 2.08 ms, and accepted-state
reconstruction 1.32 ms in that trace. At B4 the attention and recurrent
components grow substantially; event reconstruction is about 5.40 ms.
Kernel names alone do not establish which BF16 operation a kernel serves.

The separate full 10k prefill trace includes about 1.10 seconds of main
NVFP4 MLP GEMMs, 0.82 seconds of the leading FP8 matrix kernels, 0.36 seconds
of paged full attention, 0.27 seconds of NVFP4 packing, 0.22 seconds of Gemma
normalization, and substantial GDN chunk work. These are diagnostic kernel
sums, not independently measured serving latencies. Native B1 prefill is
about 3.75 seconds. Prefill fusion matters for TTFT, but decode dominates
this 1,000-output task.

The inventory measures **17.555 GB of target projection operands**:
7.214 GB FP8 matrices/scales and 10.341 GB NVFP4 matrices/scales. It excludes
the embedding's 2.54 GB allocation from a whole-matrix read: embedding is
indexed. The draft allocates 3.866 GB of weights/RoPE tables, including
indexed codebooks. Neither total is a physical-traffic measurement.
At 273 GB/s, one read of the target projection operands alone takes 64.31 ms.
A large 1 GiB device copy in this session delivers approximately 221 GB/s
when counting both useful reads and writes; it is a copy-path diagnostic,
not the exact sustainable bandwidth of GEMM.

Nsight Compute counter access returned `ERR_NVGPUCTRPERM`; driver policy
was not changed. No new LPDDR utilization or DRAM-byte claim is made.
The earlier `nvfp4-cost.md` study did collect system-memory fills into L2,
about one packed weight footprint per sampled GEMM. That historical evidence
argues against a hidden multi-fold reread problem, but is not a fresh
whole-engine controller measurement.

For small matrix row counts, arithmetic intensity explains the behavior.
Ignoring output/activation traffic, W4 with one scale byte per 16 weights
provides approximately `2*M/0.5625` FLOP/byte: 57 at M=16 and 228 at M=64.
FP8 provides about `2*M`: 32 and 128. Both are much less compute-intensive
than M=2,048 prefill. Reusing weights across useful speculative rows is the
right hardware mapping; generating unused rows does not itself improve
useful throughput.

**Width ablation.** A separate loaded k=7 process, using the same frozen
code inputs, measures 33.13 versus 31.43 decode tok/s at B1, and 106.59
versus 94.92 at B4. B4 verification drops from 114.85 to 101.95 ms. These
are one-continuation diagnostics with different output IDs, not a promoted
default or an isolated equal-operand kernel speedup.

At k=15, the B1 code continuation has 264 rounds, a mean of 3.78 emitted
tokens per round, and **zero** full 16-token rounds. B4 has only three full
rounds among 1,170 request-rounds. Increasing the maximum to 31 is therefore
poorly motivated for this code workload. The checkpoint's training config
declares block size eight even though serving has validated sixteen rows.
Widening can change the draft's bidirectional context at early positions.
A recent study of other draft/target models reports this failure mode and
uses continuation training before expansion; it is a research direction,
not a result for this checkpoint.
[Acceptance histograms and trained block expansion](https://arxiv.org/abs/2608.30427).

**Draft-only FP8 feasibility experiment.** An isolated adapter converts the
five draft layers' q/o and gate/up/down matrices to E4M3, with per-tensor
weight scales and dynamic per-tensor activation scales. The selected 20
matrices occupy 3.093 GB in BF16 or 1.547 GB in FP8, before scales. Original
BF16 tensors remain allocated in this probe, so this is a hot-operand and
dispatch experiment, not a measured capacity reduction. Dispatch records
confirm every selected matrix uses the FP8 path. Feature projection, context
KV feedback, selector weights and all target execution remain native.

The protocol was saved before the run: same exact B1 code input, 1,000
output tokens, k=15, capacity four, and a feasibility gate of at least 5%
lower complete time while retaining at least 95% of baseline emitted tokens
per request-round.

| Metric | BF16 draft control | FP8 draft probe |
|---|---:|---:|
| Full prefill + decode | 35.544 s | 33.159 s |
| Decode only | 31.788 s | 29.395 s |
| Decode output tok/s | 31.43 | 33.98 |
| Emitted tokens / round | 3.784 | 3.756 |
| Verification rounds | 264 | 266 |
| Median draft event span | 19.48 ms | 13.30 ms |
| Median target verification event span | 92.24 ms | 91.88 ms |

The probe passes the **feasibility** gate: 6.7% lower complete time with
99.25% of the control's mean acceptance. It changes the generated output
IDs; this does not establish bitwise equivalence or task quality. It is one
continuation, with separately instrumented later rounds, and is not a
production promotion or an HTTP/SGLang speedup claim. Minimum host available
memory is 78.98 GiB. The adapter and its frozen protocol are retained as
`draft_fp8_trial.py` and `draft-fp8-protocol.json` in the evidence directory.

**Implementation priorities and expected scale.** These are experiment
predictions unless a measured result is stated above.

1. **Improve the draft against the actual quantized target.**
   The later [reference-history study](gb10-acceptance-parity.md) completes the
   comparison requested below and finds only a small aggregate deficit. Treat
   retraining as a research option, not a demonstrated repair for the SGLang gap.
   Collect held-out 10k-context, 1k-output code/chat/reasoning continuations with the pinned
   target's features and actual first-rejection tokens. Separate missing
   top-16 support, wrong selection inside support, and full-block clipping.
   Before training, compare Sharar and SGLang proposals on an identical
   teacher-forced history: post-layer features, feature projection, draft
   positions/window, candidate support and conditional selector scores.
   Different free-running outputs cannot distinguish a porting discrepancy
   from target numerical drift or a weaker draft distribution. Fix an
   identified discrepancy before paying to train around it.
   Fine-tune the existing selector first when coverage permits; otherwise
   distill the backbone. Match thinking/sampling/context distributions and
   the target numerical profile. Optimize accepted prefix length per unit
   of complete round time, not independent token accuracy. DFlash2 already
   implements local conditional selection, so adding a generic lattice walk
   duplicates existing work. A stronger lattice-conditioned selector, such
   as the trained mechanism in NVIDIA's LiLiCorr, is a concrete research
   candidate requiring compatible training and a GB10 cost check.
   [DFlash2's selector](https://inco.ai/blog/dflash2/),
   [NVIDIA LiLiCorr](https://research.nvidia.com/labs/nemotron/lilicorr/).

2. **Quantize the draft's large BF16 matrix products to native FP8.** Start
   with q/o and MLP matrices; keep sensitive feature normalization, selector
   and target verification unchanged. Measure added activation packing and
   actual acceptance loss. If round time falls from 120 to 110 ms, accepted
   tokens may fall by less than 8.3% before decode becomes slower. An
   initial probe above saves 6.18 ms in the instrumented draft phase and
   6.7% in complete continuation time. Next replace the isolated adapter
   with an explicit draft-linear abstraction, investigate fused activation
   packing, and run paired AB/BA HTTP tests across held-out tasks and B1/B4.
   Keep the exact
   stochastic acceptance law when sampling. Draft-only quantization can
   preserve target verification, whereas target quantization defines a new
   numerical model profile.

3. **Explore calibrated NVFP4 for selected currently-FP8 target matrices.**
   Converting all 7.214 GB of FP8 matrix operands to the current NVFP4
   storage ratio would save at most about 3.16 GB per target traversal,
   before alignment, new scales and activation costs: roughly 12–14 ms at
   273–221 GB/s. This is a material additional lever, not a twofold speedup
   by itself. Start with layerwise sensitivity and calibration using the
   original high-precision checkpoint if available; preserve sensitive
   recurrent projections as needed. Validate task quality, long recurrent
   trajectories and acceptance jointly. A cast of existing FP8 tensors is
   not a calibrated quantization recipe, and a new target must be compared
   at an explicit quality budget.

4. **Use an explicit narrow/wide policy only after a held-out width matrix.**
   Capture bounded shapes at startup and preserve a fixed control. The
   relevant tradeoff is `round_ms / accepted_tokens`, separately by B and
   context. A wide block is useful on copy-heavy text and can hurt harder
   batch-4 requests. Historical confidence/width policies failed their
   gates; do not revive them merely because one new prompt prefers k=7.
   Compare static k=3/7/15 and protect cold completion and arrival latency
   before considering hysteresis based on observed acceptance history.

5. **Graph and fuse the post-verification path as a bounded cleanup.** Pack
   accepted counts/slots into persistent buffers, gather the accepted
   feature rows, update draft KV/rings and copy convolution endpoints with
   masked kernels. Retain CPU ownership validation and publish only after
   the correct stream dependencies. A graph may reduce small launches and
   per-round allocations; it cannot remove the GPU work hidden inside the
   acceptance host wait. Use the fresh few-millisecond gap, rather than the
   older 16 ms estimate, as the starting budget. Reconstruction already
   reads and writes 144 MiB of FP32 SSM per request; at B4 that is about
   1.21 GB per replay, consistent with its roughly 5.4 ms time at the copy
   diagnostic's bandwidth. Large replay gains need less state traffic,
   not just a different Python wrapper.

   One concrete traffic experiment is to defer the accepted-prefix replay
   into the next GDN verification kernel, keeping the reconstructed state
   in registers while starting that layer's next verification. Eliminating
   one separate 144 MiB state read per request has a bandwidth-only budget
   of about 0.7 ms at B1 or 2.7 ms at B4, before occupancy and input costs.
   This is an unimplemented inference, with substantial ownership work:
   cache publication, request release, cancellation and branch loans must
   materialize any pending committed state, and raw replay inputs must not
   be overwritten early. Benchmark the combined kernel; fusing loops can
   increase register pressure enough to lose the traffic saving.

6. **Optimize prefill epilogues and data layout for TTFT.** First attempt
   paired gate/up output tiles whose epilogue computes SiLU/product and
   writes NVFP4 data plus block scales, avoiding BF16 materialization and
   rereads. The current gate/up outputs occupy separated halves along N;
   an ordinary local epilogue cannot multiply an unavailable partner tile.
   Compare a paired/interleaved tile design against the simpler two-GEMM
   auxiliary-input design, including the latter's extra launch/input cost.
   Preserve BF16 rounding before the existing FP32 SiLU/product/quantization
   if claiming exactness. The earlier study estimates roughly 6–12%
   prefill headroom, which is only about 0.6–1.3% of the 35.5-second code
   completion measured here. Conv-produced contiguous q/k/v and a retuned
   norm/quant kernel are smaller complementary changes. Do not confuse the
   useful combined savings with multiplicative independent speedups.

7. **Tune attention and short recurrent verification only where B4 pays.**
   Full attention is about 2 ms at B1 but around 8 ms in the B4 trace.
   Test split/reduction geometry and larger KV page sizes against actual
   head dimension 256, GQA ratio six, FP8 KV and growing 10k histories.
   Stable reduction geometry is also a numerical requirement: the repo's
   earlier fixed-operand study found attention reduction drift across verify
   shapes. Prior GB10 attention wavefront/cache-reordering research is a
   candidate kernel mechanism, not evidence of a model-level 60% gain.
   [GB10 attention scheduling study](https://arxiv.org/abs/2601.16032).
   Changes to GDN recurrence order, tensor-core block formulations or BF16
   state require explicit numerical validation. Existing FP32 replay and
   full-snapshot oracles remain useful controls.

8. **Keep systems work tied to its workload.** AOT tail-shape coverage,
   warmup completeness, optional-cache slab reclamation and P-core placement
   improve robustness/startup/tails. This host reports higher-frequency
   cores at IDs 5–9 and 15–19; test engine-thread affinity against the
   scheduler's current placement, with the same CPU contention on SGLang.
   Shared memory bandwidth means CPU model offload is not a second DRAM
   channel. NVLink multi-GPU, RDMA, paged cold-tier capacity and warm-prefix
   hits do not accelerate this single-device cold-input measurement.
   Do not introduce those mechanisms into the claimed cold 10k/1k win.

Existing two-path and fixed-tree studies failed their completion screens;
the latter projected only about 7.7% even after extra candidate coverage.
More nodes alone are not a fresh mechanism. Likewise, the shipped mixed
prefill/decode forward was removed after arrival latency regressions. Revisit
these only with new coverage or scheduling evidence, not by repeating a
previously rejected implementation.

Additional algorithm experiments should have narrow hypotheses. A smaller
distilled draft could save backbone traffic; a more expensive draft can also
win if its acceptance increase exceeds its complete-round cost increase.
Prompt/output n-gram lookup is a cheap proposal source for copying and editing
tasks, with normal target verification and a fallback when matches disappear.
It should be reported by workload, since repeated filler overstates its value.
Structured pruning or more aggressive target precision requires a new trained
or calibrated checkpoint and a measured quality budget; the advertised sparse
FP4 rate is not available by toggling a flag on these dense weights.

For kernel exploration, tune actual M=8/16/32/64 rows, projection dimensions
and scale layouts, including register/shared-memory occupancy and tile tails.
The previous isolated NVFP4 tile variants failed their screen, so another
custom GEMM needs a measured traffic or occupancy mechanism. A persistent
kernel cannot hold a 17.6 GB target in 24 MiB of L2. Treat persistent scheduling,
cache hints and launch fusion as improvements to measured overhead/reuse,
not ways to bypass compulsory matrix reads. FP8 KV compression is already
present; further KV compression addresses a small part of B1 round time at
10k compared with projection traffic.

For a concrete target, the B1 code continuation takes approximately
`3.756 + 31.788 = 35.544 s`. Keeping 3.756-second prefill and a 120 ms round,
a 1.5x completion speedup needs about **6.0 emitted tokens/round**; a 2x
completion speedup needs about **8.6**. If engineering reduced prefill to
3.2 seconds and rounds to 100 ms, 2x would still need about **6.9 tokens per
round**. These are conditional arithmetic targets, not predicted acceptance
or an already achieved lead over SGLang.

Against the **measured 30.891-second SGLang-16 baseline**, rather than against
current Sharar, a 1.5x completion lead requires about **7.1 tokens/round** at
120 ms and 3.756-second prefill; a 2x lead requires about **10.3**. At 100 ms
and 3.2-second prefill, those acceptance requirements become approximately
**5.7** and **8.2**. This makes the proposal-quality target explicit before
investing in a larger block or an expensive kernel rewrite.

A better draft or a new target quantization may also accelerate SGLang. A
twofold improvement over today's Sharar is not automatically a twofold engine
lead. Compare compatible new drafts/target profiles on both engines when
claiming an engine advantage; otherwise label the result as a complete
inference-system configuration comparison. With unchanged useful tokens per
round and unchanged matrix storage, the measured times and mandatory operand
reads do not support a twofold gain from kernel/host tuning alone.

Before a wide-margin claim, freeze multiple task families and exact input
IDs, output budgets, sampling, target/draft revisions and tokenizer/template
hashes. Use cold cache, separate engine windows, a common memory floor and
enough KV/state capacity for four simultaneous 11k contexts. Record effective
running capacity rather than just the requested value. Compare native
SGLang DFlash2 with both eight and sixteen verify rows, validate useful output
quality, report paired completion time and TTFT separately from decode TPS,
and retain all failures and slow tails. Repeat winning candidates in
independent AB/BA windows; two samples or one code task cannot certify a
broad performance lead.

Reproduction of the component audit, inside the GPU image with exclusive
model memory and the host guard:

```bash
python3 -m bench.gb10_audit --output /results/new-gb10-audit
python3 -m bench.gb10_audit --output /results/new-gb10-audit-k7 --depth 7 --cases code
```

Raw inputs, full output IDs, round histories, source hashes, allocation
figures, compressed traces, event spans, container commands and resource
samples are retained in the dated results directory. The host-side
`window.py` there records image identity and restores the original production
container and memory guard in `finally`.

The matched HTTP drivers and FP8 adapter also live in that directory:

```bash
python3 window.py http-sharar /proof/compare_job.py sharar --repeats 2
python3 window.py --image lmsysorg/sglang:dev-qwen38-27b-dflash2 http-sglang8 /proof/compare_job.py sglang --depth 7 --repeats 2
python3 window.py --image lmsysorg/sglang:dev-qwen38-27b-dflash2 http-sglang16 /proof/compare_job.py sglang --depth 15 --repeats 2
python3 window.py draft-fp8 /proof/draft_fp8_trial.py --output /proof/profile-draft-fp8 --cases code --batches 1 --max-batch 4
```

These host drivers are tied to the recorded local container/cache layout.
Use a new evidence directory with the drivers and frozen `profile-k15/inputs.json`;
their output directories deliberately refuse overwrite. The image digests
and full executed commands are saved with every window. The compact
`gb10-10k1k-results.json` manifest identifies the archived evidence bundle.

Validation: the existing CPU unit suite passes **338 tests, with seven
skipped**, in 6.69 seconds. The initial sandboxed attempt stalled after 151
passes and was interrupted; the rerun with local test networking available
completed. The new harness compiles, evidence-manifest checks pass, and the
diff passes whitespace checks. The original `sharar-qwen38` container is
healthy, its original image is running, the memory guard is active, and
`tools/verify-sharar.sh` passes model/profile checks and its deterministic
chat canary. No serving implementation or production image is promoted by
this audit.
