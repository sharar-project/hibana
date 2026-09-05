# GB10 optimization experiments

The goal is faster local inference through hardware-specific execution and
memory design as well as efficient kernels. Phase 7B supplies ownership,
proposer and feature seams. Phase 7C freezes workloads and profiles the first
experiments; 7D and 8 implement promising speculative/cache/memory mechanisms.
Research starts alongside measurement, rather than waiting for a tuning phase.

These are hypotheses, not established speedups or claims of original invention.
Before publishing a novelty claim, compare the mechanism with current engine
implementations and relevant papers. Preserve the simple reference path so
each experiment can be accepted or discarded independently.

## Hardware facts we must respect

GB10 shares physical DRAM between CPU and GPU. That does not mean every CUDA
allocation is coherently accessible by the CPU: NVIDIA explicitly distinguishes
`cudaMalloc` allocations and host allocation paths on this platform. A mapped
allocation experiment must establish its actual visibility and synchronization
contract. [NVIDIA CUDA porting notes](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html).

Use host `MemAvailable` alongside allocator and process measurements; CUDA's
free-memory report does not account for OS-reclaimable memory in the same way.
CPU activity competes for the shared memory system, so benchmark background
CPU traffic and interactive load as well as an idle machine. [NVIDIA UMA and
optimization guidance](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html).

## Candidate mechanisms

| Experiment | Hypothesis and implementation seam | Evidence needed to continue |
|---|---|---|
| Device-resident speculative progress | Keep proposal acceptance, processed lengths, page-tail selection and recurrent commit metadata on the device; notify the CPU through bounded completion records. `SpecDecoder` and `StateManager` expose the transaction to replace. This could remove a CPU dependency from every round and reduce repeated metadata preparation. | A trace attributes exposed time to the current acceptance transfer/host launch path. A prototype removes that dependency, preserves EOS/cancel/replay semantics and improves client-visible gaps/completion time. Reject if completion visibility, launch planning or polling consumes the gain. |
| Selective recurrent checkpoints | Current verification materializes recurrent snapshots for every candidate position. Compare selective checkpoints plus bounded replay/recomputation, or a fused accept-and-state-selection design, while retaining FP32 state. `SpecDecoder` owns the intermediate buffers and `StateManager.commit` owns publication. | Measure bytes actually written/read, not allocated capacity. Preserve the exact snapshot at every possible stop/accept position. Continue only if saved bandwidth and capacity exceed replay/extra synchronization costs across short budgets and real acceptance lengths. |
| Session cache optimized for saved time per byte | Choose target recurrent snapshots, target KV, draft features/KV and reconstruction checkpoints jointly. A long prefix is useful only if its recurrent and draft state can be restored cheaply. `memory/` owns policy; proposers declare retained state needs. | Replay multi-turn coding/chat sessions under equal physical memory budgets. Count end-to-end time saved, draft reconstruction, misses and eviction cost. Compare with the existing radix policy; larger cached token counts alone do not pass. |
| Reclaimable unified-memory slabs | Split optional cached state into reclaimable slabs while keeping captured graph references valid through explicit indirection/lifetime rules. Shared DRAM may support a larger useful session working set, but returning IDs to current pools does not release it. | Demonstrate physical memory recovery under external pressure, correct active loans and graph replay, and acceptable hit latency. Compare device, managed and mapped paths, including first touch, CPU contention and steady bandwidth. Reject a hot tier that loses the inference gain. |
| Prepacked immutable weight layout | Prepare the final NVFP4/FP8 layout once, then compare direct loading and registered/mapped access to the artifact. Eliminate redundant materialization/swizzling where the allocation path permits. `weights/` owns the layout and versioned identity. | Bitwise weight identity, lower load peak and startup time, and unchanged hot GEMM/GEMV bandwidth. Measure reads, faults, transformation, registration and first inference separately; shared memory alone is not proof that a copy disappears. |
| Adaptive local execution | Select speculative width and prefill allowance using measured round cost, cache/draft readiness, remaining output budget and queued interactive work. Scheduler chooses policy; executor/proposer expose bounded capabilities. | Better completion time and inter-token gaps under mixed arrivals and natural stopping at the same quality and memory limits. Higher acceptance or aggregate throughput alone does not pass. Keep an explicit fixed policy as the control. |

## First experiment and acceptance procedure

Start with a fixed-token GB10 trace of ordinary decode and MTP at batch 1,
then repeat at batch 4 and a longer prefix. Attribute CPU preparation, GPU
compute, weight/KV/recurrent traffic, acceptance synchronization and state commit.
Record the actual shapes and byte counts of the recurrent intermediate writes.
For this checkpoint, the current k=3 layout reserves 576 MiB of FP32 SSM
snapshots plus 11.25 MiB of convolution windows per batch row. At batch 4 that
is about 2.29 GiB, before draft KV or graph/workspace allocations. The SSM
calculation is `48 layers × 4 verify inputs × 48 heads × 128 × 128 × 4 bytes`.
This is allocated capacity, not a measured DRAM traffic counter. Selective
checkpoints may matter most by enabling wider verification under the same
memory limit; test that hypothesis separately from speeding up today's k=3.

The first prototype should address whichever of **device-resident progress**
or **selective recurrent checkpoints** has measured headroom. If both have
little exposed cost, move to the session-cache experiment rather than assuming
the proposed mechanism must help.

Before implementing a prototype, freeze the workload manifest, baseline and
decision threshold. A useful initial target is at least a 10% reduction in
completion time or a material, demonstrated capacity gain on its named local
workload, with no more than 5% regression on protected workloads. These are
proposed experiment gates for 7C to freeze, not current performance claims.
Use independent repeated runs and confidence intervals appropriate to the
sample count. Do not infer tail latency from a handful of requests.

Each experiment records:

1. The bottleneck evidence, predicted traffic/synchronization reduction and
   implementation owner; identify the CPU/GPU coherence and stream assumptions.
2. Pinned source/image/compiler, model/draft/tokenizer, shapes, allocation path,
   cache state, sampling, clocks/power conditions and competing host load.
3. An ablation against the same target work; separate kernel-cost comparisons
   from generation runs when changed continuations alter the amount of work.
4. A1–A8 and relevant feature/graph/state tests, quality results, raw latency
   samples, emitted tokens and peak/available memory.
5. A keep/remove decision against the declared threshold, including negative
   results. Promote the mechanism only after the production gates pass.

Broader comparison with current SGLang, DSpark, DFlash2 and other validated
GB10 engines belongs in 7C. The 7B unchanged-profile comparison establishes
refactor cost; it cannot establish a competitive lead.

## Selected 7C experiment: reconstruct the accepted recurrent state

The corrected 7C GB10 traces put MTP k=3 device activity at 96–98% of the
instrumented interval across B=1/4 and T=128/8,192. The independent unprofiled
rounds take about 109–137 ms. These traces include instrumentation, so the
remaining gap is not all removable host overhead. Removing host round trips
alone has no evidence here for the frozen 10% completion-time target.

The current verifier instead offers a concrete capacity target: 2.25 GiB of
FP32 recurrent snapshots plus 45 MiB of convolution windows, allocated for
B=4, k=3. Every layer computes recurrent state sequentially anyway. The
experiment will retain compact verified recurrence inputs and reconstruct
only the state after the accepted prefix once acceptance/EOS/budget is known.
This is a hardware-specific state-retention experiment, not a claim that the
algorithm is unprecedented or that its speedup is already demonstrated.

- **Owner:** target verification in `engine/spec.py`, recurrence kernels in
  `kernels/fla/`, and the existing `StateManager` commit boundary. Proposers,
  scheduler policy and HTTP should not depend on which retention mode is used.
- **Control:** preserve the current full FP32 per-token snapshots and graph
  path. Add an explicit startup experiment flag; default serving stays on the
  control until its gates pass. No BF16 state compression as a shortcut.
- **Mechanism:** save the verified projected/gated recurrence inputs, replay
  exactly the accepted input count from the untouched committed state into a
  single final-state scratch buffer, then use the existing commit operation.
  Preserve the same recurrence arithmetic/order, convolution windows, pending
  input and selected-feature lifetimes. Choose whether to retain zero or a
  small number of intermediate checkpoints from measured replay cost.
- **Memory hypothesis:** one final FP32 scratch state per row rather than
  four per-token snapshots reduces that allocation by 75% before input-stash
  overhead. Measure the **total** retained inputs, final scratch, graph pools
  and workspace; a smaller named tensor alone does not pass. Shared GB10 DRAM
  makes that saved capacity available to the local process/OS only if the
  allocation is genuinely removed, not merely marked as free IDs.
- **Cost hypothesis:** extra accepted-prefix recurrence competes with saved
  intermediate writes on the same GB10 memory system. Do not assume a latency
  gain. Measure fixed inputs/acceptance positions to hold work constant, then
  the frozen natural-output HTTP workloads. Physical DRAM counters are still
  needed for a traffic claim; the current trace only measures time/activity.
- **Coherence/lifetime:** all verification, replay and commit run on the
  engine's ordered CUDA stream. Graph outputs remain borrowed until the next
  replay. Committed state cannot change before the accepted-prefix replay has
  read it. No new concurrent CPU access to hot state or managed-memory faulting
  is part of this experiment.
- **Proof before keeping:** bitwise committed FP32 SSM/convolution state for
  every accepted/EOS/budget position, B=1/2/3/4, eager/graphs, cache miss/hit,
  continuation, cancellation and failure cleanup. Retain A1–A8 and both
  32K boundary cases. Check plain and MTP, including sampled fallback.
- **Keep/remove gate, frozen in `local_v1.json`:** at least 25% reduction in
  total verify snapshot allocation at fixed B/k, no more than 5% protected
  completion-time regression, all quality checks and the 16 GiB floor. A
  speed claim additionally needs the 10% selected-workload target and repeated
  independent confirmation. Wider k and more cached sessions are separate
  experiments; neither may substitute for an equal-profile comparison.

If replay cost defeats those gates, discard this prototype and test session
cache value or verified-prefix draft readiness next. The alternative of
device-resident progress remains useful if later scheduling/width changes
expose a larger host bottleneck.

## 7D implementation and next mechanism

[The 7D experiment](phase7d.md) implements compact-input reconstruction while
retaining full snapshots as the default control. Total verify buffers at B=4,
k=3 fall from 2,349 to 633.28 MiB (73.04%). The initial equal-strategy HTTP
comparison shows small latency changes, not a demonstrated 10% gain. Keep
capacity, measured CUDA allocation and physical traffic claims separate.
The new MTP cache/bonus-KV repairs add necessary work; their corrected baseline
must replace the old broken path as the optimization control.

Streaming checkpoint-map release is also implemented: the measured net host
load peak falls from 39.40 to about 21.3 GiB with unchanged CUDA peaks and
load time. It removes overlapping host mappings, not the CPU-to-CUDA copy.
Test versioned prepacked layouts and allocation visibility before claiming
zero-copy load or placing hot weights in mapped/managed memory.

The next state-retention prototype should let StateManager authorize a fused
reconstruction into the committed slots after every count is validated. It
could remove the final FP32 scratch and subsequent copy. Source state must be
read before each tile writes its result; accepted page tails and feature
publication must still obey the shared transaction boundary. Measure it against
the current reconstruction path, including batch padding and early stop.

For speed, first profile the newly working DFlash2/DSpark graphs: parallel
draft projection, conditional candidate selection and accepted feature-to-KV
ingestion can expose costs absent from the MTP trace. Fuse only measured
dominant work. Maintain an independent BF16 arithmetic oracle and the common
exact stochastic verifier. Current fixed-width pilot costs differ only modestly,
so confidence tuning alone is not an established large speed opportunity.

The [DFlash2 follow-up](draft-selection.md) now supplies that profile. An exact
device traversal removes 52 proposer stream synchronizations per B=4 round;
interleaved confirmation lowers fixed-work round time 2.8–5.4%, with all HTTP
token outputs unchanged in the separate pilot. The vocabulary top-k itself is
only 0.14 ms at B=4, so it is not the next priority.

The [accepted-feature batching follow-up](accepted-batch.md) implements the
next weight-reuse experiment: feedback cost falls from about 8.1 to 2.2 ms
at B=4, with one shared feature/KV projection pass. A pinned FP32 oracle checks
BF16 shape changes; private rings, endpoint snapshots and single-row arithmetic
retain their contracts. This establishes a small measured step before a larger
mixed target forward. Direct reconstruction into validated
committed slots remains the complementary capacity/copy experiment above.

The 250 ms separate-prefill feedback pilot is rejected for default use: it
reduced observed stream stalls but more than doubled mixed-session completion
time. A new hypothesis is to share target projections across prefill tokens
and short verification blocks, then run their distinct recurrent operations
and preserve their separate commits. This may amortize another weight traversal
on bandwidth-limited GB10. First attribute the duplicated work in a mixed trace;
then compare a bounded mixed-forward prototype with the existing separate
forwards, at equal memory and the same arrival/quality gate. Do not replace
clear ownership with a general scheduler framework to attempt this experiment.
