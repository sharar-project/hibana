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
