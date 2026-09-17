# Historical studies and decisions

Use [current performance](../performance.md), [operations](../operations.md)
and [the roadmap](../../roadmap.md) for today's behavior and work order.
Reports below retain their original source/protocol claims, including failed
gates. Historical commands may require the recorded source revision.

## Decisions to preserve

| Decision | Evidence |
|---|---|
| Retain exact BS1 QKV/MLP bundle; other numerical candidates unqualified | [BS1](studies/gb10-bs1-speedups.md) |
| Keep k=15 and existing BS4 scheduling defaults | [BS4](studies/gb10-bs4-engine.md) |
| Retain checkpoint-owned features; reject short-resume scheduler | [Retention](studies/gb10-agent-retention.md) |
| Mixed forwarding failed latency/quality protection | [Mixed forward](studies/mixed-forward.md) |
| Two-path/trees did not qualify a serving policy | [Two path](studies/two-path-study.md), [tree coverage](studies/tree-coverage-study.md) |
| Narrow NVFP4 tiles failed their integration screen | [NVFP4](studies/nvfp4-cost.md) |
| Fixed-output agent traces do not establish task success | [Agent workloads](studies/agent-workloads.md), [quality](studies/greedy-quality.md) |
| Train the drafter on recorded serving features; target prefill numerics are shape-dependent | [Draft training](studies/gb10-draft-training.md) |
| A 15-row prompt-lookup branch beside the draft chain does not raise served throughput; per-round union oracles over-value copy-region tokens | [Lookup branch](studies/lookup-branch.md) |
| Judge arithmetic changes on the family gate's round-cost interval; acceptance is unresolvable below about 7% on 18 families; bit-identical outputs are not a gate | [Family gate](studies/family-gate.md) |
| The 101-family set resolves acceptance to about ±3%; 1 to 2% claims still need hundreds of real traces | [Family set v3](studies/family-set-v3.md) |
| At batch one the GPU is 98% busy and every large GEMM streams at 84 to 97% of the ceiling; kernel headroom is about 5% at batch one, 11% at batch four | [Round profile](studies/gb10-round-profile.md) |
| 3.25-bit decode copies give 1.17x at batch one with unchanged acceptance and no detectable quality loss; batch four needs a tensor-core CUDA kernel | [3-bit decode](studies/w3-decode.md) |
| A tensor-core CUDA kernel streams the 3-bit copies at the memory ceiling for 16 to 64 rows: round cost x0.66 at batch one and x0.76 at batch four with the head and draft included | [3-bit CUDA kernel](studies/w3-cuda.md) |
| GSM8K and HumanEval accuracy is unchanged by the relaxed default and by 3-bit decode weights (paired, n=300 and 164) | [Accuracy](studies/accuracy-benchmarks.md) |
| Relaxed acceptance is an opt-in speed mode (+10% at batch one for 4% non-argmax tokens), never a default without a judged quality comparison | [Relaxed acceptance](studies/relaxed-acceptance.md) |

The exact pre-cleanup tree is tagged `archive/pre-cleanup-2026-09-11`.
Create a separate checkout with `git worktree add --detach /tmp/sharar-history
archive/pre-cleanup-2026-09-11` when an old file layout is needed. Individual
experiments may require an earlier revision recorded in their report.
The duplicate-save prefix study uses `ea6a9de`; the retired command on current
main deliberately refuses its now-identical capacity comparison.

## Previous guides

- [Roadmap before cleanup](roadmap-2026-09-11.md)
- [Benchmark commands before cleanup](benchmark-guide-2026-09-11.md)
- [Engine next steps after the BS4 study, 2026-09-10](gb10-engine-next-2026-09-10.md)
- [GB10 experiment plan and criteria, September 2026](gb10-experiments-2026-09.md) (the evergreen criteria moved to CONTRIBUTING)
- [September 11 maintenance cleanup](maintenance-2026-09-11.md)

## Study index

- [Batched accepted-feature feedback](studies/accepted-batch.md)
- [Long-session cache work on GB10](studies/agent-context.md)
- [Agent workloads and sustained generation](studies/agent-workloads.md)
- [Atlas comparison and target-verifier experiment](studies/atlas-verifier-study.md)
- [Batch-4 verify costs, small-M GEMM policy, and the ShareGPT comparison](studies/batch4-and-sharegpt.md)
- [Wider DFlash2 blocks and the unified-memory cold tier](studies/cold-tier.md)
- [Sharar versus SGLang on the same machine, model and harness](studies/comparison.md)
- [Fresh GB10 comparison: Sharar and SGLang](studies/competitive.md)
- [DFlash2 becomes the default, three serving fixes, and the prefill kernel study](studies/default-dflash2.md)
- [**Sharar on DGX Spark: what reaching 100 tokens/s requires**](studies/dgx-spark-100tps-review.md)
- [Direct accepted-state reconstruction](studies/direct-state.md)
- [DFlash2 selection on the GPU](studies/draft-selection.md)
- [Development environment](studies/env.md)
- [GB10 audit for 10,000 input / 1,000 output tokens](studies/gb10-10k1k-audit.md)
- [Plan: 1.5x cold 10K/1K TPS on GB10](studies/gb10-1p5x-plan.md)
- [GB10 acceptance parity investigation](studies/gb10-acceptance-parity.md)
- [GB10 agent checkpoint retention](studies/gb10-agent-retention.md)
- [GB10: AIPerf 10,000 input / 1,000 output, greedy](studies/gb10-aiperf-results.md)
- [BS1 throughput: implementation and measurements](studies/gb10-bs1-speedups.md)
- [BS4 engine study](studies/gb10-bs4-engine.md)
- [Exact-arithmetic GB10 engine speedups](studies/gb10-engine-speedups.md)
- [**GB10 FP8 profiling and next experiments — 2026-09-09**](studies/gb10-fp8-profile.md)
- [GB10 speed priorities after the acceptance investigation](studies/gb10-next-speedups.md)
- [GB10 optimization implementation and measurements](studies/gb10-optimization-results.md)
- [GB10 target arithmetic repairs and benchmark iteration](studies/gb10-target-fix.md)
- [Why the GB10 code benchmark is still slower](studies/gb10-target-path.md)
- [GB10 TPS validation and push recommendation](studies/gb10-tps.md)
- [Current 16-row GB10 engine study](studies/gb10-verifier16.md)
- [Greedy text parity and answer quality](studies/greedy-quality.md)
- [> Archived experiment: reproduce from commit `56605f0`. Its model, executor,](studies/mixed-forward.md)
- [Shared prefill and verification projections](studies/mixed-projection.md)
- [NVFP4 verification cost and narrow-tile study](studies/nvfp4-cost.md)
- [Parity findings (Phase 1)](studies/parity.md)
- [Phase 2 notes: decode loop, CUDA graphs, kernel fusions](studies/phase2.md)
- [Phase 3 notes: memory ledger and radix prefix cache](studies/phase3.md)
- [Phase 4 notes: scheduler](studies/phase4.md)
- [Phase 5 notes: HTTP API](studies/phase5.md)
- [Phase 6 notes: promotion and runtime image](studies/phase6.md)
- [Phase 7 notes: speculative decoding with the MTP head](studies/phase7.md)
- [Phase 7A: request ownership and failure handling](studies/phase7a.md)
- [Phase 7B: Extension contracts](studies/phase7b.md)
- [Phase 7C: local workload pilot and first GB10 experiment](studies/phase7c.md)
- [Phase 7D: native drafts and GB10 state retention](studies/phase7d.md)
- [One GDN prefill plan per forward](studies/prefill-plan.md)
- [Prompt-lookup n-gram branch coverage screen](studies/ngram-branch-screen.md)
- [Prompt-lookup second verification branch: serving measurement](studies/lookup-branch.md)
- [The family gate: one command, a fixed code prompt, and what it can resolve](studies/family-gate.md)
- [The 101-family set: the gate's resolution at three percent](studies/family-set-v3.md)
- [One decode round on the GB10, profiled](studies/gb10-round-profile.md)
- [Three-bit decode weights: kernel, speed and quality](studies/w3-decode.md)
- [Three-bit decode weights: the tensor-core kernel, head and draft](studies/w3-cuda.md)
- [Task accuracy: exact, relaxed default and 3-bit decode](studies/accuracy-benchmarks.md)
- [Sharar versus SGLang on the 10K/1K cell, 2026-09-16](studies/sglang-comparison-2026-09-16.md)
- [Relaxed speculative acceptance: opt-in speed mode, batch one and batch four](studies/relaxed-acceptance.md)
- [Runtime cleanup and clean-image promotion review](studies/promotion.md)
- [Request cache policy for agent sessions](studies/request-cache-policy.md)
- [Ten-second restarts, a sync-free sampled path, and the tier's park cost](studies/restart-and-syncs.md)
- [Sixteen-row draft-tree coverage study](studies/tree-coverage-study.md)
- [Chat controls and the single-agent two-path study](studies/two-path-study.md)
- [Verifier and prefill numerical controls](studies/verifier-numerics.md)
- [Target prefill numerics: the shape dependence bisected](studies/target-numerics.md)
- [Controlled DFlash2 costs and compact convolution screen](studies/verify-cost.md)

JSON manifests stay beside their companion reports. Raw artifacts remain under
`$SHARAR_RESULTS`; their paths and content hashes are unchanged. Moving a report
does not change its historical qualification or deploy a production image.
