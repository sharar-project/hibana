# Completed experiments

These implementations are retained for numerical/policy regression tests and
reproduction. They are not selected by the serving engine. Their published
numbers apply to the recorded source revisions and protocols. The exact
pre-cleanup layout is tagged `archive/pre-cleanup-2026-09-11`.

| Modules | Decision / evidence |
|---|---|
| `agent_schedule` | Mixed throughput and streaming regressions; [retention study](../../docs/archive/studies/gb10-agent-retention.md) |
| `gb10_bs1_kernels`, `gb10_family_candidates`, `gb10_fp8_m16` | Exact bundle retained in runtime; other candidates unqualified; [BS1 study](../../docs/archive/studies/gb10-bs1-speedups.md) |
| `gb10_selector_fit`, `gb10_tree_screen` | Offline coverage screens; [FP8 profile](../../docs/archive/studies/gb10-fp8-profile.md) and [verifier study](../../docs/archive/studies/gb10-verifier16.md) |
| `gb10_target_fp4` | Layer screens do not qualify whole-model conversion; [optimization study](../../docs/archive/studies/gb10-optimization-results.md) |
| `nvfp4_tiles`, `nvfp4_round`, `gb10_quant_pack` | Kernel/cost experiments; [NVFP4 cost](../../docs/archive/studies/nvfp4-cost.md) and [verifier study](../../docs/archive/studies/gb10-verifier16.md) |

Current imports use `bench.archive.<module>`. Historical commands in dated
reports use their original paths; run those at the recorded revision.
The old prefix-capacity comparison is retired: `ea6a9de` preserves its
duplicate-save protocol. Current retention uses `bench.gb10_agent_retention`.
Shared infrastructure lives in [support](../support/README.md).
