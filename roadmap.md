# Sharar roadmap

Updated 2026-09-11. This is the current work order. Historical milestones and
their original gates are in the [archived roadmap](docs/archive/roadmap-2026-09-11.md).

## Supported baseline

One GB10, the pinned Qwen3.8-27B NVFP4 target, 32K context, maximum batch four.
DFlash2 defaults to 15 proposals and accepted-state reconstruction. The launcher
defaults to BF16; FP8 is the optimized measured draft profile. Keep target and
draft weights fixed for the current engine work. Training is deferred.
See [operations](docs/operations.md) and [performance](docs/performance.md).

## Current state

- Exact BS1 QKV/MLP integration is retained; it improves family decode 1.43%.
- Narrower BS4 verification and tested scheduler policies failed their gates.
  Keep k=15, 2,048-token prefill chunks and a 4,096-token prefill budget.
- Draft features now follow target recurrent-slot ownership. Resumed throughput
  improves on the qualified traces; cold throughput is essentially unchanged.
- Shared GDN prefill planning, accepted-state reconstruction, prefix caching,
  managed-memory cold storage and the host memory guard are implemented.
- Chat accepts auto/null/none tool choice; forced choices and constrained JSON
  output remain unsupported. Tool execution is the client's responsibility.
- Documentation, benchmark support, distribution contents and stale contracts
  were cleaned up. The pre-cleanup tree is tagged `archive/pre-cleanup-2026-09-11`.

## Draft adaptation track

Started 2026-09-11 on `feat/draft-traces`; see [the design](docs/draft-adaptation.md).
The served target, head and verification stay fixed; only the DFlash2 drafter
changes, so outputs remain the target's own and the gates are speed, TTFT and
memory. Done in this branch: (1) opt-in request traces (`SHARAR_TRACES`,
per-request `trace: false`, `sharar traces`); (2) SpecForge studied and used as
the numerical reference rather than as the pipeline, since its Qwen3.8 recipe
needs patched SGLang capture servers; (3) `sharar train`: DFlash2 training in
Sharar against the NVFP4 target with held-out tokens-per-round evaluation and
`best/` promotion, checked against the serving proposer, the publisher's
reference model and SpecForge's objective by `bench/dflash2_reference.py`.
Next: (4) in-place draft reload in the engine and the 18-family collapse check;
(5) user-started training with an engine-busy pause and multi-machine
transport. Training never runs beside serving on the same GPU.

## Next work, in order

1. **Restore CI availability.** GitHub refused the previous run before any job
   steps because account billing/spending limits blocked execution. Preserve
   the CPU, distribution and documentation checks; resolve the account issue.
2. **Protect long-lived sessions under memory pressure.** Add bounded serving
   admission/backpressure and report live/reserved/non-PyTorch memory before
   relying on optional cache growth. The external guard remains required.
   `Ledger.check_floor()` is not yet integrated into the serving loop.
3. **Measure the next engine candidate under controlled work.** Follow
   [engine-next](docs/gb10-engine-next.md): separate cold BS1, already-prefilled
   BS4, finite cohorts and a replenished backlog. Compare complete output work,
   TTFT and streaming gaps. Large cold gains remain unproven; tiny kernel wins
   or changed continuations are not additive speed forecasts.
4. **Extend request cache policy.** Implement scoped reuse/store modes and
   expiring retention hints from the [proposal](docs/request-cache-policy.md),
   then measure target-prefix misses under independent growing sessions.
   Retained draft features cannot recover an evicted target snapshot.
5. **Overlap restoration and reclaim optional capacity.** Replace cold-restore
   blocking synchronization with event-owned lifetimes; consider graph-safe
   reclaimable slabs. Returning IDs to monolithic pools does not release DRAM.
   These need separate correctness, pressure and end-to-end performance gates.
6. **Complete output-control and quality qualification.** Add request-owned
   constraint state and speculative rollback before supporting forced tool
   choice or structured output. Evaluate actual tasks/tool execution and CPU
   contention; fixed-budget simulated conversations do not establish task success.
7. **Finish release/startup work after profiles stabilize.** Prepared-weight
   caching, versioned AOT kernels and a smaller compiler-free image remain
   separate projects. Current SM121 execution still uses JIT compilation.

## Contribution gates

- Preserve [request ownership](docs/phase7a.md), private KV tails, recurrent
  commits, graph-buffer lifetimes and stop-truncated draft feedback.
- Identify source, image, precision, input/output work and memory geometry for
  every performance claim. Retain failed runs and exact comparison limits.
- Run CPU contracts for control changes; use affected GPU/numerical tests for
  execution changes, and paired full-workload runs for performance changes.
- Measure changed paths before optimizing them. Use the
  [experiment criteria](docs/gb10-experiments.md); a larger redesign should
  establish at least 5% aggregate improvement with protected BS1 and latency,
  and target 10% or more to justify substantial complexity.
- Keep raw dumps outside Git and compact evidence linked from dated reports.
  Keep licensing/attribution and numerical reference implementations intact.
- Use coherent commits with validation evidence and no attribution trailers.
  Production image changes require completed promotion evidence. Validation
  windows restore the original service and memory guard.

Do not repeat unchanged rejected selectors, numerical kernel replacements,
mixed-forward policies or scheduler experiments. Consult the
[decision index](docs/archive/README.md) before starting another screen.
