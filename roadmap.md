# Sharar roadmap

Updated 2026-09-12. This is the current work order. Historical milestones and
their original gates are in the [archived roadmap](docs/archive/roadmap-2026-09-11.md).

## Supported baseline

One GB10, the pinned Qwen3.8-27B NVFP4 target, 32K context, maximum batch four.
DFlash2 with 15 proposals and accepted-state reconstruction; BF16 draft by
default, FP8 as the measured serving profile. The target, its vocabulary head
and the verifier are fixed. The DFlash2 drafter is trainable on recorded
traffic and serving follows the promoted checkpoint; the publisher's draft is
the fallback. See [operations](docs/operations.md), [performance](docs/performance.md)
and [draft adaptation](docs/draft-adaptation.md).

## Current state

- Engine: exact BS1 QKV/MLP integration (+1.43% family decode), k=15 with
  2,048-token prefill chunks and a 4,096-token prefill budget, checkpoint-owned
  draft features, shared GDN prefill planning, accepted-state reconstruction,
  prefix caching, managed-memory cold storage and the host memory guard.
  Narrower verification, mixed forwarding, numerical kernel replacements and
  scheduler variants failed their gates ([decision index](docs/archive/README.md)).
- Draft adaptation: opt-in traces with serving-exact features, `sharar train`
  (checked against the serving proposer, the publisher's model and SpecForge),
  trained-draft serving, `sharar-ctl` and the start/stop launcher. On a
  repository-coding workload a 1,000-request serve-then-train loop raised
  serving throughput 12.4% ([report](docs/archive/studies/gb10-draft-training.md)).
- API: chat with auto/null/none tool choice, thinking, streaming; forced tool
  choice and constrained JSON output remain unsupported.
- Known property: the target's prefill numerics depend on sequence length and
  chunking, so greedy continuations can change with batch shape, verification
  width or the drafter ([performance](docs/performance.md#limits)).

## Next work, in order

1. **Train on real traffic.** Record a few hundred requests of actual use and
   repeat the held-out and served-throughput measurements; the repository
   proxy has only about 150 distinct prompts and overfits after one session.
2. **Harden promotion.** Use a smaller learning rate when continuing from an
   adapted draft, and add a fixed general regression set to the promotion
   gate so a draft that fits the newest traffic cannot regress everything else.
   Then use the recorded on-policy rows (the target's argmax after a rejected
   proposal) as additional supervision.
3. **Reload without a restart.** An engine command that copies a promoted
   draft into the serving weights between rounds, including the FP8
   conversion, so `train stop` does not relaunch the container.
4. **Restore CI availability.** GitHub refused the previous run on account
   billing limits; keep the CPU, distribution and documentation checks.
5. **Target numerics.** Find the length-dependent kernel dispatch behind the
   prefill shape sensitivity; it likely bears on the unresolved cross-engine
   greedy parity as well.
6. **Protect long-lived sessions under memory pressure**: bounded admission
   and backpressure, live/reserved memory reporting; `Ledger.check_floor()` is
   not yet in the serving loop.
7. **Request cache policy**: scoped reuse/store modes and expiring retention
   from the [proposal](docs/request-cache-policy.md).
8. **Cold-restore overlap, output controls, release work**: event-owned
   restore lifetimes; request-owned constraint state before forced tool choice
   or structured output; prepared-weight caching and AOT kernels once profiles
   stabilize.

## Contribution gates

- Preserve [request ownership](docs/phase7a.md), private KV tails, recurrent
  commits, graph-buffer lifetimes and stop-truncated draft feedback.
- Identify source, image, precision, input/output work and memory geometry for
  every performance claim. Retain failed runs and exact comparison limits.
- Run CPU contracts for control changes, the affected GPU tests for execution
  changes, and paired full-workload runs for performance changes; an engine
  change proves no regression on the standard benchmark before merging.
- A larger redesign should establish at least 5% aggregate improvement with
  protected BS1 and latency, and target 10% or more to justify complexity.
- Drafter changes are judged on held-out tokens per round and served
  throughput on the user's own newest traffic, never on a public corpus.
- Keep raw dumps outside Git and compact evidence manifests linked from dated
  reports. Keep licensing, attribution and numerical reference paths intact.
- Coherent commits with validation evidence and no attribution trailers.
  Validation windows restore the original service and memory guard.

Do not repeat unchanged rejected selectors, numerical kernel replacements,
mixed-forward policies or scheduler experiments. Consult the
[decision index](docs/archive/README.md) before starting another screen.
