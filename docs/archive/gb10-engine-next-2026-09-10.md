# Next engine work for BS1 and BS4

2026-09-10. Target and draft weights remain fixed; training and new numerical
model profiles are deferred. The [BS4 study](studies/gb10-bs4-engine.md) completes the
previous width and scheduling task. **It does not establish a large cold
throughput gain, and no BS4 serving change is promoted.**

Keep the current FP8 DFlash2 k=15 source baseline and the exact BS1 QKV/MLP
integration. The restored production container still uses the BF16 draft;
source validation and production deployment are separate.

The subsequent [agent checkpoint-retention study](studies/gb10-agent-retention.md)
implements coupled draft-feature and recurrent-slot lifetime with explicit
memory accounting. It evaluates warm-session recovery separately from cold
decode, and keeps its short-resume scheduler experiment in `bench/`.

## What the BS4 study decided

- Keep 16 verifier rows per request. Across five groups covering 18 families,
  pooled decode is 132.70 TPS; width 8 gives 129.25 TPS and width 4 gives
  104.80 TPS. Full draft generation remains k=15 in this isolated screen.
- Keep the 2,048-token prefill chunk, 4,096-token budget and one decode step.
  An 8,192-token budget and four decode steps fail the HTTP throughput gate.
- Keep runtime-count normalization/RoPE as an **initialization experiment**
  in the frozen `source-candidate/`. It removes repeated compilation and
  preserves all 27 fixed native outputs, but pooled warm HTTP estimates are
  3.8% lower for finite BS4 and 1.1% lower for the replenished trace. Admission
  and continuations differ; a steady serving benefit is not established.
  Production kernels are unchanged.
- Full-BS4 target verification accounts for about 110 ms of GPU kernel time,
  versus 14 ms for drafting and 7 ms for acceptance plus feedback. Fourfold
  narrower verification saves only about 16 ms of target work and loses too
  many emitted tokens. These are diagnostics, not additive latency forecasts.

See the report for source hashes, complete outputs, work accounting,
independent HTTP loads, natural-stop checks and the limits of the measurements.

## Next bounded work

1. **For another cold-BS4 candidate, tighten the scheduling experiment first.**
   Use deterministic admission and prefill membership alongside ordinary
   free-running HTTP. Separate four already-prefilled decodes, a finite
   four-request cohort, and a shared replenished backlog. Record actual
   available work, prefill interruptions, physical graph rows, emitted tokens,
   total latency and TTFT. The completed fixed-client-lane trace includes its
   drain and workload imbalance; its underfilled rounds are not all recoverable
   scheduler waste. This is a measurement prerequisite, not a promised gain.
2. **For warm agent sessions, build on checkpoint retention.** The subsequent
   study binds draft features to recurrent slots, including eviction, flush
   and cold-tier restore. Follow [the cache plan](studies/request-cache-policy.md)
   for request modes, namespaces and expiring retention policy. The historical
   duplicate-save repair's 11.40 → 42.38 TPS rescue and recovery on a larger
   working set are related cache-fallback repairs, not additive cold gains.
3. **Keep BS1 host overlap a small project.** Device-side stop-aware acceptance,
   pinned asynchronous download and queued reconstruction can target the
   measured 0.7–1.3 ms gap. That is about 1% round-time headroom, not a large
   cold-BS1 optimization. Preserve recurrent commits, private KV tails,
   graph-buffer lifetimes and proposer feedback.

## Changes not justified by this screen

No adaptive width controller, extra BS3 graph, or true shorter-draft execution
is promoted. A controller must beat the best fixed width using only information
available before the next round; a hindsight width oracle is not a policy.
Further physical-shape changes can change numerical continuations even with
identical weights, and need their own quality and throughput qualification.

The [mixed-forward prototype](studies/mixed-forward.md) already failed arrival TTFT
and protected quality. Sharing projections is not a new untested shortcut;
revisit it only with a concrete change addressing those failures. The earlier
numerical FP8 replacement saves about 2 ms in the real matrix sequence and
fails its family throughput gate. Repeating an unchanged failed kernel sweep
or adding separate microbenchmark savings is not a demonstrated 1.5x path.

Before a larger redesign, require at least 5% aggregate improvement with a
paired family interval above 1, no material BS1 regression, and explicit TTFT
and streaming-latency protection. Aim for 10% or more to justify substantial
new complexity; this is a decision gate, not a forecast. Preserve outputs
where arithmetic and partitioning are unchanged. Large cold BS1/BS4 gains
remain an open research problem under the fixed-weight constraint.
