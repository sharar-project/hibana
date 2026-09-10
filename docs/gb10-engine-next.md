# Next engine work for BS1 and BS4

2026-09-10. User priority: engine throughput at BS1 and BS4, with target and
draft weights fixed. Draft training and new numerical model profiles are
deferred. The [completed BS1 study](gb10-bs1-speedups.md) is the new baseline.

**Start with a BS4 verification-width and scheduling study.** This is a
bounded search for a larger gain, not a demonstrated 1.5x path. The measured
BS1 host gap supports about 1% round-time headroom; spending that budget
cannot deliver a large cold BS1 gain.

## First task: account for useful work at both batch sizes

Keep FP8 DFlash2, the current target and the exact QKV/MLP integration fixed.
Measure three cells separately: cold BS1; four already-prefilled requests
decoding together; and HTTP concurrency four with both finite batches and
a replenished queue. Client concurrency is not proof that four requests
participate in every decode forward.

Record actual running/prefilling batch sizes, physical graph rows, accepted
and emitted tokens, rejected/padded rows, prefill interruptions, draft
fallbacks, round durations and total request latency. Use the existing
family tasks plus frozen mixed-length arrival traces. Include natural-stop
checks; the standard code prompt's post-EOS continuation is insufficient.

Test target verification widths 4, 8 and 16 at BS1 and BS4. These correspond
to k=3/7/15 and, at full BS4, 16/32/64 target rows. First keep the same
15-proposal draft generation and truncate only target verification. This
isolates the verifier tradeoff; it does **not** save draft-generation cost.
`SpecDecoder.verify_depths`, `verify_depth` and `width_graphs` already provide
the experimental mechanism. Production currently captures only its configured
depth. Separately evaluate true shorter draft execution if the first screen
justifies it; changing the draft's physical shape can alter its proposals.

Optimize **emitted tokens / complete round time**, not acceptance percentage
or row utilization alone. Rejected rows are worthwhile when their occasional
acceptance costs less than another full target traversal. BS3 currently pads
to the BS4 graph, but exact-size graph capture also needs a measured benefit
before adding another shape or general ragged execution.

## Implement only what the study supports

1. **BS4 width selection and graph dispatch.** Compare each policy with the
   best fixed width at the same precision. If there is headroom, select width
   from active batch size and past acceptance, using hysteresis. Preserve
   private KV tails, stop-truncated recurrent commits and draft feedback.
   A hindsight best-width result is only an upper bound, never a serving
   policy. Earlier confidence policies failed their screens; do not repeat
   them without evidence from the current cost/acceptance measurements.
2. **BS4 prefill/decode scheduling.** The scheduler can execute two separate
   prefill forwards before its decode step. Measure the cost and compare
   token budgets and admission order. Avoid starving either newly arriving
   prompts or active decodes. If separate forwards account for enough lost
   throughput, prototype sharing projection work across prefill and verify
   rows; mixed attention/GDN state handling and graph behavior are a larger
   change. Batch-shape changes require fresh numerical and quality checks.
   Decode-prioritized mixed batching is an established mechanism in
   [vLLM's chunked-prefill design](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill);
   its gains and configuration values do not transfer automatically to GB10
   or Sharar's speculative recurrent target.
3. **BS1 exact acceptance/commit overlap.** Device-side acceptance and
   stop-truncated counts, pinned asynchronous download, an event, and queued
   reconstruction can overlap the measured 0.7–1.3 ms host gap. This is a
   small follow-up. Extra streams do not by themselves remove the much
   larger target-weight reads.

Use paired independent loads and report both aggregate decode TPS and
end-to-end throughput, with TTFT and p95 latency. As a proposed advancement
gate, require at least 5% aggregate improvement with a family interval above
1 and no material BS1 regression before expanding the BS4 redesign. Aim for
10% or more to justify substantial new engine complexity; that is a target,
not a forecast. Preserve exact outputs where arithmetic/partitioning is
unchanged; separately qualify changed-shape continuations.

## What the existing evidence does and does not support

- The current exact integration improves BS1 HTTP decode 1.13–1.27%.
  Its M=16 dispatch is not an M=64 BS4 optimization.
- The real 176-matrix FP8 sequence saves about 2 ms with the numerical
  replacement, and its family throughput gate fails. Another generic tile
  sweep is not a demonstrated large-gain project.
- The older [width experiment](gb10-optimization-results.md) combines width,
  precision and separate measurement windows. Its 8.67% four-client code
  reduction is against an older BF16 sixteen-row control, not the current
  FP8 baseline. Re-measure the isolated width choice before crediting a gain.
- Without fewer target traversals, fewer weight bytes, or a newly measured
  major inefficiency, there is no demonstrated large cold BS1 engine gain.
  Prior tree, selector and target-precision failures remain relevant; no
  unchanged failed approach is promoted here.

For **cached agent sessions**, coordinated retention of target KV, recurrent
state and draft features is a separate engine opportunity affecting both
batch sizes. The duplicate-save repair is already on main. Broader joint
eviction, budget accounting and fallback to an earlier complete checkpoint
remain open in [the cache plan](request-cache-policy.md). The historical
11.40 → 42.38 TPS repair prevented fallback on a cached trace; it is neither
a new remaining 3.7x gain nor a cold BS1/BS4 throughput result.
