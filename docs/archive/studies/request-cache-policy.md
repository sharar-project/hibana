# Request cache policy for agent sessions

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

The subsequent [checkpoint-retention study](gb10-agent-retention.md) implements
joint feature/recurrent-slot lifetime and prospective capacity accounting.
The measurements below document the earlier duplicate-save repair; they are
not additional gains from the new implementation. Request-level cache modes,
namespaces and retention priorities remain proposals.

The cache experiment reproduces a large cached-request performance loss and
fixes duplicate draft checkpoint saves without increasing the cache budget.
Request-level API fields below remain a design proposal, not implemented
or advertised as supported parameters. The cold 1.5×–2× TPS goal remains open.

Sharar already caches attention KV together with GDN convolution/FP32 SSM
snapshots. GDN is this target's recurrent component; "Mamba cache" describes
the analogous state-cache problem. A token match can resume only at a retained
recurrent snapshot, and fast DFlash2 continuation also needs compatible draft
features at that boundary.

## Baseline behavior and the original performance gap

`RadixCache` is a single token-keyed tree. It evicts unlocked leaves by LRU;
when only recurrent slots are exhausted, it first discards unlocked interior
snapshots while retaining KV. Active cache loans lock their ancestor path.
Finished requests publish snapshots; an unfinished request's newly computed
prefix is not yet reusable by later admissions.

The target cache and draft feature cache have different lifetimes. With the
measured 24-slot/maximum-batch-four configuration, target snapshots have 23
usable slots, but projected draft features have only four 2,048-token windows
(80 MiB). Radix nodes contain handles into that separately evicted LRU. If the
handle is gone, `ParallelProposer.ingest_prefill()` marks the request invalid
for speculation. Attention KV and recurrent state still count as a prefix
hit. The baseline feature cache also receives repeated saves at the same
position; the accompanying fix reuses a live handle for that request's same
immutable checkpoint. Evicted handles are saved again, and different requests
never share features merely because their token positions are equal.

The subsequent implementation binds draft features to retained recurrent
slots, so snapshot deletion also releases its feature window. Merely adding a request
priority to radix nodes would not protect the draft payload. An experimental
larger feature budget diagnoses this mismatch; it is not itself a complete
ownership or priority implementation.

## Measured cached 10K/1K result

The fixed trace is a code prompt with 10,000 inputs and one output, a design
prompt with 10,000 inputs and one output, then the identical code prompt with
1,000 outputs. It uses the same checkpoint, FP8 DFlash2 draft, depth 15,
reconstruction, 60,000 KV entries and 24 recurrent slots as the current study.
All compared repeated code requests retain 9,984 input tokens.

| Cache behavior | Draft-feature cap | Decode TPS | Complete 1K-output latency |
|---|---:|---:|---:|
| Baseline, draft features evicted | 80 MiB | 11.397 | 87.763 s |
| Experimental larger feature cache | 460 MiB | 42.363 | 23.692 s |
| Duplicate-save fix | 80 MiB | 42.379 | 23.682 s |

Values are two-run means. Baseline/larger-budget samples use ABBA in one
load; the fix uses a fresh load, an excluded full immediate-hit control, then
two trials with independently seeded caches. Each load's immediate-hit control
is excluded from these timings. Startup and model loading are also excluded.

The baseline falls back to ordinary decoding: zero speculative rounds. The
other two paths retain speculation for all 220 rounds. Every retained-budget
and fixed run reproduces both immediate-hit controls' complete output IDs and
per-step emitted work. Baseline repeats agree with each other, but diverge
from the speculative control at output index 11. The roughly 3.71× completion
speedup therefore repairs a cache-induced fallback; it is not an
identical-output comparison against that fallback or a cold decode speedup.
TTFT stays near 0.105–0.110 s in the ABBA comparison.

The initial 32-output pilot also checks zero, one and two unrelated requests.
Before the fix, even one causes the feature miss despite intact KV/GDN state;
increasing the budget preserves speculation in both interleaving cases. The
first immediate-hit pilot pays a cold shape cost and is not used as a latency
baseline. The fixed 80 MiB implementation is confirmed for **one** intervening
request. A larger working set can still evict useful features: deduplication
does not solve coordinated retention, session policy or memory-pressure churn.

## Proposed request contract

The same typed object should work on `/generate`, `/v1/completions`, and
`/v1/chat/completions`, with the OpenAI SDK forwarding it through `extra_body`:

```json
{
  "cache_control": {
    "mode": "read_write",
    "namespace": "repo:sharar:revision-42",
    "session_id": "agent-17",
    "retention_priority": 80,
    "priority_duration_s": 600,
    "state_retention": "latest"
  }
}
```

These are proposed semantics, not current parsing behavior:

| Field | Meaning |
|---|---|
| `mode` | `read_write` reuses and publishes; `read_only` reuses without adding cache entries or checkpoint payloads; `bypass` neither reads nor publishes. Bypass does not flush other requests' caches. |
| `namespace` | Partition cache matching. Identical token prefixes may share only within the same namespace and model/arithmetic identity. It is separate from a session ID, so agents can share a repository prefix. Caller-chosen strings alone are not an authorization boundary. |
| `session_id` | Group a linear rollout's retention hints without overriding exact token matching. It never substitutes for the actual prompt or selects arbitrary mutable state. |
| `retention_priority` | Bounded preference for keeping an unused entry, independent of request scheduling priority. It is a preference under the memory budget, not a pin or admission guarantee. |
| `priority_duration_s` | Time for the elevated preference, after which normal eviction applies. This is not a data-expiration or deletion promise. Shared prefixes must track expiring contributions without accumulating permanent maximum priority. |
| `state_retention` | `latest` prefers the newest complete resume point for a linear session; `branches` retains useful fork points. Both preserve active readers and follow server memory limits. The existing default remains unchanged when omitted. |

One server eviction policy combines these hints with recency, reuse and
recomputation cost. Requests should not independently select conflicting LRU,
LFU or SLRU algorithms for a shared pool. Under state-only pressure, compare
the cost of losing a recurrent/draft resume point, not just its KV byte count.
Retain useful repository/tool-prefix branch points before redundant old
linear-turn checkpoints. A `latest` policy needs retry semantics: an identical
prompt still needs a checkpoint before its last token to regenerate logits.

Caller-defined checkpoint boundaries can follow in a second phase. They must
be validated against prompt length and the recurrent kernel's supported
boundaries. Report the effective boundary rather than pretending an arbitrary
token offset has a snapshot. Checkpoint policy changes can alter floating-point
execution partitioning, so full continuation/quality validation is required.

Pool sizes, recurrent-state dtype, physical cache layout and CUDA graph
strategy remain engine configuration. Changing those per request would need
separate compatible pools and numerical profiles. The request controls express
reuse and retention intent within the server's configured implementation.

## Ownership and accounting

- Admission and scheduler capacity checks use the same request policy. A
  bypassed cache must not make an otherwise oversized admission appear to fit.
- A reusable bundle ties target KV, recurrent snapshot and optional proposer
  features to the same prefix, model and draft identity. Losing optional
  features must be visible and trigger a measured fallback policy.
- Snapshot deletion, duplicate insertion, radix splitting, cancellation,
  retraction, flush and cold-tier restoration must preserve ownership and
  release auxiliary payloads exactly once. Session hints cannot evict an
  active branch or overwrite its state.
- Include draft features in the prospective unified-memory ledger. Bound
  pending copies and host/managed allocations; a cache budget is not free
  memory simply because CPU and GPU share DRAM.
- Preserve namespace, retention and compatibility metadata through the cold
  tier. Use CUDA events and deferred reuse to remove host synchronization
  only after validating stream and allocation lifetimes.

## Measurements and order of work

Keep the cold 10K-input/1K-output benchmark frozen. Add matched cache traces:
immediate repeated prompt; repeat after unrelated requests; append-only agent
turns; two branches sharing a repository prefix; and churn that exhausts
recurrent slots, then KV pages, then the managed-memory tier independently.

Report TTFT, complete 1K-output latency, sustained decode TPS, usable cached
tokens, raw matching tokens, retained state/feature bytes, draft fallback
count, eviction reason, and p50/p95 under repeated interleaving. Cached TPS
must not be presented as a cold-kernel speedup. Different execution paths
must retain their actual output IDs and receive quality checks.

1. Reproduce and repair the target/draft retention mismatch with bounded
   ownership and ledger accounting. Duplicate-save removal and the subsequent
   slot-bound retention implementation are documented separately. An earlier complete
   checkpoint remains a possible recovery policy when compatible draft state
   is unavailable for reasons other than independent feature eviction.
2. Add reuse/store modes and namespaces with lifecycle and endpoint tests.
3. Add expiring retention priority and session checkpoint policy; compare
   against unchanged LRU on held-out agent traces.
4. Evaluate publication of immutable prefixes while a request is still
   running, and asynchronous cold-tier restoration. These need separate
   cancellation, reader-lock and concurrency validation.

Raw protocols, commands, model/image identities, output IDs, samples and
source snapshots live under
`$SHARAR_RESULTS/reviews/2026-09-09-prefix-policy`. The larger budget is
benchmark-only. The runtime fix changes checkpoint retention when built;
the original production image is restored after each experiment.
The [compact evidence manifest](request-cache-policy-results.json) points to
the sealed reports and artifact hashes. CPU validation passes **403 tests
with eight skips**. The affected GPU suite passes **22 tests**, covering
BF16/FP8 drafts, batches one through four, eager/graph execution, mixed
sampling and long batched prefill. Minimum recorded host `MemAvailable` is
74.44 GiB; final health and memory-guard checks pass.

## SGLang comparison

SGLang exposes global radix eviction and Mamba state-cache strategies in its
[server arguments](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/server_arguments.mdx).
Its [unified-cache component contract](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/unified_cache/components/README.md)
separates full-attention and recurrent resource accounting, validates cache
hits across components, and defines component-specific eviction and locks.
That is useful architectural precedent; it does not establish that every
strategy supports Sharar's FP8 DFlash2 path or that a server option already
works as a per-request hint.
