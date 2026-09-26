# Request cache policy proposal

Request-level fields below are not implemented. Current cache entries own
target KV, a recurrent snapshot and compatible draft features together; see
[the ownership contract](interfaces.md) and [current performance](performance.md).
The earlier duplicate-save experiment is [archived](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/request-cache-policy.md).

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

## Implementation order

1. Add and test explicit reuse/store semantics at the request boundary.
2. Add namespaces and bounded expiring priority, preserving active cache loans.
3. Compare independent growing sessions under target-state and KV pressure;
   record completed work, target misses, feature fallbacks and streaming tails.

Follow the [roadmap](../roadmap.md) for current priorities. A retained feature
payload cannot restore a target checkpoint that has itself been evicted.
