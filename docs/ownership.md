# Request ownership and failure handling

These invariants remain required for changes to scheduling, speculation,
caching and HTTP lifecycle. The original repair status and source are
[archived](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/ownership.md).

| Repair | Result |
|---|---|
| A1 — termination | First-token EOS/length requests reach release. Queued cancellation becomes terminal and is published even when the scheduler becomes idle. Completion, cancellation and release are idempotent. |
| A2 — speculative commit | Truncate accepted output at the budget/EOS before selecting the GDN/conv snapshot and retaining KV pages. `committed_len` explicitly identifies the prefix represented by state and KV; cache insertion rejects a mismatched label. |
| A3 — flush | Hide invalidated entries immediately. Retain pinned pages/snapshots until their final reader releases them; bypass insertion while the flush drains. `/flush_cache` acknowledges invalidation, and `flush_pending` reports retained readers. |
| A4 — retraction | Replay prompt plus already emitted output on readmission. Preserve output IDs, stream cursor, original first-token time, output budget and RNG state. Replay samples only the next new token. |
| A5 — sampling | Each seeded request owns its generator. Identical seeds no longer share an evolving generator through a global seed cache. |
| A6 — accounting | Scheduler decode counts use actual output deltas. Speculative emitted counts exclude candidates discarded by stopping; per-request tokens/round excludes the first prefill token. |
| A7 — limits | Reject invalid IDs, empty prompts, nonpositive/noninteger budgets, invalid sampling values and prompt+budget overflow before admission. Bound batch/depth configuration. Derive graph sizes/context buffers from executor limits, including speculative padding, and check KV lengths before replay planning. |
| A8 — failures | Fail outstanding tickets/futures, return HTTP 503 or an SSE error, make health/readiness fail, and reject new work. Release owned resources where safe; failed forwards cannot publish cache entries. |


`StateManager` owns target resource transfers; `EngineLoop` owns scheduler
mutation and request tickets on its thread. HTTP input validation lives in
`api/validation.py`, while endpoint response adapters live in `api/server.py`.
The final ticket is retired before completion is notified to the HTTP loop.

Retained draft features belong to their exact recurrent slots. Freeing or
recycling a slot invalidates its feature handle. Prompt-end snapshots remain
immutable when an active request continues decoding. Cold-tier import binds
features to the restored target slot. See [interfaces](interfaces.md).

CPU lifecycle/runtime-safety tests cover cancellation, failure, flush and
retraction; checkpoint-feature tests cover auxiliary ownership. GPU lifecycle,
draft-lifecycle and cold-tier tests validate the actual model/state path.
Exact state ownership and cross-shape numerical continuation are distinct
checks; floating-point partition changes can change greedy output IDs.
