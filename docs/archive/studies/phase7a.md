# Phase 7A: request ownership and failure handling

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

Review baseline: `21019b01780df17658ade146c85da0757ac6df0d`.
The repair changes are in the working tree. They cover A1–A8; proposer
interfaces, draft-aware caching and competitive tuning remain later phases.

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

The repair also reserves enough KV pages for both speculative and ordinary
rows before either advances. Otherwise a mixed batch could complete its
speculative rows, fail allocation for the ordinary rows, and retry work that
already produced output. Uncommitted speculative reservations are attached
to the request so failure cleanup can recover them.

## Invariants to preserve in the next refactor

- Immediately after a generation step, `committed_len == context_len - 1`.
  The last emitted token is the pending input for the next step. During
  chunked prefill/replay, `committed_len == prefill_pos`.
- Request-owned pages, cache-owned pages and free pages cannot overlap, except
  that a request explicitly borrows its locked cached prefix. Dummy page/slot
  zero never enters a free list.
- The token prefix, full-attention KV and recurrent snapshot in a cache entry
  describe the same processed inputs. Draft KV validity remains a separate
  Phase 7D requirement; this change does not introduce a draft-aware cache.
- Output history and RNG progress survive retraction. Previously delivered
  tokens cannot be redelivered, and replay consumes no new random draws until
  it reaches the next ungenerated token.
- Every accepted server ticket receives a terminal result. Fatal engine
  failure closes admission before draining the command queue.

`spec_accepted` still measures matching draft candidates before stopping;
`spec_emitted` measures useful output after stopping. `committed_len` counts
processed input tokens, `output_ids` counts generated tokens, and
`stream_cursor` counts tokens handed to the HTTP side. That cursor is not an
acknowledgment that a remote client received the bytes. String-stop filtering
remains in the detokenizer and can make visible text shorter than generated
token usage.

Replaying the same history through prefill can round differently from token
decode on this quantized hybrid model. Seed isolation guarantees independent
random draws for identical logits; it does not promise identical outputs
across different GEMM shapes. Cache insertion also retains the first valid
snapshot for an already cached prefix. The tests separate exact ownership
checks from numerical continuation checks.

## Proof and reproduction

Maintained tests:

- `tests/unit/test_lifecycle.py`: terminal cleanup, idle cancellation, every
  acceptance length 0–3 and stop position 1–4, and active cache flush.
- `tests/unit/test_runtime_safety.py`: real executor/scheduler with a small
  exact CPU recurrence, forced pool pressure, streamed replay, fixed-logit RNG
  isolation, request/profile validation, reservation cleanup, engine failure
  injection and HTTP/SSE error behavior.
- `tests/gpu/test_lifecycle.py`: opt-in real-checkpoint checks across eager
  and CUDA graphs, batches 1/2/3/4, cache disabled/miss/hit, mixed stop
  positions, cached continuation, active flush, retraction, the exact 32K
  boundary and frozen semantic canaries.

Direct before/after reproduction: the same seven selected behavioral
regressions produce **7 failed on the original commit, 7 passed after repair**.
These failures occur at the behavioral assertions, rather than at imports or
test setup. Logs and exact container commands are under
`${SHARAR_RESULTS}/reviews/2026-09-04/repairs/` (`before.log`, `after.log`,
`before-after.json`).

Final CPU suite: **103 passed, 9 skipped** (snapshot/GPU unavailable in the
isolated CPU invocation). GB10 suite: **194 passed**, comprising 110 unit/kernel
tests and **84 real-model lifecycle cases**, in 250.71 seconds including model
load, graph capture and warmup. The final GPU run used at least **73.99 GiB
MemAvailable**. Two CPU pressure tests were added after the GPU run collected
its tests and passed in the final CPU invocation.

The 84 GPU cases comprise 48 speculative stop/commit/resume configurations,
8 active shared-prefix flush configurations, 24 streamed-retraction
configurations, 2 scheduler/32K-boundary configurations and 2 frozen-canary
configurations (all four canaries per execution mode). Every case checks
full pool recovery and uniqueness after release/flush. GPU logs, JUnit XML,
exact command and memory samples are in `gpu-tests.log`, `gpu-tests.xml`,
`gpu-command.json` and `memory-guard.jsonl` in the evidence directory.

An intermediate test run exposed a test-design issue: continuing one request
inserted another request's prefix before the latter's ownership assertion.
The cache correctly retained the existing valid snapshot, whose rounding
could differ. Ownership checks now precede those continuation inserts; exact
snapshot assertions and cold/warm continuation checks remain intact. That
earlier log is retained as `gpu-test-design-failure.log`.

Run CPU checks in the pinned runtime environment:

```bash
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/unit -q
```

Run the GPU suite with the pinned snapshot and frozen parity HTTP fixtures
available under `SHARAR_HF_CACHE` and `SHARAR_RESULTS`. Arrange sufficient
exclusive memory before loading another model, and run the host memory guard
against the **test container's name**:

```bash
SHARAR_RUN_GPU=1 python3 -m pytest tests/unit tests/gpu/test_lifecycle.py -q
```

The tests use 35,000 KV pages, 12 state slots and MTP depth 3; they do not
size pools to consume the host's remaining memory. The first attempt beside
the serving instance nevertheless reached the host floor during model setup.
Both the test guard and the existing serving guard stopped their respective
containers. The subsequent validation runs use the GPU with serving stopped;
the original `sharar-qwen38` container was then restarted, and its
`sharar-memory-guard.service` is active. `tools/verify-sharar.sh` passed health,
pinned model/profile identity and the exact `SPARKLANG_OK` generation canary.
The original image remains `sha256:b893cdc2e3171bc2b977949305535e1cc62ebb41cee7d607638d628588d50b1b`;
these source repairs have not been built into or promoted to that image.

These are correctness results. They establish no speedup over SGLang, DSpark
or DFlash2. Keep the historical performance figures labeled as historical
until the controlled Phase 7C comparison is rerun.
