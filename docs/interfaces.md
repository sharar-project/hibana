# Execution and ownership contracts

The rules every change to scheduling, speculation, caching or the HTTP lifecycle keeps. The seams are Python
structural types in `engine/contracts.py` (which imports no GPU backend), not a plugin registry.

| Boundary | Owner | Consumers |
|---|---|---|
| `ExecutionBackend` | Executor executes steps and owns graph/workspace setup | Scheduler chooses admissions, chunks, decode and retraction |
| `RequestState` / `StateManager` | Request page/slot ownership, the request's cache lock, reservations and processed prefix | Executor and shared verifier |
| `Proposer` | Draft weights/KV, readiness, pending tokens, retained features and draft graphs | Shared speculative driver |
| `ForwardResult` / `FeatureRequest` | Target computes logits inputs and only requested features | Prefill driver and verifier deliver features to the proposer |

## Targets

Model facts live in a `Target` value (`hibana/target.py`): the pinned checkpoint, the layer kinds, the attention and
recurrent head shapes, the vocabulary, the serving profile and the load budget. The executor reads `model.target`
once and sizes the pools, the ledger, the attention plans and the verify buffers from it. One process serves one
target, chosen at launch (`hibana serve TARGET`, `HIBANA_TARGET`); nothing is discovered at run time. Device
constraints are in `hardware/gb10.py` and serving limits in `engine/profile.py`. Kernels take their shapes from the
tensors and the target, but are validated only for the pinned targets.

A model whose arithmetic changes stored prefixes may expose a JSON-compatible `arithmetic_identity` value.
The executor includes it in the SSD store identity; `None` preserves the target's existing identity.

A target adds a `Target` instance (family-specific facts are optional fields, absent for the 27B), a loader in
`target.load_model` under its `Target.family`, and a model package whose forward honors `ForwardResult`; the
scheduler, state manager, radix cache, verifier and proposer contracts do not change. Every model implements
`register_state`, `register_pages` and `forward_extras`:

- `register_state` adds per-slot tensors through `StatePool.register_aux`; they are cleared, copied and stored with
  the recurrent states (Flash-Next's PLE convolution window). `register_pages` adds per-page rows, indexed like K/V
  (Flash-Next's indexer keys), through `KvPool.register_page_aux`. The 27B adds neither.
- `forward_extras()` returns the object that puts a forward's side inputs on `ForwardContext.extras`
  (`model/flashnext/extras.py`: PLE history and table rows, sparse-attention bookkeeping). The engine builds them
  through it for eager forwards, lets it own and fill the graphs' static side inputs, and asks it for the graph
  variant ("dense" or "selecting") a batch needs. A model without side inputs returns None (the 27B).

## Request state

`Request` owns logical prompt/output history, RNG, stream cursor and lifecycle. Its `RequestState` base groups
resource handles and `committed_len`. Only `StateManager` transfers request page/slot ownership or advances this
processed prefix. Kernels may write supplied views on the executor's CUDA stream. Callers use pool allocation, free
and `free_count`, never the private free lists.

After prefill or successful decode, the last emitted token is pending. Both target KV and the recurrent slot
describe exactly `prompt_ids + output_ids[:-1]`. That length is `committed_len`. A cache entry may publish only that
same prefix and its matching snapshot.

- `can_admit(prompt_ids, reserve_tokens)` takes actual prefill IDs, including replayed output history, and is pure:
  it evicts, restores and allocates nothing. It budgets the uncached suffix plus reserve (the scheduler adds the
  running requests' outstanding decode reserve) against the free and evictable KV, and a slot against the free and
  evictable slots, while a probe lock (`RadixCache._lock_node`) keeps the candidate's own match out of the evictable
  totals. The probe counts no cache hits; admission follows on the same owner thread, one request at a time.
- `prefetch(prompt_ids, reserve_tokens)`, for the queue head before `can_admit`, starts its SSD restore when the
  restore fits by the same count; False while one is reading.
- `admit` matches, takes the request's one lock (`req.lock`, a `radix.TreeLock`) at the matched node before it
  evicts anything for the request's own slot (`req.slot`) or pages, allocates them, and copies the matched recurrent
  snapshot into that slot. Allocation failure returns the slot and the lock. Prompt pages are reserved before prefill
  has processed them.
- `prefill_checkpoints` allocates the private snapshots (`req.private`) each row's forward places
  (`Executor._checkpoint_positions`: message boundary and last chunk end, then branching point, then a spare chunk
  end), round-robin over the rows; a spare one takes only a free slot or a spare snapshot's, and none takes a locked
  one. `commit_prefill` publishes the successfully processed extent; `discard_private` frees a failed forward's
  destinations on every row.
- `checkpoint(req, up_to)`, after each prefill chunk the drafter has ingested, inserts the prompt up to the deepest
  written private snapshot with every one of them, then moves the lock there (lock the new node, unlock the old one,
  nothing allocated in between) and frees the request's previous spare chunk end; `page_ids` marks the pages the
  cache took -1, and pages the tree already had are swapped for the tree's. With an SSD tier the inserted blocks, and
  the not-spare snapshots' tails, are queued for backup. `req.private` is empty between scheduler steps.
- `reserve(requests, width)` attaches one disjoint pending reservation per request. `step_pages` borrows committed
  pages plus the reservation. Callers may write through these views; they do not acquire ownership.
- `commit(req, count, ssm=..., conv=...)` copies the verify snapshot after input `count-1`, keeps exactly `count`
  pages and frees the rejected tail. Ordinary decode commits one page after a successful forward/sample.
- `rollback` frees pending pages and disables caching. It does **not** undo in-place kernel writes. Verification
  leaves the committed slot intact; after an ordinary forward failure, release or retract/replay before any retry.
  The serving failure path terminates its tickets and releases them.
- `retract` drops physical resources. Executor preserves emitted history, remaining budget, RNG and stream cursor,
  and replays that history on admission.
- `release(req, attach=None)` is idempotent. It transfers a valid prefix into the cache, its own slot as the reply
  end (`attach`, the proposer's `prepare_release`, puts its metadata in that snapshot first), or frees the request's
  suffix and slot, then ends the lock. `retract` ends the lock without inserting; what was checkpointed stays cached.
  A flush defers reclamation until the last lock ends.

When only recurrent capacity is exhausted, the radix cache takes unlocked snapshots by `get_priority` (spare ones
first, then the least recently used): an unlocked leaf goes whole, any other node keeps its KV pages and loses only its
snapshot and draft metadata (`snapshot_evictions`); this returns a slot to the allocator, not physical DRAM. A
duplicate snapshot at an occupied position keeps the existing one and frees its own slot. `StateManager.check` asserts
the lock counts, the evictable totals, one lock per request, and that every slot and page has one owner
([cache design](cache_design.md#protection-and-eviction)).

The runtime is single-threaded at this boundary and calls are ordered on its CUDA stream; callbacks must not free or
reuse buffers on another stream. This is an ownership transaction, not a CUDA exception rollback. Replace the cache
object only at startup or in tests; on an active runtime use `Executor.flush_cache()`.

## Forward features and lifetimes

`ForwardContext.gdn_plan` belongs to one immutable prefill request layout. `model/gdn_plan.py` derives sequence
starts, initial-state flags, 64-token chunk indices/offsets and checkpoint locations from CPU metadata once; all GDN
layers share the resulting tensors. Create a new context when lengths, slots, prefixes or checkpoints change; layers
must not rebuild the plan or read lengths back from CUDA.

For `T` flattened tokens and hidden size `H`, `ForwardResult['norm']` is BF16 `[T,H]`: the final Gemma RMSNorm
output consumed by the target LM head. The target needs it even when no proposer requests features.

`FeatureRequest.layers` selects **zero-based post-block residual streams**: `hidden + residual` after that block's
MLP, before the next block's input norm (or before final norm for the last block); on Flash-Next, the FP32 mean of the
hyper-connection streams leaving that block, as its DFlash2 drafter was trained. Each selected BF16 `[T,H]` tensor
has fresh storage, so later in-place residual updates cannot overwrite it. Unselected layers perform no feature
addition or allocation. With no selected layers, `ForwardResult` has no `features` key. `capture=True` remains the
separate, expensive parity-dump mode; proposers should not enable it.

`FeatureRequest.final_norm` requests delivery/retention of normalized features to the proposer, not an additional
target normalization. In verification, `VerifiedFeatures` carries `[B,k+1,H]` tensors, or `None` for unrequested
data. For each row, input 0 is the pending token, followed by the k draft tokens. After emitting `count` tokens,
feature index `count-1` produced the last output.

Lifetimes: prefill and ordinary decode pass views into the forward's outputs through `ingest_prefill`; raw graph
outputs are borrowed until the next replay, so consume them on the owner stream or clone the rows you keep. The
verifier clones the requested features before `accepted_batch`, so those may be retained across replays. Retaining a
row view retains its whole backing allocation; budget it or clone the small rows you keep.

## Proposer lifecycle

`proposers/factory.py` builds an adapter, which receives `ProposerContext(device, max_batch, max_context)`, a fixed
width `k` and immutable `ProposerRequirements`, and is installed once before graph capture. The executor checks
width (`max_depth`), context and features before allocating shared verify buffers, without knowing the adapter's
name. MTP supports k=1–8 and DFlash2 k=1–15. Target verify graphs have batch buckets 1, 2, 4, ... up to `max_batch`
(at most 128). With a depth per bucket or per request (`engine/depth.py`), each depth's verify graphs cover only its
buckets and a bucket at depth 0 decodes plainly while the proposer still ingests every committed row. A proposer
with `Capabilities.short_drafts` (MTP and DFlash2) gets `propose(..., depth=d)` and drafts only those `d` tokens (a
DFlash2 round runs one block of the anchor and `d` masked rows), and its `capture_graphs(log, depths={d: buckets})`
receives the verify graphs' pairs (DFlash2 captures a draft graph at each; MTP's one-step graphs serve every depth);
others draft `k` and the first `d` are verified. A proposer with `Capabilities.scored_drafts` (DFlash2) fills
`Proposal.acceptance` with each drafted position's probability of acceptance once `score_drafts` is set; adaptive
verification (`engine/depth.RowBudget`) then has it draft `k` and decides after drafting what the round verifies:
a depth per request in packed verify graphs, accepted on the host (a target without side inputs: the 27B).

- `ingest_prefill(req, ids, start, result)` consumes each freshly computed chunk; `start(req, token)` publishes the
  first emitted token; `ready(req)` decides whether a request may enter speculation (otherwise it decodes plainly).
- `propose(requests, pages, StepInputs, graphs=...)` returns int64 `[B,k]` tokens (`[B,d]` with short drafts) or a `Proposal` (the tokens, their
  actual conditional probabilities, optional sparse support and confidence). `StepInputs` supplies pending tokens,
  positions and slots; `pages` is borrowed `[B,k+1]` target reservation metadata. Adapters cannot allocate target
  pages, mutate logical history or commit target recurrent state, and validate their own token IDs (the hot path
  checks only shape, dtype and device).
- `accepted_batch(requests, emitted, features)` is called once, after **all** target rows are committed and their
  histories updated. `emitted` holds the stop/budget-truncated token lists; row `i` consumes only
  `features[i, :len(emitted[i])]` (the old pending token followed by accepted draft inputs; the last emitted
  correction/bonus is unprocessed). Do not read unused verify padding.
- `prepare_release(req)` attaches draft metadata before target ownership transfers; `release(req)` drops adapter
  state on finish, failure or retraction. `capture_graphs` owns only draft graphs; the engine owns target verify
  graphs and intermediate snapshots.

Optional features are declared in `ProposerRequirements.capabilities` and read once at setup; `check_capabilities`
rejects a flag without its protocol: `device_feedback` (below), `checkpoint_cache`, `stored_payload` (SSD-tier
codec), `draft_head`, `short_drafts`.

The driver verifies `[pending, proposal...]` with greedy prefix acceptance or exact rejection sampling, then
truncates to EOS/budget. For sampled verification, probabilities are FP32 after temperature/top-k/top-p and all
selector effects: dense q is `[B,k,V]`, sparse q and its unique support `[B,k,C]`, and `None` means point-mass
proposals. The verifier accepts with `min(1,p[y]/q[y])`, samples a rejection from normalized `max(p-q,0)`, draws a
target bonus after full acceptance and consumes no target draws past the first rejection; with `accept_soft` s > 0
it also accepts y when u·q[y] < e^s·p[y] on the same draw u, the residual unchanged. Each sampled request owns
its generator, seeded or not. MTP proposals are point masses, or with `HIBANA_MTP_SAMPLE` (on in `fast` and `throughput`)
a sampled batch's drafts drawn from each request's law over the draft head's candidates, with that law as sparse q
([operations](operations.md#sampled-mtp-drafts)); a `greedy_only` proposer uses ordinary decode for
positive temperature. For an all-greedy DFlash2 batch, `kernels/draft_selection.py` walks the conditional selector
lattice on the device with the same candidate rank and tie rules.

## Draft cache and memory

Drafter metadata lives only in a `Snapshot`'s `auxiliary` dict, keyed by target/draft revisions, feature schema,
layout, window and the draft checkpoint source, so a checkpoint served with `--draft` never reuses another draft's
payloads. The drafter writes it into a prefill forward's private snapshots at ingest and into the reply end at
release (`prepare_release(req, snapshot)`), and reads the matched snapshot's (`req.match`) at the first prefill chunk,
which the request's lock keeps until then. Metadata follows the exact node endpoint, never another node's retained
pages. A missing or foreign MTP row (an SSD store written without it) is replaced by a stand-in, one approximate draft
KV row, and the request keeps drafting; a missing DFlash2 window makes the draft unready and the request decodes
plainly. Either is counted in `cache.draft_resumes.fallbacks`.

MTP retains endpoint target norms and writes all teacher-forced committed draft KV, including the bonus input on
full acceptance. DFlash2 owns private KV rings and its radix entries hold projected-feature handles:
`CheckpointFeatureCache` reserves one feature window per usable recurrent slot and `StatePool.free()` invalidates a
slot's handle before recycling it, so every path that frees the slot reclaims its features. At release, endpoint
features belong to the request's own slot being transferred, the reply end; SSD-tier codecs
import with `slot=` naming the new slot, and restoration projects only draft K/V from the saved features.

Pass `SpecConfig` to `Executor` at startup to reserve draft weight/KV/feature costs and verify buffers before the
optional target cache is sized (`cache/spec_budget.py`). Adapter bytes include reserved feature-cache capacity, not
current CUDA allocations; graph, first-use and activation bytes are measured estimates (`cache/ledger.after_plan_bytes`). The pools leave the 6 GiB host floor (`HIBANA_HOST_FLOOR_GIB`) for the OS and other programs.

## Engine and HTTP ownership

HTTP validation lives in `api/validation.py`; `engine/loop.py` owns the scheduler thread and request tickets, whose
events the endpoint adapters in `api/server.py` consume (`api/messages.py` maps Messages onto the chat adapter). A
completed ticket is removed before its terminal notification crosses to the HTTP event loop.

## Device-resident acceptance

With `HIBANA_DEVICE_ACCEPT=1` (on for Flash-Next in `fast` and `throughput`), `reconstruct` state and a proposer
that declares `device_feedback`, `engine/accept_graphs.py` replaces the host's `SpecDecoder._accept` with one
captured graph per batch bucket of each verify depth: it decides acceptance with the host's rule
(`engine/accept.py`) from the verify graph's static outputs, reconstructs the state and commits the convolution
windows for exactly those counts, and calls the proposer's `feedback_device`; the host reads one packed result and
only publishes. All-greedy batches take it; all-sampled batches of MTP drafts (point masses, or sampled drafts with q on
the draft head's 16 candidates; `top_k` at most 64, no
constraint) first replay a decision graph and fall back to the host when a row's distribution does not fit its
candidates. Mixed batches, requests with more than `accept.MAX_STOPS` stop token ids and other proposers stay on the
host path.

Ownership does not move. `StateManager.authorize_reconstruction` checks the batch (unique private slots,
reservations of the verified width, window shapes) before the replay and `publish_reconstructed` transfers pages and
advances `committed_len` after it; a failure between them poisons the batch as the host commit does. The proposer's
`accepted_prepare` validates its private prefixes and returns each request's start position and ring slot,
`feedback_device` writes rows past a request's count to the dummy ring slot or page 0, and `accepted_publish`
advances its state afterwards. Padded rows (budget 0, slot 0) emit nothing and touch only dummies.

The whole-bucket projection rounds BF16 (and, with an FP8 draft layer, shares one activation scale) differently from
the host's projection of the accepted rows, so draft rings agree only to that precision and proposals can drift; the
target's outputs are the same and the committed recurrent state is bitwise the host's for identical round boundaries
(`tests/gpu/test_device_accept.py`, `test_mtp_device_accept.py`). It is judged like any other arithmetic change
([evaluation](evaluation.md#judge-a-change-to-arithmetic)).

## Recurrent retention

`reconstruct` is the default (`--verify-state`, `HIBANA_VERIFY_STATE`), and a quantized recurrent state
(`HIBANA_SSM_STATE=int8/int4`, GDN or Mamba-2) requires it. It retains compact recurrence inputs, replays exactly the
stop/budget-truncated accepted prefix from untouched committed state, and writes directly into the private committed
slots authorized by `StateManager.commit_reconstructed`, which validates counts, verified width, unique non-dummy
slots and convolution bounds before any mutation. Each kernel program reads its whole K-by-V tile before overwriting
it. On the 27B's FP32 pools the verify recurrence and the replay are the CUDA pair of `kernels/gdn_verify.py`, one
arithmetic, so the committed state is the verify's own bit for bit (Flash-Next and other shapes keep the vendored Triton
pair). A failure after mutation marks every affected request uncacheable, and a post-write `MemoryError` is not
retried, so no retry reuses partially advanced state. `snapshots` retains every verify state (in the pool's dtype)
and is the test reference for reconstruction (`tests/gpu/test_lifecycle.py`). An FP16 Mamba-2 state
(`HIBANA_SSM_STATE=fp16_sr`, Nemotron) works with both: verify rounds its running state after every step
exactly as decode stores it, with the same stochastic-rounding bits (keyed on the token position and layer, never on
the slot or batch), so snapshots, replays and decode stay bitwise equal within the format.

## Failure and lifecycle invariants

| Area | Invariant |
|---|---|
| Termination | First-token EOS/length requests reach release. Queued cancellation becomes terminal and is published even when the scheduler is idle. Completion, cancellation and release are idempotent. |
| Speculative commit | Accepted output is truncated at the budget/EOS before the GDN/conv snapshot is selected and KV pages are kept. `committed_len` names the prefix the state and KV represent; cache insertion rejects a mismatched label. |
| Flush | Invalidated entries are hidden at once; locked pages and snapshots stay until their last lock ends, and placement and insertion are bypassed while the flush drains. `/flush_cache` acknowledges invalidation; `flush_pending` reports retained locks. |
| Retraction | Readmission replays prompt plus emitted output and keeps output IDs, stream cursor, first-token time, output budget and RNG state; replay samples only the next new token. |
| Sampling | Each seeded request owns its generator; identical seeds never share one. |
| Accounting | Decode counts use actual output deltas; speculative counts exclude candidates discarded by stopping; tokens per round excludes the first prefill token. |
| Limits | Invalid IDs, empty prompts, bad budgets or sampling values and prompt+budget overflow are rejected before admission; graph sizes and context buffers derive from executor limits; KV lengths are checked before replay planning. |
| Failures | Outstanding tickets fail, HTTP returns 503 or an SSE error, health fails and new work is rejected; owned resources are released where safe, and a failed forward never publishes cache entries. |

CPU lifecycle tests cover cancellation, failure, flush and retraction; GPU lifecycle and draft-lifecycle tests cover
the real model path. State ownership and numerical continuation are separate checks: floating-point partition
changes can change greedy output IDs.
