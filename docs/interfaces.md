# Execution and ownership contracts

The rules every change to scheduling, speculation, caching or the HTTP lifecycle keeps. The seams are Python
structural types in `engine/contracts.py` (which imports no GPU backend), not a plugin registry.

| Boundary | Owner | Consumers |
|---|---|---|
| `ExecutionBackend` | Executor executes steps and owns graph/workspace setup | Scheduler chooses admissions, chunks, decode and retraction |
| `RequestState` / `StateManager` | Request page/slot ownership, cache loans, reservations and processed prefix | Executor and shared verifier |
| `Proposer` | Draft weights/KV, readiness, pending tokens, retained features and draft graphs | Shared speculative driver |
| `ForwardResult` / `FeatureRequest` | Target computes logits inputs and only requested features | Prefill driver and verifier deliver features to the proposer |

Model facts live in a `Target` value (`sharar/target.py`): the pinned
checkpoint, the layer kinds, the attention and recurrent head shapes, the
vocabulary, the serving profile and the load budget. The executor reads
`model.target` once and sizes the pools, the ledger, the attention plans and
the verify buffers from it; pools and the ledger accept a `target` and default
to the 27B for tools written against it. One process serves one target,
chosen at launch (`sharar serve --target`, `SHARAR_TARGET`); nothing is
discovered at run time. `model/qwen38_config.py` holds the 27B's constants,
`config.py` is the facade over that default target, device constraints are in
`hardware/gb10.py` and the limits in `engine/profile.py`. The attention and
recurrent kernels take their shapes from the tensors and the target, but are
validated only for the pinned targets.

A second target adds a `Target` instance, a loader in `target.load_model`,
and a model package whose forward honors `ForwardResult`; the scheduler,
state manager, radix cache, verifier and proposer contracts do not change.
Qwen3.8-Flash-Next (`flashnext`, `sharar/model/flashnext/`) is the second
one: `Target.family` picks its loader, the family-specific facts (experts,
hyper-connections, the prefix-lookup embedding, the sparse-attention indexer,
the GDN output gate) are optional fields that default to "absent" for the 27B,
and `Target.cache_env` names the model cache (`SHARAR_FLASHNEXT_CACHE`).
Two small extensions carry its extra state through the existing contracts:
`StatePool.register_aux` allocates per-slot tensors that are cleared, copied
and parked with the recurrent states (the PLE's convolution window lives
there), and the model's `forward_extras()` puts everything else a forward
needs on `ForwardContext.extras`: the PLE history and table rows and the
sparse-attention bookkeeping (`model/flashnext/extras.py`). The engine builds
them through that object for eager forwards, lets it own and fill the graphs'
static side inputs, and asks it for the graph variant ("dense" or
"selecting") a batch needs; a model without side inputs returns None (the
27B). Every model implements `forward_extras`, `register_pages` and
`register_state`.
Checkpoint construction lives in `proposers/factory.py`. `max_depth` and feature
requirements let the executor validate an adapter without knowing its name.

MTP accepted feedback owns bounded pinned token staging and a copy-completion
event. The ordinary target acceptance download completes the previous copy;
direct callers also get an event lifetime check before staging reuse. One upload
feeds per-request KV correction at the original projection row shape, preserving
BF16 rounding and target-conditioned bonus-input KV. Required writes are queued
on the executor's CUDA stream.

## Request state

`Request` owns logical prompt/output history, RNG, stream cursor and lifecycle.
Its `RequestState` base groups resource handles and `committed_len`.
Only `StateManager` transfers request page/slot ownership or advances this
processed prefix. Kernels may write supplied views on the executor's CUDA
stream. Pool free lists are private allocator details; callers use allocation,
free and `free_count` operations, never mutate those lists.

After prefill or successful decode, the last emitted token is pending. Both
target KV and the recurrent slot describe exactly
`prompt_ids + output_ids[:-1]`. That length is `committed_len`. A cache entry
may publish only that same prefix and its matching snapshot.

- `can_admit(prompt_ids, reserve_tokens)` takes actual prefill IDs, including
  replayed output history. It budgets the uncached suffix plus reserve while
  temporarily protecting the matching path. This probe does not increment
  cache-hit counters. Admission follows on the same owner thread.
- `admit` secures state capacity before locking a matched prefix, allocates the request's suffix and slot, and
  copies the matched recurrent snapshot. Allocation failure returns the slot
  and cache loan. Prompt pages are reserved before prefill has processed them.
- `prefill_checkpoints` gives each row its latest interior boundary first,
  then reserves earlier intervals round-robin. `checkpoint` may recycle only
  that request's older completed snapshots; destinations beyond `committed_len`
  are pending kernel writes and cannot be recycled. An exact prompt-end copy
  preserves the latest interior boundary needed by identical retries.
  `commit_prefill` publishes the successfully processed extent.
- `reserve(requests, width)` attaches one disjoint pending reservation per
  request. `step_pages` borrows committed pages plus the reservation. Callers
  may write through these views; they do not acquire ownership.
- `commit(count, ssm=..., conv=...)` copies the verify snapshot after input
  `count-1`, keeps exactly `count` pages and returns the rejected tail. Ordinary
  decode commits its single page after a successful in-place forward/sample.
- `rollback` frees pending pages and disables caching. It does **not** undo
  in-place kernel writes. Verification leaves the committed slot intact;
  after an ordinary forward failure, release or retract/replay before any
  retry. The serving failure path terminates its tickets and releases them.
- `retract` drops physical resources. Executor preserves emitted history,
  remaining budget, RNG and stream cursor, and replays that history on admission.
- `release` is idempotent. It transfers a valid prefix into the cache or frees
  the request's suffix, slot and checkpoints, then unlocks the borrowed prefix.
  Active cache flush defers reclamation until the last loan is released.

When only recurrent capacity is exhausted, the radix cache can drop an unlocked
interior snapshot and its matching draft metadata while retaining KV pages and
newer resume points. `snapshot_evictions` counts these separately from leaf
evictions. This returns a state slot to the allocator; it does not release the
monolithic pool's physical DRAM. Locked paths remain protected. A new snapshot
attached to an existing node must bring its own matching auxiliary metadata;
discarding a duplicate snapshot must preserve the existing metadata as well.

The runtime is single-threaded at this boundary. Calls are ordered on its
current CUDA stream; callbacks must not free/reuse buffers on another stream.
This is an ownership transaction, not a general CUDA exception rollback.
Replacing the cache object is a startup/test operation requiring quiescent
requests and a flushed old cache. Use `Executor.flush_cache()` for an active
runtime; swapping owners while requests hold loans violates their lifetime.

## Forward features and lifetimes

`ForwardContext.gdn_plan` belongs to one immutable prefill request layout.
`model/gdn_plan.py` derives sequence starts, initial-state flags, 64-token
chunk indices/offsets and checkpoint locations from CPU metadata once; all
GDN layers share the resulting tensors. Create a new context when lengths,
slots, prefixes or checkpoints change. Layers must not reconstruct this plan
or read lengths back from CUDA. Planning does not set `ctx.cu_seqlens` for a
cold single request, because that would change `ctx.batched` and select a
different attention path. Standalone chunk callers may omit the prepared
metadata and use the existing fallback.

For `T` flattened tokens and hidden size `H`, `ForwardResult['norm']` is BF16
`[T,H]`: the final Gemma RMSNorm output consumed by the target LM head.
The target needs it even when no proposer requests features.

`FeatureRequest.layers` selects **zero-based post-block residual streams**:
`hidden + residual` after that block's MLP, before the next block's input norm
(or before final norm for the last block). Each selected BF16 `[T,H]` tensor
has fresh storage, so later in-place residual updates cannot overwrite it.
Unselected layers perform no feature addition or allocation. With no selected
layers, `ForwardResult` has no `features` key. `capture=True` remains the
separate, expensive parity-dump mode; proposers should not enable it.

`FeatureRequest.final_norm` requests delivery/retention of normalized features
to the proposer, not an additional target normalization. In verification,
`VerifiedFeatures` carries `[B,k+1,H]` tensors, or `None` for unrequested data.
For each row, input 0 is the pending token, followed by the k draft tokens.
After emitting `count` tokens, feature index `count-1` produced the last output.

Prefill passes views into that forward's outputs. Eager selected features have
independent storage. Raw graph outputs are borrowed until the next replay;
the shared verifier clones only requested features before calling `accepted`.
Those callback tensors may safely be retained across replay. Retaining a row
view retains its whole backing allocation; adapters must budget that memory
or clone the small rows they keep. MTP retains final normalized rows and
releases them with `req.proposer_state`.
Ordinary decode delivers borrowed graph rows through `ingest_prefill` too:
consume them on the owner stream before the next replay or clone retained rows.
The built-in adapters project/write their cache immediately and clone endpoint
metadata. `accepted` is the callback whose graph feature tensors are owned clones.

## Proposer lifecycle

An adapter receives `ProposerContext(device, max_batch, max_context)`, a fixed
width `k` and immutable `ProposerRequirements`. Install once before capturing
graphs. The executor rejects unsupported width/context/features before
allocating shared verify buffers. MTP supports k=1–8 and DFlash2 k=1–15. Target verify graphs have batch buckets 1, 2, 4, ... up to
`max_batch` (at most 128). Normal startup captures the
selected verification depth; benchmarks can explicitly capture additional
supported depths. `SHARAR_SPEC_DEPTHS` sets a depth per batch bucket
(`engine/depth.py` `BatchDepth`): each depth's verify graphs cover only its
buckets, and a bucket at depth 0 decodes plainly while the proposer still
ingests every committed row, so speculation resumes when the batch shrinks. A
proposer that declares `Capabilities.short_drafts` (MTP) receives
`propose(..., depth=d)` and drafts only the `d` tokens a shallower round
verifies; others draft `k` and the round verifies the first `d`.

`ingest_prefill(req, ids, start, result)` consumes each freshly computed chunk;
`start(req, token)` publishes its first emitted token. `ready(req)` selects
whether a request may enter speculation. Otherwise ordinary decoding
continues. MTP owns its prefill alignment and pending hidden/token state; the
tests' n-gram proposer (`tests/ngram_proposer.py`) searches a bounded prompt/output
suffix and requires no target features.

`propose(requests, pages, StepInputs, graphs=...)` returns int64 `[B,k]` tokens
or a `Proposal` containing those tokens, their actual conditional probabilities,
optional sparse support and confidence. `StepInputs` supplies pending tokens, positions and slots;
`pages` is borrowed `[B,k+1]` target reservation metadata. MTP uses those IDs
to index its own KV allocation. Adapters cannot allocate target pages, mutate
logical history or commit target recurrent state. Validate proposal token IDs
within the adapter; the hot path validates shape/dtype/device without a new
device-to-host vocabulary scan.

The driver verifies `[pending, proposal...]`, performs greedy prefix acceptance
or exact rejection sampling, truncates to EOS/budget, commits the corresponding target state,
then calls `accepted_batch(requests, emitted, features)` once, after **all**
target rows are committed and their emitted histories updated. `emitted` is a
list of stop/budget-truncated token lists; row `i` consumes only
`features[i, :len(emitted[i])]`. These features describe processed verification
inputs (the old pending token followed by accepted draft inputs); the last
emitted correction/bonus remains unprocessed. Adapters must not read unused
verify padding. `release(req)` drops adapter
state on finish, failure or retraction. `capture_graphs` owns only draft graphs;
the engine owns target verify graphs and intermediate snapshots.

MTP and DFlash2 share this path. Feature feedback covers prefill, ordinary decode
and verification. Accepted-callback graph features are owned clones; callers still
need to budget retained views. DFlash2 discards the anchor query, then walks its
sparse conditional selector lattice, without importing the reference framework.

DFlash2 concatenates the committed feature rows for one shared projection
and five layer KV projections, then scatter into their private rings. Temporary
work is bounded by `max_batch * (k + 1)` rows. Metadata uses one asynchronous
upload; no GPU scalar extraction is added. Single-row feedback and interior
checkpoint cases use the per-row helper. MTP's batch callback runs its existing
row calculations. Target state/page ownership remains
with StateManager.

For an all-greedy DFlash2 batch, `kernels/draft_selection.py` traverses that
lattice on the device and returns owned IDs. It preserves candidate rank/tie
semantics, borrows score/candidate graph views only for the ordered kernel
launch, and adds no target-state ownership. Mixed/sampled batches keep the
conditional-law path below.

For sampled verification, probabilities are FP32 after temperature/top-k/top-p
and all selector effects. Dense q has shape `[B,k,V]`; sparse q and its
unique token support have shape `[B,k,C]`. `None` means point-mass proposals.
The shared verifier accepts with `min(1,p[y]/q[y])`, samples a rejection from
normalized `max(p-q,0)`, and draws a target bonus after full acceptance. It does
not consume target draws past the first rejection. Each sampled request owns
its generator, including requests without a supplied seed. MTP proposals are
verified as point masses; a `greedy_only` proposer uses ordinary decode for positive temperature.

## Draft cache and memory

Radix auxiliary metadata is keyed by target/draft revisions, feature schema,
layout, window and the draft checkpoint source, so a trained checkpoint served
with `--draft` never reuses another draft's payloads. Trained checkpoints keep
the feature and K/V projections of the publisher's draft, so their projected
rings are interchangeable; the identity is conservative anyway. Metadata follows the exact node endpoint and is never
attached to another node's retained target pages during duplicate insertion.
`prepare_release(req)` runs before target ownership transfers. `release(req)`
releases the active draft ring slot. In a cached executor, parallel-draft
feature handles share the lifetime of their owning target recurrent slot.
Stale/evicted/foreign metadata makes the draft unready and
uses ordinary decoding without reading uninitialized KV.

MTP retains endpoint target norms and writes all teacher-forced committed draft
KV, including the bonus input on full acceptance. External drafts own private
KV rings; radix entries hold projected-feature handles. `CheckpointFeatureCache`
reserves one maximum-size feature window per usable recurrent slot, with
allocations made only for saved checkpoints. `StatePool.free()` invalidates the
associated handle before recycling its slot. Radix eviction, duplicate snapshot
insertion, cancellation, retraction and flush therefore reclaim matching feature
payloads through the same slot lifetime. The standalone `FeatureCache` LRU remains
a benchmark control and a fallback for adapters outside a cached executor.

At release, endpoint features must belong to the committed slot being transferred,
not the duplicate prompt-end checkpoint that `StateManager` frees. Cold-tier codecs
import with `slot=` naming the newly allocated target slot. Copies and frees remain
on the engine stream. Prospective allocation checks include the feature-window cap
and gather/replacement temporaries before sizing optional target pools; dropping
feature tensors returns allocator capacity, not a promise to release DRAM to the OS.
DFlash2 retains a 2,048-token window.
Restoration projects only draft K/V from saved features. Target-snapshot eviction
uses bounded LRU, not an optimized saved-time-per-byte policy.

Pass `SpecConfig` to `Executor` at startup to reserve checkpoint-derived draft
weight/KV/feature costs and verify buffers before optional target cache sizing.
Graph/activation/slack reserves are explicitly estimates. Adapter declared
bytes include reserved feature-cache capacity; they need not equal current
CUDA allocations. The host floor is 16 GiB. Reclaiming cache IDs does not
release monolithic physical allocations.

## Engine and HTTP ownership

HTTP validation lives in `api/validation.py`; `engine/loop.py` owns the scheduler
thread and request tickets. Endpoint adapters in `api/server.py` consume their
events; `api/messages.py` translates Anthropic Messages requests onto the chat
adapter. A completed ticket is removed before its terminal notification crosses
to the HTTP event loop. Proposer checkpoint-cache binding and cold-tier codecs
are optional capabilities discovered once during executor setup; adapters
without them retain the base proposer contract.

## Device-resident acceptance

For an all-greedy batch on the graph path, `engine/accept_graphs.py` replaces
the host's `SpecDecoder._accept` with one captured graph per batch bucket of
each verify depth (a depth per bucket, `SHARAR_SPEC_DEPTHS`, captures only its buckets)
(`SHARAR_DEVICE_ACCEPT=1`; the `fast` profile turns it on for Flash-Next, `0` keeps the host path for every round). The graph
reads the verify graph's static outputs, decides acceptance with the same rule
(`engine/accept.py`: exact or relaxed match, cumulative prefix, budget clamp,
first stop token), runs the accepted-state reconstruction and the convolution
commit for exactly those counts, and calls the proposer's `feedback_device` to
project and scatter the accepted rows. The host reads one packed result
(proposals, argmax rows, accepted count, emitted count, relaxed flags) through
a pinned buffer and an event, and only publishes.

Ownership does not move. `StateManager.authorize_reconstruction` performs the
batch checks (unique private slots, reservations of the verified width,
window shapes) before the replay; `publish_reconstructed` transfers pages and
advances `committed_len` after the counts are known, and a failure between
them poisons the batch exactly as the host commit does. A proposer opts in by
providing three methods: `accepted_prepare(requests)` validates its private
prefixes before the replay and returns each request's start position and ring
slot; `feedback_device(features, starts, ring_slots, counts, steps, *, norm,
tokens, pages)` is the captured feedback over the bucket's `[b*steps, H]` verify
features (or, for a proposer fed final-normalized rows such as MTP, the verify
rows with the block's input tokens and pages), writing rows past a request's
count (and rows of padded requests) to the dummy ring slot or page 0;
`accepted_publish(requests, emitted)` advances its state afterwards.
Padded rows of a bucket carry budget 0 and target slot 0, so they emit nothing
and touch only the dummy recurrent slot. Requests with more than
`accept.MAX_STOPS` stop token ids, sampled batches and proposers without the
three methods stay on the host path.

The projection over the whole bucket rounds bf16 differently from the host
path's projection of exactly the accepted rows, so the drafter's rings agree
to bf16 precision and its proposals can drift after a few rounds; the
target's outputs are the same, and the committed recurrent state is bitwise
the host's for identical round boundaries (`tests/gpu/test_device_accept.py`,
`test_mtp_device_accept.py`). With an FP8 draft layer (Flash-Next's MTP head) the
whole-bucket projection also shares one dynamic activation scale, so the draft KV
matches the host's to FP8 rounding and round boundaries, hence outputs, move sooner.
Acceptance is judged like any other arithmetic change ([evaluation](evaluation.md#judge-a-change-to-arithmetic)).

## Recurrent retention

`snapshots` retains every FP32 verify state. `reconstruct` retains compact
recurrence inputs, replays exactly the stop/budget-truncated accepted prefix
from untouched committed state, and writes directly into the private committed
slots authorized by `StateManager.commit_reconstructed`. The manager validates
all counts against the current verified width, unique non-dummy slots and
convolution bounds before any mutation;
it also prepares retained page tensors before launching reconstruction. Each
kernel program reads its entire K-by-V-tile state before overwriting that tile.
Convolution copies and page/prefix publication follow on the same stream, then
the engine publishes emitted tokens and calls proposer feedback.

No final FP32 output buffer is retained. A failure after mutation marks every
affected request uncacheable. Post-write `MemoryError` becomes a non-retryable
failure so pool-pressure retry cannot reuse partially advanced recurrent state.
Rollback frees reservations; it does not undo in-place writes. Full snapshots
remain the default and bitwise control. No CPU reads of hot state, BF16 state
compression or managed-memory faults are introduced.

## Failure and lifecycle invariants

| Area | Invariant |
|---|---|
| Termination | First-token EOS/length requests reach release. Queued cancellation becomes terminal and is published even when the scheduler is idle. Completion, cancellation and release are idempotent. |
| Speculative commit | Accepted output is truncated at the budget/EOS before the GDN/conv snapshot is selected and KV pages are kept. `committed_len` names the prefix the state and KV represent; cache insertion rejects a mismatched label. |
| Flush | Invalidated entries are hidden at once; pinned pages and snapshots stay until their last reader releases them, and insertion is bypassed while the flush drains. `/flush_cache` acknowledges invalidation; `flush_pending` reports retained readers. |
| Retraction | Readmission replays prompt plus emitted output and keeps output IDs, stream cursor, first-token time, output budget and RNG state; replay samples only the next new token. |
| Sampling | Each seeded request owns its generator; identical seeds never share one. |
| Accounting | Decode counts use actual output deltas; speculative counts exclude candidates discarded by stopping; tokens per round excludes the first prefill token. |
| Limits | Invalid IDs, empty prompts, bad budgets or sampling values and prompt+budget overflow are rejected before admission; graph sizes and context buffers derive from executor limits; KV lengths are checked before replay planning. |
| Failures | Outstanding tickets fail, HTTP returns 503 or an SSE error, health fails and new work is rejected; owned resources are released where safe, and a failed forward never publishes cache entries. |

CPU lifecycle tests cover cancellation, failure, flush and retraction; checkpoint-feature tests cover auxiliary
ownership; GPU lifecycle, draft-lifecycle and cold-tier tests cover the real model path. Exact state ownership
and numerical continuation are separate checks: floating-point partition changes can change greedy output IDs.
