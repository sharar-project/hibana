# Execution and ownership contracts

Phases 7B/7D keep the GB10 model path specialized and expose four small seams.
`engine/contracts.py` imports no GPU backend. These are Python structural types,
not a plugin registry or a promise of arbitrary model support.

| Boundary | Owner | Consumers |
|---|---|---|
| `ExecutionBackend` | Executor executes steps and owns graph/workspace setup | Scheduler chooses admissions, chunks, decode and retraction |
| `RequestState` / `StateManager` | Request page/slot ownership, cache loans, reservations and processed prefix | Executor and shared verifier |
| `Proposer` | Draft weights/KV, readiness, pending tokens, retained features and draft graphs | Shared speculative driver |
| `ForwardResult` / `FeatureRequest` | Target computes logits inputs and only requested features | Prefill driver and verifier deliver features to the proposer |

Model facts live in `model/qwen38_config.py`, device constraints in
`hardware/gb10.py`, and supported limits in immutable `engine/profile.py`.
`config.py` remains a compatibility import point for the existing kernels and
performs snapshot/hardware validation. This split does not make the current
model-specific attention or recurrent kernels generic.

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

- `admit` locks a matched prefix, allocates the request's suffix and slot, and
  copies the matched recurrent snapshot. Allocation failure returns the slot
  and cache loan. Prompt pages are reserved before prefill has processed them.
- `checkpoint` reserves an optional recurrent snapshot at a prefill boundary;
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

The runtime is single-threaded at this boundary. Calls are ordered on its
current CUDA stream; callbacks must not free/reuse buffers on another stream.
This is an ownership transaction, not a general CUDA exception rollback.
Replacing the cache object is a startup/test operation requiring quiescent
requests and a flushed old cache. Use `Executor.flush_cache()` for an active
runtime; swapping owners while requests hold loans violates their lifetime.

## Forward features and lifetimes

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
allocating shared verify buffers. MTP/n-gram support k=1–4; the pinned external
drafts support k=1–7. Target verify graphs have B=1/2/4 buckets and external
drafts have verify-depth buckets 1/3/7 up to their configured maximum.

`ingest_prefill(req, ids, start, result)` consumes each freshly computed chunk;
`start(req, token)` publishes its first emitted token. `ready(req)` selects
whether a request may enter speculation. Otherwise ordinary decoding
continues. MTP owns its prefill alignment and pending hidden/token state;
n-gram searches a bounded prompt/output suffix and requires no target features.

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
then calls `accepted(req, emitted, features, row)`. `release(req)` drops adapter
state on finish, failure or retraction. `capture_graphs` owns only draft graphs;
the engine owns target verify graphs and intermediate snapshots.

MTP, n-gram, DSpark and DFlash2 share this path. N-gram remains a reference
adapter. Feature feedback covers prefill, ordinary decode and verification.
Accepted-callback graph features are owned clones; callers still need to budget
retained views. DSpark consumes the checkpoint's anchor prediction and applies its
Markov correction; DFlash2 discards the anchor query, then walks its sparse
conditional selector lattice. Neither imports the reference framework.

For sampled verification, probabilities are FP32 after temperature/top-k/top-p
and all selector/Markov effects. Dense q has shape `[B,k,V]`; sparse q and its
unique token support have shape `[B,k,C]`. `None` means point-mass proposals.
The shared verifier accepts with `min(1,p[y]/q[y])`, samples a rejection from
normalized `max(p-q,0)`, and draws a target bonus after full acceptance. It does
not consume target draws past the first rejection. Each sampled request owns
its generator, including requests without a supplied seed. MTP/n-gram still use
ordinary decode for positive temperature. Sampled verification is an explicit
reference path with scalar transfers, not a claimed optimized GPU sampler.

## Draft cache and memory

Radix auxiliary metadata is keyed by target/draft revisions, feature schema,
layout and window. Metadata follows the exact node endpoint and is never
attached to another node's retained target pages during duplicate insertion.
`prepare_release(req)` runs before target ownership transfers. `release(req)`
releases the active draft slot; optional radix features have independent,
bounded ownership. Stale/evicted/foreign metadata makes the draft unready and
uses ordinary decoding without reading uninitialized KV.

MTP retains endpoint target norms and writes all teacher-forced committed draft
KV, including the bonus input on full acceptance. External drafts own private
KV rings and a byte-bounded projected-feature LRU; radix entries hold handles.
DSpark retains the supported target context, DFlash2 a 2,048-token window.
Restoration projects only draft K/V from saved features. Current eviction is
bounded LRU, not an optimized saved-time-per-byte policy.

Pass `SpecConfig` to `Executor` at startup to reserve checkpoint-derived draft
weight/KV/feature costs and verify buffers before optional target cache sizing.
Graph/activation/slack reserves are explicitly estimates. Adapter declared
bytes include reserved feature-cache capacity; they need not equal current
CUDA allocations. The host floor remains 16 GiB. Reclaiming cache IDs does not
release monolithic physical allocations; reclaimable slabs remain Phase 8.

## Recurrent retention and scheduling experiments

`snapshots` retains every FP32 verify state. `reconstruct` retains compact
recurrence inputs, replays exactly the stop/budget-truncated accepted prefix
from untouched committed state, and passes one final FP32 state to
`StateManager.commit`. Both preserve the same ownership boundary. No CPU reads
of hot state, BF16 state compression or managed-memory faults are introduced.

`WidthPolicy` consumes pinned measured cost/calibration profiles and previous
round confidence, avoiding a new barrier between proposal and verification.
It changes verified width, not maximum draft work. Missing history/calibration
uses the maximum configured width. Profiles are opt-in; `bench/calibrate_width.py`
fits a pilot and never promotes it. `LatencyPolicy` keeps bounded local prefill
and decode observations, caps active-step prefill work, and guarantees some
queued progress. Its target is best effort: running GPU kernels are not
preemptible. Fixed defaults remain available for every comparison.
