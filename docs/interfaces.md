# Execution and ownership contracts

Phase 7B keeps the GB10 model path specialized and exposes four small seams.
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

## Proposer lifecycle

An adapter receives `ProposerContext(device, max_batch, max_context)`, a fixed
width `k` and immutable `ProposerRequirements`. Install once before capturing
graphs. The executor rejects unsupported width/context/features and a request
for stochastic verification before allocating shared verify buffers.

`ingest_prefill(req, ids, start, result)` consumes each freshly computed chunk;
`start(req, token)` publishes its first emitted token. `ready(req)` selects
whether a greedy request may enter speculation. Otherwise ordinary decoding
continues. MTP owns its prefill alignment and pending hidden/token state;
n-gram searches a bounded prompt/output suffix and requires no target features.

`propose(requests, pages, StepInputs, graphs=...)` returns int64 `[B,k]` tokens
on the target device. `StepInputs` supplies pending tokens, positions and slots;
`pages` is borrowed `[B,k+1]` target reservation metadata. MTP uses those IDs
to index its own KV allocation. Adapters cannot allocate target pages, mutate
logical history or commit target recurrent state. Validate proposal token IDs
within the adapter; the hot path validates shape/dtype/device without a new
device-to-host vocabulary scan.

The driver verifies `[pending, proposal...]`, performs exact greedy prefix
acceptance, truncates to EOS/budget, commits the corresponding target snapshot,
then calls `accepted(req, emitted, features, row)`. `release(req)` drops adapter
state on finish, failure or retraction. `capture_graphs` owns only draft graphs;
the engine owns target verify graphs and intermediate snapshots.

MTP and n-gram share this path. N-gram is a contract/reference adapter, not a
claimed speedup. Current limitations remain explicit: flat proposals, greedy
verification, one startup adapter, fixed k, and no draft-aware prefix cache.
Feature feedback currently covers prefill and verification; an adapter that
needs to rebuild readiness from ordinary decode steps needs an explicit
decode-observation extension in 7D. Do not silently reuse stale hidden state.
Requirement byte counts describe adapter allocations; complete prospective
draft/graph/feature/slab budgeting is the next memory-planning work in 7D/8.
