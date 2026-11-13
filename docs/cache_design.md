# Cache design

How the engine keeps a request's prefix and brings it back: the GPU pools (L1), the radix prefix cache that indexes
them, and the SSD tier below them (on in every profile; `HIBANA_SSD=0` turns it off). [Operations](operations.md) lists the settings; the code is in
`hibana/cache/`.

## Architecture

```
                         ┌──────────────────────────── engine thread ────────────────────────────┐
  request ──► admission ─┤ queue head: deeper resume point on the SSD? ─► restore_async() ─────┐   │
                         │ can_admit (pure) ─► admit: RadixCache.match(ids), lock the match    │   │
                         │   │  deepest L1 prefix with a GDN snapshot                          │   │
                         │   ▼                                                                 │   │
                         │ prefill the rest ─► decode ─► finish ─► RadixCache.insert()         │   │
                         │   │ each chunk: its snapshots go in,     │ (the reply end)          │   │
                         │   │ the request's lock moves to the last │                          │   │
                         │   └► RadixCache.insert() ────────────────┴► SsdTier.backup()        │   │
                         │ every scheduler step: SsdTier.step()  (gather queued writes,       │   │
                         │                       drain completions, enforce the disk budget)   │   │
                         └────────────────────────────────────────────────────────────────────│───┘
                                                                                              │
  L1: GPU memory (unified on GB10)                                                            │
  ┌───────────────────────────────────────────────┐   ┌─────────────────────────────────┐     │
  │ KvPool    FP8 K/V, 16 attention layers,       │   │ StatePool  GDN conv + SSM state │     │
  │           one page per token (+ page aux)     │   │            one slot per snapshot│     │
  └──────────────▲────────────────────────────────┘   └───────────────▲─────────────────┘     │
                 │ pages                                              │ slots                 │
  ┌──────────────┴────────────────────────────────────────────────────┴─────────────────┐     │
  │ RadixCache (radix.py): one tree, the only record of what is cached where            │     │
  │   Node: key tokens · pages (None ⇒ SSD-only) · snapshot · blocks{end: SsdRef}        │     │
  │         · tail: SsdRef · lock_ref · state_lock_ref (written only by inc/dec_lock_ref) │     │
  │   L1 eviction of a leaf with SSD copies ⇒ demote (free pages and slot, keep node)    │     │
  └──────────────┬─────────────────────────────────────────────────────▲────────────────┘     │
                 │ evicting(leaf)                                      │ insert restored path │
  ┌──────────────▼─────────────────────────────────────────────────────┴────────────────┐     │
  │ SsdTier (ssd.py): decides what goes to the NVMe and brings it back                   │◄────┘
  │   backup  ─► gather pages/slot into pinned staging (≤ 512 MiB in flight) ─► writers │
  │   restore ◄─ O_DIRECT reads into a registered 2 × 256 MiB staging buffer            │
  │             (8 reader threads) ─► GPU scatter into fresh pages and a slot           │
  └──────────────┬─────────────────────────────────────────────────────▲────────────────┘
                 │ write / delete (sharded by key,                     │ read_into(view)
                 │  completions in submission order)                   │  + header and CRC32 check
  ┌──────────────▼─────────────────────────────────────────────────────┴────────────────┐
  │ SsdStore (ssd_store.py): content-addressed objects on the NVMe, nothing else          │
  │   v2/e<epoch>/<hh>/<key>.kv    one block of 256 tokens: K, V of every layer, aux,   │
  │                                and a drafter's own KV (Flash-Next's MTP layer)      │
  │   v2/e<epoch>/<hh>/<key>.tail  partial block + GDN snapshot + slot aux + drafter    │
  │   file = 4 KiB header (key, parent hash, tokens, CRC32) + 256-byte-aligned segments │
  └──────────────────────────────────────────────────────────────────────────────────────┘
```

The split of responsibilities:

| Part | Owns | Never does |
|---|---|---|
| `KvPool`, `StatePool` | GPU pages and state slots, their free lists | decide what to keep |
| `RadixCache` | the tree: which tokens are in L1, which are on the SSD, the locks, their counts and the evictable totals, eviction | file I/O, placing snapshots |
| `StateManager` | what a request owns: its slot, pages, private snapshots and its one lock; admission and the checkpoints | eviction order |
| `SsdTier` | when to write and read, staging buffers, the disk budget | own any record of its own |
| `SsdStore` | files, headers, checksums, writer threads, epochs | know about tokens beyond a key |

Only the engine thread touches the tree or calls the tier. Writer threads report back through completions that
`step()` drains, so the tree changes only on the engine thread.

A drafter's state travels with a stored prefix, so a restored request keeps speculating: DFlash2's projected feature
window and Flash-Next's MTP (position, last hidden row), the snapshot's `auxiliary` metadata, go into the tail through
the proposer's `auxiliary_codec`, and the MTP layer's own KV pages, indexed by the target's page ids, go into the KV
blocks beside the target's (`stored_pages`).

## Protection and eviction

The terms are in the [glossary](#glossary); the [memory plan](memory_plan.md) sizes the pools these rules share.

**One way to protect** (SGLang's `lock_ref`). Nothing stays in the cache because of where it is or who used it last;
it stays because a lock keeps it, or until eviction needs its room. A lock (`radix.TreeLock`, taken by
`RadixCache.lock(node, owner=...)` and ended by `unlock`, which is idempotent) is owned by a running request (one each,
`req.lock`), an SSD restore (in flight, or landed and waiting for its request: `job.lock`) or admission's probe
(`_lock_node`, for the length of one check). A lock on node X counts twice, and only `inc_lock_ref` / `dec_lock_ref`,
called only by `lock` and `unlock`, write the counts:

| Count | Is | Keeps |
|---|---|---|
| `lock_ref(n)` | the locks at n or below it | n's KV pages (and an SSD-only n in the tree) |
| `state_lock_ref(n)` | the locks on n | n's snapshot and its drafter metadata |

The evictable totals move with the counts, so admission can budget without evicting: `evictable_size()` and
`protected_size()` are the L1 KV tokens with `lock_ref` 0 and above 0; `state_evictable_size()` and
`state_protected_size()` the L1 snapshots with `state_lock_ref` 0 and above 0. `evictable_leaves` (the unlocked L1
nodes with no L1 child) and `evictable_snapshots` are kept beside them. Splitting a node gives the new head the path
count and no lock of its own; the deeper original keeps its end, its snapshot and the locks on it. Nodes are split and
pruned, never merged: the SSD tier's queued copies hold `Node` references across steps.

**One owner per snapshot.** A `Snapshot` (`snapshot.py`) is a state slot, the position it holds the state after, its
drafter metadata (`auxiliary`: MTP's row, DFlash2's window handle, bound to the slot), `last_access` and `spare`. It
belongs to one node, or to one request inside one prefill step (`req.private`). Moving it moves the slot and the
metadata together; freed, it is dead (slot -1, no metadata).

**A request's lock.**

```
admission    can_admit (pure): the uncached suffix, its decode reserve and the running requests' outstanding
             reserve against the free and evictable KV, a slot against the free and evictable slots, with a probe
             lock on its own match so that match does not count as evictable. Then admit: match, lock the matched
             node, evict for its own slot and pages, copy the snapshot's state into its slot (req.slot). A landed SSD
             restore that locked the node unlocks after the request's lock is taken. Nothing matched: no lock yet.
each chunk   the forward writes the snapshots it places into private ones (allocated by prefill_checkpoints) → the
             chunk end is copied from req.slot → the drafter ingests the chunk (it reads the matched snapshot's
             metadata at the first one, req.match) → commit → checkpoint(req, up_to): insert the prompt up to the
             deepest written private snapshot with every one of them, lock that node, unlock the old one (its
             snapshot is evictable now, its KV stays under the new path), and free the request's previous spare chunk
             end. Where the tree had the span already, the request reads the tree's pages and frees its own. A
             failed forward frees every row's unwritten destinations. Both prefill paths, solo and batched,
             checkpoint only after the ingest.
decode       nothing moves: the request owns its slot and its one lock, nothing private
release      checkpoint(req, len) with req.slot as the reply end (the drafter attaches its metadata first:
             prepare_release), then unlock
retraction   unlock without inserting: what was checkpointed stays, so the retry resumes there
flush        nothing is placed or inserted; locked paths stay until their locks end
```

This is SGLang's `checkpoint` lock move (one lock per request, moved after each chunk, the insert cut back to the
last saved state), with Hibana's drafter metadata: the lock stays at the matched node until the first chunk's ingest,
because MTP's row and DFlash2's window are read then. Between scheduler steps a running request pins at most two
slots, its own and its locked snapshot, so the startup slot floor (`2 x HIBANA_MAX_BATCH + 1` and the SSD restore's
slot, `ledger.py`) is all admission ever waits on; there is no admission reserve.

**Placement** (`Executor._checkpoint_positions`, inside each prefill forward; a position inside the forward sits on
its own 64-token grid, where the GDN kernels snapshot):

| Snapshot | Where | Spare | Why |
|---|---|---|---|
| chunk end | the forward that ends the prompt: its last grid point strictly before the end. An earlier chunk: its own end | the last one no; earlier ones yes, unless the next forward places no point of its own before the end, or would have to place the message boundary or branching point at its own start | SGLang's one state per forward: the lock moves there, and the same prompt sent again resumes at the last one |
| message boundary | the start of the prompt's last message (`SamplingParams.message_boundary`, from the chat API; 0 for `/generate`), on the forward's grid | no | the conversation's next turn keeps the messages before the last one, often not its last tokens (an agent re-renders the generation prompt, or drops a transient message) |
| branching point | when `Match.raw_length` runs a grid or more past the matched snapshot: the last grid point at or before it, in the chunk that contains it (`Match.branching_seqlen`) | no | SGLang's branching point: the next request sharing that prefix resumes there |
| reply end | the request's own slot at release | yes | the next turn when it repeats the reply token for token |

There is no fixed interval grid, no margin before the prompt's end and no prompt-end copy. When slots are short a
forward's slots go first to the message boundary and the last chunk end, then the branching point, then a spare chunk
end, round-robin over a batch's rows; a slot may be missing, and nothing waits for one.

**Eviction.** Short of state slots: only snapshots with `state_lock_ref` 0, the lowest `get_priority` first:
`(not spare, last_access)`, spare snapshots before the rest and the least recently used first within each (SGLang's
SLRU, with spare the probationary segment). `last_access` is a logical tick (`snapshot.tick`), set when the snapshot is
made and when a request resumes from it, never when a request walks past it; resuming also clears `spare`. A spare
destination takes only a free slot or another spare snapshot's, never a not-spare one. An unlocked leaf goes whole
(demoted when stored on the SSD); any other node keeps its KV and loses only its snapshot. Short of KV pages: unlocked
leaves (`evictable_leaves`), the least recently used by the same tick (tensor-parallel ranks choose alike), one already
stored on the SSD first. KV goes only at unlocked leaves. A duplicate insert at a position the tree has keeps the
tree's snapshot, spare only if both were, with the later `last_access`, and frees the duplicate's slot. Every site that
frees pages or a slot tells the SSD tier first (`evicting`, `dropping_snapshot`, `pruning`, `fence`). Nothing with
`lock_ref > 0` or `state_lock_ref > 0` is ever freed.

**Dead KV.** A node no lock reaches, with no snapshot, no children and no tail stored or on its way can never serve a
resume. After every removal, snapshot eviction, demotion, unlock and SSD tail loss (the budget deleting a resident
node's copies, a failed tail write, a lost tail on restore) the tree frees such nodes up the path (`_prune`).

**Invariants** (`RadixCache.check`, which `StateManager.check` runs with what the requests own, and the unit tests
after every scheduler step): `lock_ref(n) == state_lock_ref(n) + Σ lock_ref(child)`; the `state_lock_ref`s add up to
the live locks, and every lock's owner is live (an admitted request, a restore; no probe); a locked node has a snapshot
(or is an SSD restore's target, its snapshot still on the SSD); the evictable totals, `evictable_leaves` and
`evictable_snapshots` equal a fresh recompute; every state slot is owned by exactly one of a request's slot, a node's
snapshot, a restore in flight, or the free list, and every KV page by one of a node, a request, a reservation, a
restore or the free list; no request owns a private snapshot; a request reads the tree's pages exactly up to its lock;
no dead KV is left.

**Stats.** `cache_stats` (`/get_server_info`, `hibana cache`) reports the locks by owner kind, the snapshots (spare
and not), evictions by reason (`slots`, `kv`, `flush`, `superseded`) and spare, pruned nodes, full re-prefills
(admissions that found the prompt's KV but no snapshot to resume from), resumes from a branching point, the evictable
totals, and the drafters' resume hits and fallbacks (`draft_resumes`).

## Glossary

- **KV page**: one token's K and V in every attention layer (`KvPool`); a node or a request owns it.
- **State slot**: one GDN conv and SSM state (`StatePool`); slot 0 is the dummy one.
- **Request's slot** (`req.slot`): the state after the request's committed tokens; at release, its reply end.
- **Snapshot**: a state slot holding the state after a prefix, with the drafters' metadata for it (`auxiliary`).
- **Private snapshot** (`req.private`): a snapshot a prefill forward writes, the request's until that step's
  checkpoint.
- **Lock** (`TreeLock`, `req.lock`): what keeps a node's snapshot and its path's KV; `lock_ref` / `state_lock_ref`
  count them.
- **Checkpoint** (`StateManager.checkpoint`): after a chunk's ingest, the insert of its snapshots and the lock move.
- **Chunk end**: the snapshot a prefill forward places last; the prompt's last chunk end is the last grid point before
  its end.
- **Message boundary**: where the prompt's last message starts, from the chat API.
- **Branching point** (`Match.branching_seqlen`): where a prompt's KV match runs past the snapshot it resumed from.
- **Reply end**: the request's slot inserted at release, at the end of prompt and reply.
- **Spare**: a snapshot no request is known to need yet (a reply end, an earlier chunk end): evicted first.
- **Evictable totals**: the KV tokens and snapshots no lock protects; admission's pure check counts on them.
- **Dead KV**: KV that can no longer serve a resume; pruned.
- **Slot floor**: `2 x HIBANA_MAX_BATCH + 1` state slots and an SSD restore's, the least the memory plan starts with.

## Keys

A KV block's key chains a hash over every token from the start of the sequence; the chain starts from a hash of
the serving identity (target, revision, decode weights, geometry, block size, format). The store therefore keeps
one copy of a shared prefix, whichever request wrote it, and another build never matches it. A tail's key hashes
its last full block's hash with its partial tokens. A read recomputes the key from the header's parent hash and
tokens and checks the segment sizes and two CRC32s, the header's and the payload's, so a collision, a stale file, a
flipped byte or a torn write is a miss, never wrong KV. Startup also drops a file shorter or longer than its header says.

A shared copy is equivalent to the KV another request computes for the same tokens, not always bitwise equal: kernels
round differently with the rows in a call, so the same tokens prefilled in other chunks give slightly different bits,
the drafter's KV included. The GPU cache shares the same way (`insert` keeps a resident node's pages and gives an
SSD-only node the inserting request's), so a session restored after a restart can hold another session's copy of a
stretch they share, and its output can part from the never-evicted session's at a near-tie; with speculation, a changed
acceptance count changes the verify shape and makes that likelier.

## Lifecycle of a cached prefix

```
 each chunk ──► checkpoint: insert its snapshots, the lock moves to the last ──► backup(): queue the prompt's whole
                blocks, and the tails of its not-spare snapshots
 finish ──► insert into L1 ──► backup(): queue full blocks not yet on disk + the reply end's tail
                                   │
                         step() ───┤ gather into staging ──► writers: .tmp, fdatasync, rename
                                   │                                   │
                                   │ completion ◄──────────────────────┘ SsdRef.ready = True
                                   ▼
 L1 pressure ──► evict LRU leaf ──► evicting(): copy its path's queued blocks if staging has room now, else drop
                                   it (serving waits for room only when SSD-only descendants need them; shutdown
                                   always); write its tail, waiting for the oldest writes in flight to free room
                                   ──► demote: pages and slot freed, node stays (SSD-only)
                                   ▼
 next turn ──► at the queue head, match(): L1 prefix ends at depth d, a ready tail at depth D > d
           ──► restore_async() (prefetch): if it fits by admission's pure count, take pages and a slot, lock the
               node; a worker reads blocks (d, D] and the tail while the engine serves everyone else and scatters
               each chunk as it lands; the request waits
           ──► step(): once every rank read it all, import the tail's state, insert back into L1, still locked
           ──► admission: can_admit, then match, lock (the restore unlocks), prefill only the new tokens
                                   ▼
 disk over budget ──► drop SSD-only leaves (LRU), then disk copies of GPU-resident nodes
 shutdown ──► demote every cached prefix; a restart with the same identity lists them again
```

A restore runs only when the pool can also hold the rest of the request and the decode reserve, counted as admission
counts (free and evictable, with the request's own L1 match locked first): otherwise the cache would evict the restored
path before the request could use it, then read it again. Only the queue head starts one, before its admission check,
and none starts while a landed restore still waits for its request: that one locks a slot (`ssd.RESTORE_SLOTS`, part
of the slot floor) until its request takes its own lock, `HOLD_STEPS` pass, or the request is cancelled. A restore that cannot read every
object intact (missing file, bad header, CRC mismatch) returns its pages and slot and reports 0 tokens; the request
prefills from the L1 prefix as if the SSD copy never existed.

## Store and budget

The launcher points the tier at `$HIBANA_RESULTS/ssd/<target>` (one store per target; `HIBANA_SSD_DIR` moves it, e.g.
to another drive). The directory is private (0700: tails hold token ids), one server holds it at a time (a second
one disables its tier rather than share it), and a store on a tmpfs or ramfs is refused: it would sit in memory,
which on a Spark is the GPU's. If a store cannot open, the server logs why and runs without the tier. The budget
(`HIBANA_SSD_GIB`, default 200) never fills the drive: at startup it is capped to what is stored plus the free space
beyond a reserve (a tenth of the drive, at most 64 GiB).

The tier skips prefixes shorter than 4,096 tokens: each tail carries the recurrent snapshot (147 MiB per 27B prefix
in FP32, 58 MiB with `HIBANA_SSM_STATE=int8`), and a short prefix re-prefills in about a second. A prompt's
not-spare snapshots (its message boundary, its last chunk end, a branching point) are written through at their
checkpoint, and a finished request's reply end at its release, where its conversation's next turn resumes; a spare
chunk end gets no tail. When slot pressure takes a not-spare snapshot whose tail is still queued, the tail goes out
first if staging has room right now; slot pressure never waits for the writers, so otherwise the point is dropped
(`tails_through_dropped`), as is a spare one's queued tail. A node whose descendants are already stored (an earlier
chunk end) is demoted with its KV blocks and no snapshot, unless it is not spare. Writer threads write with
O_DIRECT through a 16 MiB buffer each, or where the filesystem refuses O_DIRECT through the page cache, dropped
after each file (`direct_writes` in `cache.ssd.store` says which). A prefix whose queued blocks eviction cannot
copy at once is dropped and counted in `evict_dropped`; for its tail, eviction waits for room. A restart lists the
stored prefixes again only for the same target revision, decode weights, geometry, recurrent-state format and
prefill numerics, and deletes another format's at startup.
`cache.ssd` in `/get_server_info` reports usage, restores and read GB/s; `hibana cache` shows what the tier holds.

## Restore pipeline

```
 reader threads (8)                        GPU
 ──────────────────                        ───
 half A ◄─ O_DIRECT chunk 0
 half B ◄─ O_DIRECT chunk 1                scatter half A ─► KV pages
 half A ◄─ O_DIRECT chunk 2                scatter half B ─► KV pages
   …                                         …
 tail  ◄─ O_DIRECT                          scatter tail ─► last pages + state slot
                                            insert path into L1
```

On GB10 the tier registers the staging buffer with CUDA, and its device pointer is its host address, so the
scatter kernels read it in place: no host memcpy and no host-to-device copy.

The engine does not wait for the reads (`restore_async`). A worker thread reads and checks each chunk; the engine
thread scatters it on the main stream at its next step, in one kernel launch for every pool the tier stores
(`kernels/csrc/kv_copy.cu`, which also gathers backups into their pinned payloads), and the reader refills that
half once the scatter has read it. The scatter stays on the engine thread because a launch issued from the worker,
and the Python that plans it, wait on the GIL behind it: tens of milliseconds a chunk on the 27B. One restore runs at
a time (they share the buffer and the drive); the next request that needs one waits for it. A restore that landed
keeps its node locked (`TreeLock`, the restore its owner) until its request takes its own lock there, `HOLD_STEPS`
pass, or the request is cancelled, so eviction cannot take it first; admission counts it as locked. On two Sparks each rank reads
its own heads, and the ranks agree each step whether every rank's reads have landed before either imports the state
(one integer all-reduce per step while a restore is in flight). The chunk size, not the context, bounds the buffer,
so the pool plan reserves a constant for it (`SsdTier.memory_bound`).

## What a restore costs

Measured on one GB10 with the 27B (FP8 KV, about 32 KB per token; a 258K-token prefix is 8.6 GB in 1,009
objects plus a 147 MiB tail):

| Context | Re-prefill | GPU hit | SSD restore | SSD vs. re-prefill |
|---|---|---|---|---|
| 4K | 2.5 s | — | +0.06 s over a GPU hit | 2.5× faster |
| 258K | 353 s | — | +0.77 s over a GPU hit | 61× faster |

The drive bounds long restores (it alone needs about 620-660 ms for the 8.6 GB); fixed host work (planning, page
allocation, header checks) bounds short ones. After a restore, time to first token is mostly the prefill of the new
tokens over the long prefix.

## Measure it

`profiling/ssd_turns.py` runs concurrent multi-turn conversations against a server with a small KV pool, so turns
restore from the SSD, and reports time to first token, stream gaps and the server's restore counters. With
`HIBANA_SSD_TRACE=1` the server also writes one line per restore, with its read and scatter times, to
`<HIBANA_SSD_DIR>/v2/restores.jsonl`. Use a GPU nothing else runs on and an `HIBANA_SSD_DIR` on the NVMe itself (a
loop device or a file-backed pool serves reads from the host page cache).

## Tensor parallel

Each rank stores its own heads of every stored prefix on its own Spark, and every rank's tree takes the same steps.
What one rank cannot decide alone, the ranks settle with a minimum over all of them: a write is complete once it
completed on every rank, a restore goes ahead only if every rank read every object intact, and a restart lists
only the prefixes every rank stored ([tensor parallel](tensor-parallel.md)).
