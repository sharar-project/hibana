# Cache design

How the engine keeps a request's prefix and brings it back: the GPU pools (L1), the radix prefix cache that indexes
them, and the SSD tier below them (on in every profile; `SHARAR_SSD=0` turns it off). [Operations](operations.md) lists the settings; the code is in
`sharar/cache/`.

## Architecture

```
                         ┌──────────────────────────── engine thread ────────────────────────────┐
  request ──► admission ─┤ RadixCache.match(ids)                                                 │
                         │   │  deepest L1 prefix with a GDN snapshot                              │
                         │   │  deeper resume point on the SSD? ──► SsdTier.restore_async() ───┐   │
                         │   ▼                                                                 │   │
                         │ prefill the rest ─► decode ─► finish ─► RadixCache.insert()         │   │
                         │                                          │                          │   │
                         │                                          └► SsdTier.backup()        │   │
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
  │   Node: key tokens · pages (None ⇒ SSD-only) · snapshot_slot · blocks{end: SsdRef}   │     │
  │         · tail: SsdRef · lock                                                        │     │
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
| `RadixCache` | the tree: which tokens are in L1, which are on the SSD, locks, LRU | file I/O |
| `SsdTier` | when to write and read, staging buffers, the disk budget | own any record of its own |
| `SsdStore` | files, headers, checksums, writer threads, epochs | know about tokens beyond a key |

Only the engine thread touches the tree or calls the tier. Writer threads report back through completions that
`step()` drains, so the tree changes only on the engine thread.

A drafter's state travels with a stored prefix, so a restored request keeps speculating: DFlash2's projected feature
window and Flash-Next's MTP (position, last hidden row) go into the tail through the proposer's `auxiliary_codec`,
and the MTP layer's own KV pages, indexed by the target's page ids, go into the KV blocks beside the target's
(`stored_pages`).

## Keys

A KV block's key chains a hash over every token from the start of the sequence; the chain starts from a hash of
the serving identity (target, revision, decode weights, geometry, block size, format). The store therefore keeps
one copy of a shared prefix, whichever request wrote it, and another build never matches it. A tail's key hashes
its last full block's hash with its partial tokens. A read recomputes the key from the header's parent hash and
tokens and checks the segment sizes and two CRC32s, the header's and the payload's, so a collision, a stale file, a
flipped byte or a torn write is a miss, never wrong KV. Startup also drops a file shorter or longer than its header says.

## Lifecycle of a cached prefix

```
 finish ──► insert into L1 ──► backup(): queue full blocks not yet on disk + tails at resume points
                                   │
                         step() ───┤ gather into staging ──► writers: .tmp, fdatasync, rename
                                   │                                   │
                                   │ completion ◄──────────────────────┘ SsdRef.ready = True
                                   ▼
 L1 pressure ──► evict LRU leaf ──► evicting(): copy its path's queued blocks and tail if staging has room now,
                                   else drop it (serving never waits for the writers; shutdown does)
                                   ──► demote: pages and slot freed, node stays (SSD-only)
                                   ▼
 next turn ──► match(): L1 prefix ends at depth d, a ready tail at depth D > d
           ──► restore_async(): take pages and a slot, lock the path; a worker reads blocks (d, D] and the tail
               while the engine serves everyone else and scatters each chunk as it lands; the request waits
           ──► step(): once every rank read it all, import the tail's state, insert back into L1, hold it
           ──► admission: match, prefill only the new tokens
                                   ▼
 disk over budget ──► drop SSD-only leaves (LRU), then disk copies of GPU-resident nodes
 shutdown ──► demote every cached prefix; a restart with the same identity lists them again
```

A restore runs only when the pool can also hold the rest of the request and its decode reserve: otherwise the cache
would evict the restored path before the request could use it, then read it again. A restore that cannot read every
object intact (missing file, bad header, CRC mismatch) returns its pages and slot and reports 0 tokens; the request
prefills from the L1 prefix as if the SSD copy never existed.

## Store and budget

The launcher points the tier at `$SHARAR_RESULTS/ssd/<target>` (one store per target; `SHARAR_SSD_DIR` moves it, e.g.
to another drive). The directory is private (0700: tails hold token ids), one server holds it at a time (a second
one disables its tier rather than share it), and a store on a tmpfs or ramfs is refused: it would sit in memory,
which on a Spark is the GPU's. If a store cannot open, the server logs why and runs without the tier. The budget
(`SHARAR_SSD_GIB`, default 200) never fills the drive: at startup it is capped to what is stored plus the free space
beyond a reserve (a tenth of the drive, at most 64 GiB).

The tier skips prefixes shorter than 4,096 tokens: each tail carries the recurrent snapshot (147 MiB per 27B prefix
in FP32, 58 MiB with `SHARAR_SSM_STATE=int8`), and a short prefix re-prefills in about a second. A finished
request's snapshot is written at its end, where its conversation's next turn resumes, not at its prompt's end; a
request that regenerates from the same prompt may prefill its last turn again. A node whose descendants are already
stored (a prompt's end, a chunk checkpoint) is demoted with its KV blocks and no snapshot. Writer threads write with
O_DIRECT through a 16 MiB buffer each, or where the filesystem refuses O_DIRECT through the page cache, dropped
after each file (`direct_writes` in `cache.ssd.store` says which). A prefix eviction cannot copy at once is dropped
and counted in `evict_dropped`. A restart lists the stored prefixes again only for the same target revision, decode
weights, geometry, recurrent-state format and prefill numerics, and deletes another format's at startup.
`cache.ssd` in `/get_server_info` reports usage, restores and read GB/s; `sharar cache` shows what the tier holds.

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
thread scatters it on the main stream at its next step, a few milliseconds of kernel launches, and the reader
refills that half once the scatter has read it. The scatter stays on the engine thread because launches issued from
the worker wait on the GIL behind it. One restore runs at a time (they share the buffer and the drive); the next
request that needs one waits for it. A restore that landed keeps its path locked until its request is admitted, so
eviction cannot take it first. On two Sparks each rank reads its own heads, and the ranks agree each step whether
every rank's reads have landed before either imports the state (one integer all-reduce per step while a restore is
in flight). The chunk size, not the context, bounds the buffer, so the pool plan reserves a constant for it
(`SsdTier.memory_bound`).

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
`SHARAR_SSD_TRACE=1` the server also writes one line per restore, with its read and scatter times, to
`<SHARAR_SSD_DIR>/v2/restores.jsonl`. Use a GPU nothing else runs on and an `SHARAR_SSD_DIR` on the NVMe itself (a
loop device or a file-backed pool serves reads from the host page cache).

## Tensor parallel

Each rank stores its own heads of every stored prefix on its own Spark, and every rank's tree takes the same steps.
What one rank cannot decide alone, the ranks settle with a minimum over all of them: a write is complete once it
completed on every rank, a restore goes ahead only if every rank read every object intact, and a restart lists
only the prefixes every rank stored ([tensor parallel](tensor-parallel.md)).
