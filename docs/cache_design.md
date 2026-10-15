# Cache design

How the engine keeps a request's prefix and brings it back: the GPU pools (L1), the radix prefix cache that indexes
them, and the SSD tier below them (on in every profile; `SHARAR_SSD=0` turns it off). [Operations](operations.md) lists the settings; the code is in
`sharar/cache/`.

## Architecture

```
                         ┌──────────────────────────── engine thread ────────────────────────────┐
  request ──► admission ─┤ RadixCache.match(ids)                                                 │
                         │   │  deepest L1 prefix with a GDN snapshot                              │
                         │   │  deeper resume point on the SSD? ──► SsdTier.restore()  ────────┐   │
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

A drafter's state travels with a session, so a restored request keeps speculating: DFlash2's projected feature
window and Flash-Next's MTP (position, last hidden row) go into the tail through the proposer's `auxiliary_codec`,
and the MTP layer's own KV pages, indexed by the target's page ids, go into the KV blocks beside the target's
(`stored_pages`).

## Keys

A KV block's key chains a hash over every token from the start of the sequence; the chain starts from a hash of
the serving identity (target, revision, decode weights, geometry, block size, format). The store therefore keeps
one copy of a shared prefix, whichever session wrote it, and another build never matches it. A tail's key hashes
its last full block's hash with its partial tokens. A read recomputes the key from the header's parent hash and
tokens and checks the segment sizes and the CRC, so a collision, a stale file or a torn write is a miss, never
wrong KV.

## Lifecycle of a session

```
 finish ──► insert into L1 ──► backup(): queue full blocks not yet on disk + tails at resume points
                                   │
                         step() ───┤ gather into staging ──► writers: .tmp, fdatasync, rename
                                   │                                   │
                                   │ completion ◄──────────────────────┘ SsdRef.ready = True
                                   ▼
 L1 pressure ──► evict LRU leaf ──► evicting(): flush its queued blocks, write its tail if needed
                                   ──► demote: pages and slot freed, node stays (SSD-only)
                                   ▼
 next turn ──► match(): L1 prefix ends at depth d, a ready tail at depth D > d
           ──► restore(): read blocks (d, D] and the tail, scatter, insert back into L1
           ──► prefill only the new tokens
                                   ▼
 disk over budget ──► drop SSD-only leaves (LRU), then disk copies of GPU-resident nodes
 shutdown ──► demote every session; a restart with the same identity lists them again
```

A restore runs only when the pool can also hold the rest of the request and its decode reserve: otherwise the cache
would evict the restored path before the request could use it, then read it again. A restore that cannot read every
object intact (missing file, bad header, CRC mismatch) returns its pages and slot and reports 0 tokens; the request
prefills from the L1 prefix as if the SSD copy never existed.

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
scatter kernels read it in place: no host memcpy and no host-to-device copy. The chunk size, not the context,
bounds the buffer, so the pool plan reserves a constant for it (`SsdTier.memory_bound`).

## What a restore costs

Measured on one GB10 with the 27B (FP8 KV, about 32 KB per token; a 258K-token session is 8.6 GB in 1,009
objects plus a 147 MiB tail):

| Context | Re-prefill | GPU hit | SSD restore | SSD vs. re-prefill |
|---|---|---|---|---|
| 4K | 2.5 s | — | +0.06 s over a GPU hit | 2.5× faster |
| 258K | 353 s | — | +0.77 s over a GPU hit | 61× faster |

- **The drive bounds long restores.** A 258K restore takes about 663 ms; the drive alone needs about
  620–660 ms for the 8.6 GB.
- **Fixed host work bounds short restores.** Planning, page allocation and per-object header checks are the
  part software can still cut, and they show most at short contexts (32K: 116 ms for 1.2 GB).
- **More reader threads do not help.** Reader threads share the GIL with the engine thread; 16 threads and a
  separate checksum pool were both slower than 8.
- **After a restore, attention dominates.** Time to first token after a hit is mostly the prefill of the new
  tokens over the long prefix (about 5 s for 2,048 new tokens at 258K, 0.19 s for one).

Two things would move the numbers further: overlapping the restore with compute layer by layer, and more drive
bandwidth (a second NVMe striped with the first, or GPUDirect Storage).

## Benchmark it yourself

This measures three times to first token for one long context: re-prefilling it, finding it in GPU memory, and
restoring it from the SSD. It needs only the server. Use a GPU nothing else runs on, and an `SHARAR_SSD_DIR` on
the NVMe itself (a loop device or a file-backed pool serves reads from the host page cache).

**1. Serve with the SSD tier on.**

```bash
uv run sharar serve 27b --detach
```

**2. Save this client** (standard library only) as `ttft.py`. Each request asks for one output token, so its
latency is the time to first token; `cached_tokens` says how much of the prompt the server reused.

```python
import json, random, sys, time, urllib.request

URL = "http://127.0.0.1:30001"

def generate(ids):
    body = json.dumps({"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}}).encode()
    req = urllib.request.Request(URL + "/generate", body, {"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        meta = json.loads(r.read())["meta_info"]
    return time.perf_counter() - t, meta["cached_tokens"]

def context(n, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, 100000) for _ in range(n)]

step, n = sys.argv[1], int(sys.argv[2])
ids = context(n, seed=n)
follow = ids + context(64, seed=n + 1)  # the next turn: the same context plus 64 new tokens
if step == "prefill":                   # a context the server has not seen
    print("re-prefill  %.2f s  (cached %d)" % generate(ids))
    print("GPU hit     %.2f s  (cached %d)" % generate(follow))
else:                                   # after a restart the context is only on the SSD
    print("SSD restore %.2f s  (cached %d)" % generate(follow))
    ssd = json.load(urllib.request.urlopen(URL + "/get_server_info"))["scheduler"]["cache"]["ssd"]
    print("restores %d, %.1f GB/s read" % (ssd["restores"], ssd["restore_read_gbps"]))
```

**3. Measure.** The `prefill` step caches the context. The restart moves every cached session to the SSD (the
shutdown writes them, the startup lists them again), so the `restore` step reads the context back from the SSD.

```bash
python3 ttft.py prefill 131072
uv run sharar stop && uv run sharar serve 27b --detach   # returns once the server is ready
python3 ttft.py restore 131072
```

Repeat with other lengths (the tier skips contexts under 4,096 tokens), a fresh length each time so the first
request misses the cache. With `SHARAR_SSD_TRACE=1` the server also writes one line per restore, with its read and
scatter times, to `<SHARAR_SSD_DIR>/v2/restores.jsonl`. `uv run sharar cache` shows what the tier holds.

## Tensor parallel

Each rank stores its own heads of every session on its own Spark, and every rank's tree takes the same steps.
What one rank cannot decide alone, the ranks settle with a minimum over all of them: a write is complete once it
completed on every rank, a restore goes ahead only if every rank read every object intact, and a restart lists
only the sessions every rank stored ([tensor parallel](tensor-parallel.md)).
