# Memory plan

The GB10 has one physical memory for the OS, every program and the GPU. At startup Hibana sizes its two pools, the
KV pool and the state pool, from what is available, so that everything it allocates leaves the host floor free
(`cache/ledger.py`, `engine/executor.py`). The cache policy decides what the pools hold; see the
[cache design](cache_design.md#glossary) for its terms (KV page, state slot, snapshot, lock, spare).

## Terms

- **Host floor** (`HIBANA_HOST_FLOOR_GIB`, default 6 GiB): the MemAvailable Hibana leaves for the OS and other
  programs: a client or agent on the same Spark, the page cache. The pools get everything else once Hibana's own
  allocations are counted. Preflight also wants the 24 GiB load peak above it.
- **Guard threshold**: half the host floor (3 GiB by default). Under it other programs have used more than the floor
  left them: admission holds (`floor_holds` in `/get_server_info`), and after three samples in a row (6 s) the memory
  guard stops the server (SIGTERM, then a kill 15 s later). `HIBANA_HOST_FLOOR_GIB` moves both; main's
  `HIBANA_MIN_AVAILABLE_KIB` is retired, and the launcher refuses it.
- **Pools**: the KV pool (KV pages) and the state pool (state slots), allocated once at startup.
- **Minimum pools**: KV pages for one request at the maximum context, or for `HIBANA_MAX_BATCH` requests sharing 32K
  tokens, whichever is more; and the slot floor, 2 x `HIBANA_MAX_BATCH` + 1 state slots, for each running request's
  own slot and the snapshot its lock keeps, plus slot 0, and with the SSD tier one more, for the restore that can lock a
  slot for a queued request. The server does not start without them: with them admission never waits for a slot a
  running request holds, and a prefill's private snapshots, which live only inside its step, take what is left.
- **Slot share**: how the memory beyond the minimum pools splits between state slots and KV pages, by what one cached
  conversation costs: the snapshots the tree keeps for it (`CACHED_SNAPSHOTS` = 3: its message boundary, its prompt's
  last chunk end and its reply end) against its 32K tokens of KV pages. About 17% state slots with INT8 state, 34%
  with FP32; at least 96 state slots, and never more than 60% of that memory.
- **Post-plan allocations** (`ledger.after_plan_bytes`; `after_plan_gib` in `/get_server_info`; the server log's
  "counted after the pools" line): what Hibana allocates after the pools, counted by size:
  - workspaces: FlashInfer's decode, prefill and verify workspaces, as allocated;
  - graphs: the CUDA graphs for the batch buckets up to `HIBANA_MAX_BATCH` (an estimate from the maximum batch);
  - first use: GPU memory the first prefill allocates outside PyTorch, never freed (measured 2.18 GiB on Flash-Next);
  - activations: blocks PyTorch keeps cached after warm-up beyond what is allocated.

  The drafter's budget (its weights, its KV per page, the verify buffers) and the SSD tier's staging are counted the
  same way.
- **Headroom rule**: the pools plus the post-plan allocations must leave the host floor. An explicit
  `HIBANA_KV_POOL_TOKENS` is held to the same rule, and the server refuses to start with an error naming it otherwise.
- **`memory_gib`** (`/get_server_info`): PyTorch's allocated and reserved bytes, now and at their peak, and the
  process's resident memory.
- **Retired**: the prefill reserve (a flat 8 GiB) and the plan margin (a flat 4 GiB, or half of what was left): both
  stood in for the post-plan allocations, now counted. Also the 2^20-token KV cap and the hard-coded 16 GiB floor.

## Measured

One Spark (spark-ec65, 117 GiB available), Flash-Next, default `--throughput` (max batch 128, INT8 state, SSD tier
on, FP8 KV: measured before NVFP4 KV became the default, which holds more pages in the same memory):

| | Counted | Actual |
|---|---|---|
| Workspaces | 0.88 GiB | 0.90 GiB |
| CUDA graphs | 3.54 GiB | 3.05 GiB |
| Activations | 1.00 GiB | 1.49 GiB |
| First use | 2.25 GiB | 2.18 GiB |

The pools came to 1,252,935 KV pages (21.5 GiB) and 347 state slots (14.7 GiB); main's plan, with its 16 GiB floor,
gives 607,850 and 257. MemAvailable was 6.43 GiB after the first prefill: just above the floor, as planned. With the
benchmark client (AIPerf) on the same Spark and 35 requests primed and decoding at 32K tokens, or 126 at 8K, it fell to
4.06 and 3.60 GiB; the guard never acted and admission never held. A heavier client on the same Spark wants a higher
floor (`HIBANA_HOST_FLOOR_GIB=7` or 8).
