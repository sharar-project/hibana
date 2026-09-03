# Phase 3 notes: memory ledger and radix prefix cache

Measured on 2026-09-03 in the dev container with the reference container also
running (about 34 GB of the 119.7 GiB host).

## What exists

- `sharar/memory/ledger.py`: one budget for the process. It reads
  `MemAvailable`, records weights, pools, workspaces and graphs as they are
  allocated, plans KV pages and state slots so that four requests at the 32K
  profile always fit next to an 8 GiB prefill reserve and the 16 GiB floor,
  and can re-check the floor at any time. Cache capacity is what remains,
  halved, split 40/60 between KV pages and state snapshot slots.
- `sharar/memory/radix.py`: token-ID radix tree over KV pages. Nodes carry an
  optional GDN state snapshot slot (conv + SSM after the node's last token).
  A hybrid model can only resume from a snapshot, so `match()` returns the
  longest cached prefix that ends at one. Insert splits nodes at snapshot
  boundaries; eviction is LRU over unlocked leaves; running requests lock
  their prefix path.
- Snapshots come from three places, all owned by the cache afterwards:
  the finished request's committed slot (state after prompt plus all outputs
  but the last sampled token), a copy taken right after prefill (state after
  the prompt, for the next conversation turn), and checkpoints inside the
  prefill at every 2,048 tokens plus the last 64-token boundary before the
  prompt end (so an identical prompt re-sent later recomputes at most 64
  tokens). Checkpoint SSM states are read from the chunk kernel's per-chunk
  `h` output (`h[0, c]` is the state after `64*c` new tokens); checkpoint conv
  states are the three pre-conv inputs before the boundary.
- `sharar/engine/executor.py`: admit matches the prompt (minus its last token,
  whose logits are always needed), copies the snapshot into the request's
  committed slot, and prefills only the new tokens with FlashInfer's paged
  prefill over prefix and new pages (FP8) and the GDN kernels started from the
  snapshot (`has_initial_state`). Release hands pages and the final state to
  the cache without copying.

## Results (`sharar cache-check`, `sharar soak`)

| Check | Result |
|---|---|
| Checkpoint at 2,048 inside a 4,096 prefill vs fresh 2,048 prefill, layer 0 | SSM max rel 0.0029 (bf16 snapshot), conv exact |
| Same, min SSM cosine over 48 layers | 0.991 (deeper layers see M=4096 vs M=2048 GEMM rounding) |
| Identical 9,642-token prompt re-sent | 9,600 tokens from cache, TTFT 4,434 ms to 158 ms, same first token, 12/12 continuation tokens equal |
| Identical 1,024-token prompt re-sent | 960 from cache, same first token, continuation diverges within the noise floor |
| Multi-turn (24-token prompt, reply, new user turn) | 30 of 49 tokens from cache, TTFT 151 ms |
| Pool accounting after every check | pages in use equal cached tokens; flush returns everything |
| Soak: 30 rounds, 8K prompts, concurrency 1 to 4, partial shared prefixes | 302 evictions, no leak, min MemAvailable 37.9 GiB (floor 16) |

Planned pools in that run: 450,647 KV pages (13.75 GiB) and 50 state slots
(7.17 GiB); slots reached zero free within five rounds while pages never did,
which is why the split now favors slots (about 96 slots on this host).

## Decisions and notes

- Prefix reuse changes attention numerics: cached tokens are read as FP8
  pages where a cold prefill uses bf16 ragged attention. Continuations after a
  hit are therefore different roundings of the same distribution (6/16 equal
  tokens on text1024 versus 16/16 on the short canaries). The reference has
  the same property; section 8 of the roadmap covers why this is expected.
- KV page size stays 1. With snapshots dominating cache cost there is no
  pressure from page granularity yet.
- The last token of a prompt is never served from cache. A snapshot at the
  prompt end serves the next turn; the last-64 checkpoint serves re-sends.
- Small prompts (under 64 tokens) get no interior checkpoint and therefore no
  re-send hit; their prefill costs about 150 ms, so this is not worth a slot.
