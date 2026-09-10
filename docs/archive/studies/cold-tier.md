# Wider DFlash2 blocks and the unified-memory cold tier

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

Date: 2026-09-07. Source baseline `2117751` (DFlash2 k=7 default).

Both changes follow the same reading of the GB10 numbers: a verify round is a
pass over 19 GB of weights whatever the row count, so speed comes from
accepted tokens per pass, and the shared DRAM is a place to keep whole
sessions, not a faster bus.

## 1. Sixteen verify rows per round (k=15)

The DFlash2 checkpoint drafts a block of 7 proposals from one anchor
(`block_size: 8`). The verifier does not care how many rows it checks: 16
rows cost 2 to 3 ms more than 8 on this GB10 (`docs/two-path-study.md`,
88.1 to 90.2 ms at 8K context). The question was whether the draft, asked for
15 proposals in one block, produces anything useful beyond position 7.

**Offline test** (`bench` script `wide_block.py` in the results directory):
during a normal k=7 generation, every round also drafted a 16-row block from
the same draft state into scratch caches, and both proposals were scored
afterwards against the greedy trajectory (greedy acceptance makes the emitted
sequence the target's own argmax path, so the truth is known exactly).

| Prompt | Rounds | Accepted per round, 8-block | Accepted per round, 16-block | Rounds where wider won |
|---|---:|---:|---:|---:|
| chat | 32 | 1.19 | 1.25 | 3 |
| coding | 5 | 5.80 | 9.00 | 3 |
| tool call | 6 | 5.50 | 7.33 | 3 |
| JSON | 4 | 6.50 | 9.50 | 3 |
| reasoning | 2 | 0.50 | 0.50 | 0 |
| session | 20 | 2.10 | 2.10 | 0 |
| text1024 (filler) | 60 | 3.27 | 4.77 | 21 |
| rand128 (filler) | 25 | 4.12 | 7.36 | 14 |
| all | 154 | 3.04 | 4.42 | |

The draft's positions 8 to 15 are useful exactly when positions 1 to 7 were
all accepted, which is the case on structured text; on hard text the wide
block accepts the same as the narrow one, so nothing is lost.

**End to end** (dev container, small pools, same prompts, 256-token outputs,
`k_sweep.py`):

| Prompt | k=7 tok/s | k=15 tok/s | tokens/round k=7 | tokens/round k=15 |
|---|---:|---:|---:|---:|
| chat | 20.3 | 19.9 | 2.16 | 2.16 |
| coding | 64.8 | 79.0 | 6.80 | 8.50 |
| tool call | 60.0 | 70.2 | 6.33 | 7.60 |
| JSON | 69.4 | 90.4 | 7.25 | 9.67 |
| session | 28.7 | 28.2 | 3.05 | 3.05 |
| text1024 | 39.4 | 44.2 | 4.25 | 4.90 |
| rand128 | 57.6 | 86.1 | 6.22 | 9.44 |
| all 751 tokens | 40.8 | 46.8 | | |

Round time 108 to 110 ms. Outputs are identical in both settings. k=15 is
now the DFlash2 default (`proposers/factory.py` `MAX_DEPTHS`, the adapter's
`max_depth`, `run-sharar.sh`); an adaptive width was not needed because the
extra rows are nearly free.

## 2. Cold tier: parking evicted sessions in unified memory

### Measurements that shaped it (`um_bench.py`, `managed_tier_bench.py`, `pin_bench.py`)

| Access | Bandwidth |
|---|---:|
| GPU kernel reading device memory (2 GiB sum) | 240 GB/s |
| GPU kernel reading `cudaMallocManaged` memory | 163 GB/s |
| `copy_` device to managed / managed to device | 100 / 108 GB/s |
| `index_select` of KV pages straight into managed memory | 93 GB/s |
| `cudaMemcpy` device to pinned host or back (the DMA path) | 53 GB/s |
| FlashInfer paged decode attention over device KV, 8K tokens | 46 us per layer |
| the same over managed KV | 146 us per layer |
| managed first touch | ~330 ms per GiB |

So: managed memory is not a place to run attention (3x slower), the copy
engines are the slow path on GB10 (53 GB/s), and GPU kernels moving data
between device and managed memory are the fast path. That gives a tier that
parks and restores whole sessions with gather/scatter kernels, touched once
at startup.

### What it does (`memory/cold_tier.py`)

- When the radix cache evicts a leaf that has a GDN snapshot, the leaf's KV
  pages (all 16 attention layers) and its conv/SSM snapshot are gathered into
  a managed arena before the pages and slot are freed; the draft adapter's
  auxiliary state (the projected feature window, via a codec it registers)
  travels with it. 30K tokens: about 1 GB of KV plus 151 MB of state, about
  12 ms.
- On admission, before the radix match, every parked entry that is a prefix
  of the request (shortest first, so a parked parent comes back before its
  parked child) is restored into freshly allocated pages and a slot and
  re-inserted with its snapshot and auxiliary. The request then matches as if
  the eviction had never happened. About 11 ms for 30K tokens.
- The arena is LRU-dropped when full; a flush clears it. Budget:
  `SHARAR_COLD_TIER_GIB`, else what is spare beyond the host floor, the
  prefill reserve and a 4 GiB margin, capped at 24 GiB.

The tier never serves attention from managed memory and owns nothing of the
pools. Proof: `tests/gpu/test_cold_tier.py` (bit-exact round trip of pages,
state and auxiliary through a scribbled pool; parent-before-child restore;
LRU drop; flush).

### Results

**Eviction round trip** (dev container, 70,000 KV pages, 8 slots, DFlash2
k=15; a 30,000-token session, then two other 30,000-token sessions that push
it out, then the first session's next turn: its full history plus a 64-token
user turn; `tier_ab.py`):

| | Tier off | Tier on (4 GiB arena) |
|---|---:|---:|
| First turn, cold prefill | 18.2 s | 19.6 s |
| Next turn after eviction: prefill | 18.5 s (cached prefix 0) | **0.152 s** (cached prefix 30,007 of 30,072) |
| Restore cost | | 14 ms |
| Park cost per 30K session | | 149 ms |

Park is slower than the raw gather benchmark (about 7 GB/s against 93) and
is the next thing to look at in the tier; it runs on the eviction path, not
on the resumed request's critical path.

**Production** (`sharar-qwen38`, DFlash2 k=15, reconstruction, 24 GiB tier,
346 in-image checks, `tools/verify-sharar.sh` pass). Six-cell harness,
median aggregate tok/s of five batches (range), 256 outputs, EOS ignored:

| Input x conc | 09-04 MTP k=3 | 09-07 DFlash2 k=7 | 09-07 DFlash2 k=15 + tier | tokens/round k=15 |
|---|---:|---:|---:|---:|
| 128 x 1 | 32.7 | 65.9 | **101.0** (93-102) | 12.2 |
| 128 x 4 | 105.1 | 119.9 | **305.4** (78-317) | 12.2 |
| 2,048 x 1 | 22.8 | 27.7 | 30.1 (22-47) | 3.9 |
| 2,048 x 4 | 61.8 | 61.9 | 64.5 (51-70) | 5.2 |
| 8,192 x 1 | 19.5 | 32.3 | 26.2 (21-47) | 4.4 |
| 8,192 x 4 | 38.5 | 42.3 | 41.0 (37-49) | 5.4 |

Canaries 4/4; minimum MemAvailable 31.6 GiB with the 24 GiB arena resident.
The two 128-token cells are the repetitive filler where the draft accepts
12 of 15; the 2K/8K cells are the harness's "the target starts replying"
prompts where acceptance is 4 to 5 per round and batches vary 2x, so their
medians move with which continuation the target picks (8K x 1 was 32.3 in
the k=7 run and 26.2 here with overlapping ranges).

Frozen local workloads (156 requests, 0 failures, 0 automatic quality
failures), median completion in seconds, k=7 then k=15:

| chat | coding | tool call | JSON | reasoning | session cold / repeat / prior turn |
|---:|---:|---:|---:|---:|---:|
| 3.66 / 3.70 | 0.67 / 0.57 | 0.82 / 0.72 | 0.55 / 0.45 | 1.11 / 1.25 | 3.06 / 2.77, 2.59 / 2.18, 2.37 / 2.41 |

Run `controlled-27b-sharar-dflash2-k15-tier-20260907T041045Z`; evidence
under `${SHARAR_RESULTS}/reviews/2026-09-07e/`.

### Not done, measured and recorded

- A fused NVFP4 GEMM + SiLU-gate + block-scale-quantize epilogue through the
  cuDNN graph API (`cudnn_epilogue4.py`): the plain block-scaled GEMM builds
  and runs bit-exact at 2.77 ms against 2.92 ms for the b12x kernel (a 5%
  lead worth a backend option later), but adding the quantize epilogue
  crashes natively in cuDNN 9.20 / frontend 1.27 on SM121. The 80 ms per 2K
  prefill that the fusion would save stays on the table.
- Mapped pinned host memory as an alternative arena: torch cannot wrap it
  without cupy's pinned pointer type; managed memory is used instead.
