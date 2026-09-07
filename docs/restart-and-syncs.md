# Ten-second restarts, a sync-free sampled path, and the tier's park cost

Date: 2026-09-07 (evening). Source baseline `1f9b0b2` (k=15 default, cold tier).

## 1. Weight load: 108 s to 10.5 s

cProfile of `Qwen38.__init__` on a warm page cache: 105.4 of 108.0 s inside
`Tensor.to` for 1,751 tensors; the safetensors reads themselves took 0.1 s
(they are mmap views). Copying an mmap view to the device page-faults the
file in through the DMA staging path at a few hundred MB/s on GB10. One
tensor measured three ways (`h2d_bench.py`, the 2.4 GB embedding):

| Path | Rate |
|---|---:|
| mmap view -> device | 3.2 GB/s (cold), 5.0 GB/s (warm) |
| mmap view -> RAM (`clone`) | 25 GB/s |
| RAM -> device | 18 GB/s |
| pinned -> device | 16 GB/s, but 1.2 s to pin |

For the small tensors that dominate the count the mmap path is far worse
than 3 GB/s, which is how the whole load averaged 180 MB/s. The fix is one
line in `weights/safetensors.py`: `get()` returns `get_tensor(name).clone()`,
so every tensor is read into RAM by memcpy and then copied at DMA speed.
Full load: **10.5 s** (`Tensor.to` 5.6 s). Production restart, launch to
`/health` ready, is measured in the results below.

## 2. Host synchronizations, by exact call site

The profiler counts runtime events but not who caused them, so
`sync_attrib.py` wraps every tensor call that can block the host
(`tolist`, `item`, `cpu`, pageable `to`/`copy_`, `torch.tensor(...,
device=cuda)`, `torch.cuda.synchronize`) and records the Sharar frame that
called it, for one step of each kind:

| Step | Synchronizing calls | Where |
|---|---:|---|
| Greedy speculative round | 1 | the packed result download in `spec._accept` |
| Sampled round (DFlash2, temperature 0.7) | 1 visible, plus about 40 hidden | 15 per row in the draft's lattice walk (a 0-d CUDA index converted to `int`), and 3 per verified position in the sequential rejection sampler (`bool`/`int` of CUDA scalars) |
| Scheduler step with a 2,048-token prefill chunk | 19 | 5 uploads in the GDN prefill plan, 1 for the chunk ids, the sampled first token, a page free in the cache, and the round's download |

The hidden ones do not show up as `tolist` because they are implicit
conversions (`if tensor >= threshold`, `scores[row, j, local]` with a CUDA
`local`). The profiler saw them: 44 `cudaStreamSynchronize` per sampled step
against 1 per greedy step, 126 ms of host waiting over four steps.

Changes:

- `rejection.verify_rows`: vectorized rejection sampling per request. Same
  acceptance rule and residual law as the sequential reference
  (`verify_row`, kept for the tests), but the k uniform draws are taken at
  once and the correction or bonus token comes from one categorical draw
  over the selected law, so the request's generator is consumed identically
  whatever the outcome and nothing is read back until the round's single
  download. It is equal in distribution to the reference, not bit-identical
  for a fixed seed (`tests/unit/test_rejection.py` checks the distributions
  over 4,000 seeds). Consequence: seeded sampled outputs change once, which
  includes the frozen sampled-chat workload; its rubric was re-checked on
  the new output below.
- The draft's sampled lattice walk keeps its indices as 1-element device
  tensors used with `index_select`, so the 15 steps launch without a host
  round trip each.
- `engine/upload.py`: prefill-side uploads (GDN plan tensors, chunk ids,
  positions, sequence slots, the plain-decode inputs) go through a pinned
  staging tensor with a non-blocking copy instead of `torch.tensor(...,
  device=cuda)`, which waits for the stream to drain before copying.

## 3. Cold tier park: 149 ms to 0.6 ms of host time

The 149 ms measured yesterday was not the copy (11 ms of GPU work for 30K
tokens in isolation): the draft's feature window was exported with a
synchronous device-to-CPU copy, which waited for whatever GPU work was
queued ahead of the eviction. The feature window now lives in the arena
with the pages and state, the codec exchanges device tensors only, and park
issues no synchronization. Rerun of the eviction A/B: cold 20.5 s, resume
0.148 s, restore 15 ms, park 0.6 ms host time; the resumed request
speculates at depth 15 from the restored feature window.

## 4. The 128 x 4 tail: assessed, not engineered

The earlier attribution stands: in a four-request batch of the harness's
repetitive prompt, one request's first token occasionally differs (a
numerical effect of batch position at a low-margin token), that request
then writes free text with 2 to 3 accepted tokens per round, and it keeps
the batch alive for 50 to 80 single-request rounds after the other three
finish. Waiting for all four admissions does not remove it, and forcing the
common continuation would be cheating the benchmark. On real workloads the
equivalent is "one hard request in a batch", which is the batch-1 hard-text
rate. No change; the cell keeps its 2x range in the tables.

## 5. Second pass: the draft load, the arena, and page frees

The production restart timeline (container start to `serving on`) after the
weight-load fix still showed 30 s: imports 2.2 s, target weights 4 s, pools
1.2 s, **DFlash2 draft load 19.3 s**, cuDNN handle and graph capture 2.5 s,
warmup 1.3 s. The draft loader had the same mmap-to-device copy; the same
`clone()` fixes it.

The cold tier's arena was zero-filled from the GPU at startup to avoid page
faults on the eviction path. Measured on a 4 GiB managed block: the fill
takes 3.0 s and leaves the pages reading 5x slower afterwards than pages
touched by `cudaMemPrefetchAsync` (1 ms to issue) or by their first write.
The arena is now built from 2 GiB slabs allocated on demand, each prefetched
to the GPU when created, released when empty, and never grown when host
`MemAvailable` would fall within 4 GiB of the guard's floor. The 24 GiB
budget therefore costs nothing until sessions are parked, and the tier backs
off instead of pushing the host into the memory guard.

Page frees no longer read the device: requests carry the host ids of the
pages they own (`RequestState.page_ids`, appended on admission and on every
published reservation), cache nodes carry the ids of their pages, and the
tier restores through `alloc_with_ids`. Every free on the serving path
(release, retraction, cache insertion duplicates, leaf eviction) is now a
host list operation; `KvPool.free(tensor)` remains for tests only.

## 6. Results of the first pass

- Plain and MTP canaries token-identical to the frozen reference after the
  loader change; **666 CPU/GPU checks** pass in the built image.
- Production restart, `run-sharar.sh` to `/health` ready: **30 s** (weights
  loaded in 4 s with a warm page cache; the rest is draft load, graph
  capture, warmup and the 24 GiB arena touch). It was about 130 s.
- Frozen local workloads: 0 request failures, 0 automatic quality failures.
  Sampled chat (temperature 0.7, seed 123) completes in **3.28 s** against
  3.73 s before the sync-free sampled path; greedy cases unchanged (coding
  0.58, JSON 0.45, tool 0.73, chat 3.74, session cold 2.78 / repeat 2.19).
  The new seeded answer: "Think of the cache as a sticky note on your
  monitor where you write down the most frequently used data, so you don't
  have to look up the main reference book every time. A cache miss simply
  means the specific piece of data you need isn't on that sticky note,
  forcing you to pause and retrieve it from ..." which meets the rubric.
- Evidence: `${SHARAR_RESULTS}/reviews/2026-09-07f/`.

## 7. Results of the second pass

- **666 CPU/GPU checks** pass in the built image; frozen workloads 0
  failures, completion times unchanged (coding 0.57 s, JSON 0.45 s, tool
  0.73 s, chat 3.79 s, sampled chat 3.29 s, session cold 2.84 / repeat 2.20 s).
- Production restart, `run-sharar.sh` to `/health` ready: **12.0 s**
  (weights 4 s, pools 1.2 s, draft 0.5 s, graphs and warmup 4.3 s). The
  first chat request after the restart answers in 0.48 s, the second in
  0.45 s: no page-fault storm from the arena.
- Restart history: 130 s (September 4) -> 30 s (this morning) -> 12 s.
- Cold tier on the slab arena (idle machine, 4 GiB budget): the evicted 30K
  session resumes in **0.147 s** (cold prefill 18.6 s), restore 14.6 ms,
  park 0.47 ms of host time, two slabs allocated on demand. Run beside an
  idle production instance the tier refused to grow (`last_reject: no slab`)
  because host `MemAvailable` was within 4 GiB of the guard's floor, which
  is the intended backoff: the tier never pushes the host into the memory
  guard. `spec_stats`/`cache` now report `last_reject` for that case.
- Relaunch after the A/B: ready in 14.1 s, verify passes.
