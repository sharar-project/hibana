# Phase 4 notes: scheduler

Measured on 2026-09-03 in the dev container with the reference container also
running. Reference numbers are the frozen SGLang matrix (roadmap section 3).

## What exists

- `sharar/engine/request.py`: lifecycle (queued, prefill, running, finished,
  cancelled), timing, stop handling, a streaming cursor.
- `sharar/engine/scheduler.py`: FCFS continuous batching. One `step()` reaps
  finished or cancelled requests, admits queued ones while the running bound
  (4), the queue bound (16) and the pools allow, spends up to
  `prefill_budget` prompt tokens (4,096) on the prefilling requests oldest
  first with at most `chunk_size` (2,048) per request in one forward, then
  runs one decode step for every running request. Retraction under memory
  pressure returns the newest request to the queue. Metrics: admitted,
  finished, cancelled, retractions, rejected, prefill chunks and tokens,
  decode steps and tokens, step time, TTFT percentiles.
- `Executor.prefill(req, max_tokens)` prefills a chunk that continues from the
  state and pages the previous chunk left, the same path as a cache hit.
  `Executor.prefill_batch(chunks)` prefills several requests' chunks in one
  varlen forward: `cu_seqlens` and per-sequence state slots for the GDN
  kernels, one FlashInfer paged-prefill plan over every sequence's prefix and
  new pages, per-sequence checkpoints. `Executor.warmup()` runs each prefill
  path once at load so no request pays backend initialization (the b12x
  GEMM module alone costs 0.5 s on first use).
- `sharar sched-bench` runs the six controlled cells through the scheduler.
- `tests/unit/test_scheduler.py`: the policy against a fake executor with
  simulated time (prefill 2,500 tok/s, decode 85 ms per step): four 8K
  requests, queue and running bounds, cancellation while queued and while
  running, stop tokens, a 31,742 + 1,024 token request under pool pressure
  with retraction, decode never waiting more than one prefill step, and short
  prompts prefilling together.

## Results (`sharar sched-bench`, 256 output tokens, distinct random prompts)

| Input | Conc | Sharar tok/s | Reference tok/s | Sharar TTFT p50 / p95 (s) | Reference TTFT p50 / p95 (s) |
|---:|---:|---:|---:|---|---|
| 128 | 1 | 12.00 | 12.74 | 0.16 / 0.16 | 0.13 / 0.13 |
| 128 | 4 | 44.3 | 46.4 | 0.27 / 0.27 | 0.31 / 0.31 |
| 2,048 | 1 | 11.7 | 12.2 | 0.82 / 0.82 | 0.81 / 0.81 |
| 2,048 | 4 | 38.8 | 40.7 | 3.33 / 3.33 | 2.34 / 3.07 |
| 8,192 | 1 | 10.2 | 10.8 | 3.53 / 3.53 | 3.33 / 3.33 |
| 8,192 | 4 | 27.0 | 18.3 | 14.0 / 14.0 | 8.46 / 31.7 |

Throughput is 5 to 7% below the reference in five cells, which is the decode
FP8 GEMV gap from Phase 2 (scheduler overhead measures under 0.5 ms per step:
82.8 ms per step at 128 x 1 against 82.6 ms for the bare graph replay). The
8,192 x 4 cell is 47% faster with p95 TTFT cut by more than half.

Budget choice, concurrency-4 cells (p95 TTFT, seconds):

| prefill_budget | 2,048 x 4 | 8,192 x 4 |
|---:|---:|---:|
| 2,048 | 3.59 | 15.1 |
| 4,096 | 3.34 | 13.9 |
| 8,192 | 3.72 | 15.3 |

A single 8,192-token forward is slightly slower than four 2,048-token ones,
so the largest budget does not win; 4,096 is the default.

## Decisions

- Prefill chunks and decode steps are separate forwards that interleave. A
  running request's decode stalls for at most one prefill step (up to 4,096
  tokens, about 1.7 s). Folding decode rows into the prefill forward ("mixed
  chunk") would remove that stall at the cost of the CUDA-graph decode path
  for those steps; it stays a listed option, not the default.
- FCFS with a shared budget, oldest first. Fairness between long and short
  prompts is left to Phase 8 measurements.
- Retraction frees the newest request entirely; its cached prefix, if any,
  is reused when it is re-admitted.
