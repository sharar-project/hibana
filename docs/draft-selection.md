# DFlash2 selection on the GPU

The first follow-up to 7D profiles the external draft, then removes a measured
host barrier without changing proposal scores or target execution. This is a
small execution optimization for low-concurrency GB10 serving, not a new draft
algorithm or a unified-memory bandwidth claim. DFlash2 and reconstruction remain
opt-in; the production MTP/full-snapshot profile is unchanged.

## Mechanism and ownership

The old greedy path walked each request and candidate position in Python.
Selecting the next score row with a CUDA scalar predecessor caused repeated
host waits. At B=4/k=7 the two-round trace records 104 stream synchronizations
inside `propose`, plus two at target acceptance. The candidate records none
inside `propose`, retaining the acceptance synchronization: **53 to 1 per round**.
At B=1 this is **14 to 1**. These counts exclude three explicit device
synchronizations used by the two-round profiler itself.

`kernels/draft_selection.py` walks the conditional lattice with one warp per
request. Each chosen candidate rank selects the next predecessor row. It keeps
PyTorch's first-maximum/first-NaN semantics and the publisher's top-k ordering.
The measured kernel takes about 3.5 microseconds. It returns owned token IDs;
candidate/score inputs can remain borrowed CUDA-graph views. All operations run
on the existing ordered CUDA stream. There are no new target-state writes,
allocators, scheduler hooks, weights, precision modes or configuration flags.

The adapter uses it only when every request is greedy. Mixed/sampled requests
retain the exact conditional-law reference path. DSpark retains its Markov
correction path. The runtime change is one small kernel module and four added
adapter lines; proposer/verifier and StateManager boundaries stay intact.

## Initial separate-process comparison

Same GB10, target and DFlash2 revisions as 7D; k=7, reconstruction, CUDA graphs,
35,000 target pages and 12 state slots. Each cell has one warmup and three
measured repetitions of eight rounds, with fresh requests and prefix caching
disabled. Every emitted token ID matches across modes and repetitions.

| Context / batch | Control ms/round | Candidate ms/round | Reduction |
|---|---:|---:|---:|
| 128 / 1 | 115.93 | 110.38 | 4.8% |
| 128 / 4 | 144.22 | 135.99 | 5.7% |
| 8,192 / 1 | 119.50 | 113.13 | 5.3% |
| 8,192 / 4 | 155.18 | 143.51 | 7.5% |

The fresh-model **interleaved confirmation** is the more conservative speed
result. The packaged image alternates the original and candidate proposer
methods on identical weights/graphs, with fresh requests for every sample.
Each cell has three measured candidate/control/control/candidate blocks after
one warmup block: six eight-round samples per method. All output sequences
match, including across methods and repetitions.

| Context / batch | Control ms/round | Candidate ms/round | Reduction |
|---|---:|---:|---:|
| 128 / 1 | 115.02 | 110.80 | 3.7% |
| 128 / 4 | 140.32 | 136.33 | 2.8% |
| 8,192 / 1 | 117.82 | 113.78 | 3.4% |
| 8,192 / 4 | 151.38 | 143.16 | 5.4% |

Use **2.8–5.4% lower round time** for the confirmed local result. The larger
first-run reductions remain pilot observations. The kernel is retained because
selection is exact and both timing experiments improve every tested shape;
this does not close the broader external-draft quality/promotion gates.

Both frozen HTTP pilots completed 78 requests. **All 78 output token sequences
match**, including sampled, mixed-arrival, cancellation and short-budget cases.
There are no request or automatic coding/tool/JSON/reasoning failures. Main
completion medians improve 0.6–4.4%; the short-budget group changes by +0.24%.
This does not establish a 10% end-to-end gain. Initial-token latency is not the
optimized path; its group medians vary roughly -4.6% to +6.9%, excluding the
single cancellation observation.

Assistant review covers all 26 rubric-scored outputs per mode (seven distinct
texts). Each has 22 passes and the same four sampled-chat rubric misses: the
retrieval analogy omits explicit recomputation. No rubric or output changes
were made to manufacture a quality pass. General draft promotion remains open.
Three HTTP repetitions are a pilot, not a population tail-latency study.

## What the profile selects next

At B=4/T=128, target verification is about **96 ms**, the draft graph **21 ms**,
and acceptance roughly **20 ms**. Nested event spans are diagnostic and must
not be added to their parents. Host time includes waiting for earlier queued
GPU work; it is not additional GPU compute time.

- Vocabulary projection costs about **3.3 ms**; top-k only **0.14 ms**. A new
  vocabulary top-k kernel is a poor next use of effort for this profile.
- Accepted feature projection is called four times and totals **4.8 ms**;
  context-KV generation adds **3.4 ms**. A bounded `accepted_batch` hook could
  concatenate only committed feature rows, read the projection weights once
  for the batch, and scatter results to the existing private draft rings.
  The pinned checkpoint has 250 MiB of feature-projection weights plus 100 MiB
  of context-KV weights; the current callbacks present these operands four
  times. This is logical operand volume, not measured physical DRAM traffic.
  Variable accepted lengths, ring wrap and cached endpoint features must stay
  explicit. BF16 GEMM shape changes need an independent numerical/quality gate.
- Reconstruction costs **5.3 ms**, then recurrent/conv commits another
  **5.0 ms**. StateManager-authorized reconstruction directly into validated
  committed slots could remove the final FP32 scratch and its copy. Preserve
  read-before-write tile ownership and validate all counts before mutation;
  require bitwise state/stop/cache tests against the current scratch path.
- Target verification still dominates. For a larger mixed-workload gain, first
  trace prefill/verify projection duplication, then prototype shared projection
  work with separate recurrent operations and commits. The rejected tiny-chunk
  policy is not a reason to add another unmeasured scheduling heuristic.

These are measured priorities, not implemented speed claims. No physical DRAM
counters or fresh SGLang head-to-head are collected here. Equal-total-memory
cache-pressure and competitive comparison gates remain on the roadmap.

## Reproduction and evidence

`bench/draft_profile.py --output NEW_DIRECTORY --workloads` saves uninstrumented
round timing, separate CUDA-event component diagnostics, compressed CUPTI
traces, source hashes and raw HTTP results. Run in the exclusive GB10 harness
with the 16 GiB watchdog and restore serving afterward.

`--compare-proposer PATH_TO_FROZEN_PARALLEL_PY` performs a fresh-model timing
confirmation using the original proposer method and the current one on the
same weights/graphs. Each cell runs candidate/control/control/candidate, one
warmup plus three measured blocks. Every round's output work must match. This
mode deliberately excludes profiler instrumentation and HTTP arrival timing.

The portable measurement summary is [draft-selection-results.json](draft-selection-results.json).
The candidate passed **48 GPU kernel cases** (B=1/2/3/4, widths=1/3/7, BF16/FP32,
eager/graphs; random scores, ties, infinities/NaNs and strided borrowed buffers)
and **10 real-checkpoint cases**. The latter compare reference and device
selection through cold/reused prefixes, padded B=3, mixed sampling and widths
1/3/7, checking output equality, zero invalid-draft fallbacks and complete page/
slot recovery. The CPU regression suite passed **171 tests**, with six skips.

Raw local evidence is under
`/home/yke/sharar-results/reviews/2026-09-05/draft-speed/`, including source
snapshots, commands, logs, memory traces, XML tests, raw outputs and review
records. The standalone candidate image is
`sha256:2277b769b147f0911dea6a7be38c3db048ff6f7f2ae1eab473ecf183fc8803ae`.
The watchdog recorded minimum available memory of 79.33/79.73 GiB for the
initial control/candidate runs and 80.21 GiB for final image validation and
confirmation, above the unchanged 16 GiB floor. The original serving image
and memory guard are restored and checked after the experiments.
