# Three-bit decode weights: kernel, speed and quality on the GB10

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

2026-09-16, main `afab63c` plus the working tree. Superseded on the kernel side by the [tensor-core CUDA kernel](w3-cuda.md) (2026-09-17), which replaced the Triton kernel and the row-sliced layout described here. The research direction
from the [roadmap](../../../roadmap.md#research-directions-2026-09-15): keep the
NVFP4 and FP8 weights for prefill, add a 3.25-bit copy of every MLP and FP8
projection for decode, and see whether the byte saving (about 7 GB of a
22 GB round) turns into throughput without costing quality or the drafter's
acceptance. Unified memory makes the second copy free to hold (9.2 GiB).

## Format and kernel

`sharar/kernels/w3.py`. Codes of 3 bits, one E4M3 scale per 32 weights along
K plus one FP32 multiplier per row: 3.25 bits per weight (72% of NVFP4's 4.5,
41% of FP8's 8). Bits are sliced along rows: one 32-bit word holds one bit
of 32 consecutive output rows at one k, so each plane of a 32-row block is a
contiguous K-long stream and a program expands a word vector into a
[32 rows, 32 k] code tile with one broadcast shift. Per group of 32 k values
the Triton kernel loads three words per row block, runs a K=32 tensor-core
dot against the BF16 (or static-scale FP8) activation slice and scales the
[M, 32] result by the group's scales, so scales are read once per group.
Enabled by `SHARAR_W3_DECODE=1` for calls of at most 16 rows
(`w3.MAX_ROWS`); wider calls, and all of prefill, keep the existing kernels.
Quantization in this study is round-to-nearest from the served weights
(NVFP4 dequantized for the MLPs, FP8 for the projections); the packer is
where a calibrated source plugs in.

Three kernel iterations on the six decode shapes (`bench/w3_bench.py`,
random weights, error against the FP32 reference 0.2%):

| Kernel | MLP gate/up, 16 rows | MLP down | GDN qkv | z / o_proj | GDN out | attention qkv | at 64 rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| bit planes along K, 3-D tile | 120 GB/s | 100 | 87 | 75 | 72 | 94 | 54 to 87 |
| planes along rows, 2-D tile | 165 | 133 | 108 | 101 | 119 | 145 | 65 to 105 |
| per-group dots, scales once per group | **187** | **156** | **160** | **130** | **140** | **177** | 78 to 112 |

Against the serving kernels' measured times (NVFP4 gate/up 442 us, down 234;
FP8 qkv 242, z/o 159, out 151, attention qkv 338), the third kernel is
1.14x, 1.01x, 1.81x, 1.61x, 1.65x and 2.00x faster at 16 rows: 18 ms less
per batch-one round on these projections, out of 78. At 64 rows it is
compute-bound (the expansion costs as much as four times the dot work
buys) and would lose 12 ms per batch-four round, so the path is limited to
16 rows. Closing the gap to the ceiling (187 of 239 GB/s, and the 64-row
case) needs a CUDA kernel with `mma.sync` fragments built directly from the
planes; the Triton expansion is the limit, not the memory system.

## Quantizer error, before any calibration

On real matrices (CPU, against the FP8 release as a near-lossless reference):

| Matrix | served today vs release | 3-bit RTN from served | 3-bit with per-group scale search | 3-bit from the release |
|---|---:|---:|---:|---:|
| layer 12 MLP down (NVFP4 today) | 9.9% | 21.3% | 19.6% | 17.3% |
| layer 12 GDN out (FP8 today) | 3.8% | 20.0% | 17.7% | 17.4% |
| layer 11 attention q (FP8 today) | 3.8% | 20.0% | 17.7% | 17.3% |

An MSE-searched scale removes about a tenth of the error, a Lloyd-Max
codebook another tenth (16.1% on Gaussian-like matrices); none of that is
transformative. Three bits per weight is two to five times the relative
error the model is served with today, and only a calibration-aware method
(GPTQ, QuIP#-style incoherence, or a short QAT pass) changes that order of
magnitude in output space.

## Quick round-time check (20 rounds, `bench/gb10_round_profile.py`)

| Cell | Round | Tokens per round in the window |
|---|---:|---:|
| batch one, today | 104.7 ms | 3.75 |
| batch one, 3-bit decode | **91.3 ms (−12.7%)** | 2.8 |
| batch four, 3-bit build (path off at 64 rows) | 136.9 ms (134.4 before) | 17.0 |

The round-cost gain is real. The tokens-per-round column on one prompt is a
warning, not a measurement: the family gate below is the measurement.

## Family gate, batch one

`tools/family-gate.py --set family-eval-v3 --order N,C1,N` with
`bench.gate_hooks:w3_decode` as the candidate: three passes of 105 tasks,
all ok, one exclusive window.

| 101 families, paired | Decode TPS | Round cost | Tokens per round | Identical outputs |
|---|---|---|---|---:|
| Native repeat vs native (noise) | x0.9992 [0.9989, 0.9995] | x1.0008 | x1.0000 | 105 / 105 |
| **3-bit decode vs native** | **x1.174 [1.137, 1.214]** | **x0.870 [0.8699, 0.8705]** | x1.021 [0.989, 1.057] | 0 / 101 |

Family means: native 47.9 TPS, 5.05 tokens per round, 105.4 ms per round;
3-bit 56.0 TPS, 5.14 tokens per round, 91.7 ms per round. The round-cost
saving is exactly the microbenchmark's prediction (13.0% of the round; the
head and the draft are untouched). Accepted tokens per round did not fall:
+2.1% with an interval of [−1.1%, +5.7%], so the drafter agrees with the
3-bit target at least as often as with the NVFP4 one (code 0.996, design
1.040, reasoning 1.038 by category). The quick check's 2.8 tokens per round
on one prompt was a 20-round fluctuation. Six of the 101 families reached
EOS before 1,000 tokens under the 3-bit target against none under exact
(they are forced to continue, so throughput is unaffected); that is a
behavioral change to keep an eye on, not a measured loss. The four benchmark
prompts moved by tens of percent each way, as expected for a
continuation-changing candidate.

## Judged quality

> **Correction (2026-09-17).** The judged completions and the task-accuracy runs below were generated with four requests in flight, and this kernel served calls of at most 16 rows, so nearly every round ran the native path; they measure almost nothing about 3-bit weights. The [CUDA kernel study](w3-cuda.md) measures the 3-bit target at concurrency four: GSM8K about two points under exact, HumanEval within noise, with the round-to-nearest uniform grid about five points under on both. The batch-one gate results in this study (round cost, acceptance) stand.

The same 53 prompts (29 coding, 24 ShareGPT), natural-length completions
(1,024-token cap) generated by the 3-bit server in a window, judged by the
unmodified production model with exact acceptance against the exact
completions from the relaxed-acceptance study, both presentation orders,
thinking off, correctness first.

| | 3-bit wins both orders | Exact wins both | Tie or split | No verdict |
|---|---:|---:|---:|---:|
| All 53 | 12 | 9 | 31 | 1 |
| Coding 29 | 6 | 4 | 18 | 1 |
| Chat 24 | 6 | 5 | 13 | 0 |

Sign test p = 0.66: no detectable difference at this size. Completion
lengths and the share ending at EOS are the same (836 tokens mean, 21 of
53 finished, against 842 and 22 for exact). As with relaxed acceptance,
this rules out a large quality loss (an exact preference above roughly two
in three would have shown) and cannot see a small one; a same-model judge
may also favor the model's own text.

## Task accuracy

GSM8K (300 problems) 95.7% against 95.0% exact, HumanEval (164) 91.5%
against 90.9%, neither difference significant (paired McNemar p 0.62 and
1.0): [accuracy benchmarks](accuracy-benchmarks.md).

## Decision

- **Batch one: a real 1.17x; the quality claim was void (correction above).** Round cost −13.0%
  exactly, acceptance unchanged within ±3%, with round-to-nearest 3-bit weights and a first Triton
  kernel. Merge as an opt-in (`SHARAR_W3_DECODE=1`, off by default). Making
  it the default is a target change and deserves the same treatment as
  relaxed acceptance: a few hundred judged prompts, and a look at the early
  EOS behavior.
- **Batch four: nothing yet.** The Triton kernel is compute-bound at 64 rows
  and would lose 12 ms per round, so the path stays off there. The bytes
  are the same 7 GB of a 27 GB round, about 20 ms if a kernel reached the
  16-row kernels' rate: a CUDA kernel building `mma.sync` fragments from the
  planes is the next piece of work, and it would also lift batch one from
  187 toward 230 GB/s (another 3 to 4 ms).
- **Quality headroom is untouched.** Round-to-nearest at 17 to 20% weight
  error already passes the judge; GPTQ or QuIP#-style calibration from the
  FP8 release (in the local cache, 128x128 block scales) would cut the
  output error several-fold at the same byte count and is the natural
  follow-up before a default. The drafter can also be retrained on traces
  served by the 3-bit target.
- **Extending the format** to the vocabulary head (0.7 GB, 3 ms) and the
  draft (2.5 GB, about 10 ms) is straightforward and would take batch one
  toward 1.3x.

## Evidence

`$SHARAR_RESULTS/reviews/2026-09-16-w3/`: `w3-bench.json` (kernel sweeps),
`q-*.json` (quick round checks), `gate-w3/` (three passes, comparison JSONs),
`w3-completions.json`, `w3-judged.json`, the window scripts, and the
combined `results.json` named in the [manifest](w3-decode-results.json).
