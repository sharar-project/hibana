# Three-bit decode weights: the tensor-core kernel, head and draft

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

2026-09-17, main `afab63c` plus the working tree. Follows the
[first 3-bit study](w3-decode.md), whose Triton kernel reached 130 to 187
GB/s at 16 rows and fell to 78 to 112 GB/s at 64, so the 3.25-bit copies
served batch one only (round cost x0.870). This study replaces the kernel
and the storage layout, extends the copies to the lm_head and the parallel
draft, and measures batch one and batch four.

## Kernel

`sharar/kernels/csrc/w3_gemm.cu`, driven by `sharar/kernels/w3.py`.

- **Format.** Eight-level codebook of E4M3 values (default the Lloyd-Max
  levels of a Gaussian scaled to 3.5: 0.406, 1.25, 2.25, 3.5 and their
  negatives; the uniform grid is the alternative), one E4M3 scale per 32
  weights along K, one FP32 multiplier per row: 3.25 bits per weight. The
  Lloyd-Max levels cut the relative error against the uniform grid by 8%
  on Gaussian weights and 12% on heavy-tailed ones (0.180 vs 0.195, 0.204
  vs 0.232, CPU test).
- **Layout.** Codes are stored as bit planes in the order the `mma.sync`
  m16n8k32 B fragment consumes them: a warp owns 32 output rows and streams
  128-k chunks as three 16-byte loads per lane, plus one 32-byte scale load.
  The bit for plane b of the j-th weight of sub-step s sits at position
  (4j + b + rot(s)) mod 32, so a lane turns four codes into the four E4M3
  bytes of a fragment register with three rotates, three masked ORs and one
  PRMT byte lookup into the codebook: under two integer instructions per
  weight, against roughly ten in the Triton kernel.
- **Activations.** E4M3 in the tensor core: the engine's static-scale FP8
  projection inputs as they are; BF16 MLP inputs quantized per row by
  `quant_rows` (2.7% relative error on Gaussian rows, finer than the FP4
  block quantization the NVFP4 path applies to the same tensors). Each
  32-k group's product is scaled once in FP32 with the scale of the
  accumulator's own column (the first version indexed the scale by the
  B-fragment row and was wrong for seven rows in eight).
- **Streaming.** Evict-first weight loads, two chunks in flight per warp,
  activations from L1/L2 with 4-byte loads, no shared memory. Narrow shapes
  split K across blocks and the last block of a column reduces the FP32
  partials in the same launch (`SPLITS` table from the sweep below). 84,
  146, 190 and 183 registers for 16, 32, 48 and 64 rows, no spills.

Kernel bandwidth per decode shape (`bench/w3_bench.py`: random weights,
weight copies rotated past L2, ten launches per CUDA graph, median of 15
replays; error against the FP32 reference on the same E4M3 activations
0.17% at every shape and row count):

| Shape | Bytes | 16 rows | 32 rows | 64 rows | Triton 16 / 64 rows |
|---|---:|---:|---:|---:|---:|
| MLP gate_up 34816x5120 | 72.5 MB | 301 us, 241 GB/s | 303 us, 240 | 319 us, 228 | 388 us, 187 / 646 us, 112 |
| MLP down 5120x17408 | 36.2 MB | 155 us, 235 | 162 us, 223 | 180 us, 202 | 232 us, 156 / 377 us, 96 |
| GDN qkv 10240x5120 | 21.3 MB | 88 us, 243 | 89 us, 240 | 100 us, 213 | 134 us, 160 / 264 us, 81 |
| GDN z, attention o 6144x5120 | 12.8 MB | 57 us, 226 | 59 us, 216 | 72 us, 179 | 99 us, 130 / 119 us, 107 |
| GDN out 5120x6144 | 12.8 MB | 59 us, 218 | 62 us, 206 | 75 us, 170 | 92 us, 140 / 139 us, 92 |
| Attention qkv 14336x5120 | 29.9 MB | 123 us, 243 | 125 us, 239 | 191 us, 156 | 169 us, 177 / 300 us, 100 |
| lm_head 248320x5120 | 517 MB | 2300 us, 225 | 2322 us, 223 | 2508 us, 206 | not built |
| draft q_proj 4096x5120 | 8.5 MB | 40 us, 214 | 44 us, 196 | 63 us, 136 | not built |
| draft o_proj 5120x4096 | 8.5 MB | 41 us, 210 | 44 us, 194 | 57 us, 151 | not built |

The measured streaming ceiling of the GB10 is about 239 GB/s. At 16 rows
every target shape is at 91 to 100% of it; at 64 rows the wide shapes stay
above 200 GB/s and the narrow ones drop to 156 to 180 (tail effects at 112
blocks for attention qkv, eight warps per SM at 183 registers). A
launch-bounds variant that forces three blocks per SM spills; left as is.

## Round profile

`bench/gb10_round_profile` (20 rounds after 8 skipped, FP8 draft, the
frozen family inputs), one exclusive window per arm:

| Arm | Batch one | Batch four |
|---|---:|---:|
| Native (NVFP4 MLPs, FP8 projections) | 104.7 ms (2026-09-16) | 139.1 ms |
| 3-bit layers, Triton kernel (2026-09-16) | 91.3 ms | path off |
| 3-bit layers, CUDA kernel | 75.1 ms (x0.72) | 115.4 ms (x0.83) |
| + lm_head (`SHARAR_W3_HEAD=1`) | 72.8 ms (x0.70) | |
| + parallel draft (`SHARAR_W3_DRAFT=1`) | **68.8 ms (x0.66)** | **106.3 ms (x0.76)** |

Resident copies: 9.22 GiB for the layers, 0.48 GiB for the head, 0.59 GiB
for the draft. The batch-four remainder (about 60 ms) is attention, the
recurrent verify kernel, the draft's other matrices and the accept path,
which the profile of 2026-09-15 measured; the 3-bit GEMMs themselves are
now about 48 ms of it.

## Family gate (batch one)

`tools/family-gate.py`, `family-eval-v3` (101 families plus four benchmark
prompts, 10K in), FP8 draft, each arm in its own exclusive window; paired
per-task log ratios against the first native arm, 95% intervals. The first
run (native, layers, then the head arm) hit the memory guard while packing
the lm_head with the whole-matrix packer; the packer now works in row
chunks and the remaining arms ran as a second gate on the same inputs.

| Arm | Round cost | Tokens per round | Decode TPS | Mean round | Mean decode |
|---|---:|---:|---:|---:|---:|
| Native (first arm) | 1.000 | 1.000 | 1.000 | 105.4 ms | 47.8 |
| 3-bit layers | 0.713 [0.712, 0.713] | 0.982 [0.957, 1.008] | 1.378 [1.343, 1.414] | 75.1 ms | 65.4 |
| + lm_head | 0.691 [0.690, 0.691] | 0.968 [0.942, 0.994] | 1.401 [1.364, 1.440] | 72.8 ms | 66.6 |
| + lm_head + draft | 0.649 [0.648, 0.649] | 0.965 [0.937, 0.994] | **1.488 [1.444, 1.532]** | 68.3 ms | 70.8 |
| Native (last arm) | 0.998 | 1.000 | 1.002 | 105.2 ms | 47.8 |

Round cost resolves to a tenth of a percent and the two native arms agree
on tokens per round exactly (the noise floor). The layers alone leave
acceptance inside the band (x0.982, interval containing 1); the head and
the draft each cost about 1.5% of tokens per round, just resolvable, for a
further 8% of decode throughput. One family stopped at EOS early in every
arm including native, none extra under 3-bit (the first study's Triton
build had six). The benchmark prompts no longer match the SGLang reference
transcripts under any 3-bit arm, as expected from changed numerics; the
native arms match three of four.

## Batch-one and batch-four replays

`bench/draft_ab` on the 29-request coding test set (`traces-test`, held out
from drafter training), production (native, main `658a01f` image) against a
window serving the working tree with every 3-bit copy on. Aggregate TPS is
the set's completion tokens over wall time; the paired median is the
per-request TPS ratio.

| Cell | Native TPS | 3-bit all TPS | Aggregate | Paired median | Tokens per round |
|---|---:|---:|---:|---:|---:|
| Batch one, exact | 42.3 | 66.8 | x1.579 | x1.534 | 4.82 to 4.72 (x0.979) |
| Batch one, relaxed default (top-2, 1.0) | 46.2 | 75.4 | x1.633 | x1.539 | 5.29 to 5.43 (x1.027) |
| Batch four, exact | 124.9 | 159.1 | x1.273 | x1.240 | 17.84 to 16.97 (x0.951) |
| Batch four, relaxed default | 129.8 | 181.1 | x1.395 | x1.324 | 19.62 to 20.39 (x1.039) |

Batch four with the relaxed default reaches 181 tokens per second on this
set (45 per request), against 130 native. The relaxed share of emitted
tokens is 3.4 to 3.9%, as on native.

## Task accuracy and the activation-scaling fix

`bench/accuracy_eval.py`, exact acceptance pinned, GSM8K 300 items and
HumanEval 164, McNemar-paired against the exact arm of 2026-09-16 (95.0%
and 90.9%). The first build's 95.7% and 91.5% are void: the harness keeps
four requests in flight, that kernel served 16-row calls only, and at four
requests the verify calls have 64 rows, so it measured the native path
(its wall time matched exact's). This build serves 64 rows, so these are
the first real accuracy numbers for 3-bit decode weights.

| Arm (this build) | GSM8K | HumanEval | Notes |
|---|---:|---:|---|
| Layers, Lloyd codebook, per-row FP8 MLP inputs | 93.3% (3/8, p 0.23) | 89.6% (4/6, p 0.75) | first version of this kernel |
| + lm_head | 93.7% (3/7, p 0.34) | 87.8% (4/9, p 0.27) | the head adds nothing detectable |
| + lm_head + draft | 93.7% (2/6, p 0.29) | 88.4% (4/8, p 0.39) | the draft cannot change exact outputs |
| Layers, uniform codebook, per-row FP8 | 90.7% (3/16, p 0.00) | 82.3% (4/18, p 0.00) | per-row scaling fails badly with the coarser grid |
| Layers, Lloyd codebook, per-group FP8 | 93.0% (2/8, p 0.11) | 90.2% (5/6, p 1.00) | shipped: `quant_groups`, one E4M3 scale per row and 32 channels |

| Layers, uniform codebook, per-group FP8 | 90.3% (2/16, p 0.00) | 86.0% (7/15, p 0.13) | the codebook, not the activations, is the larger term |

Two findings. Per-32-group activation scales (applied in the kernel with
the weight group scale, no bandwidth cost) restore HumanEval to exact for
the Lloyd codebook and cut the activation error on outlier rows from 2.0%
to 0.3% in the bench; per-row scales are not acceptable. And
round-to-nearest 3-bit weights cost accuracy on their own: about five
GSM8K and HumanEval points with the uniform grid, about two GSM8K points
(HumanEval within noise) with the Lloyd-Max levels. The runs of this build
are near-identical greedy outputs, so the two-point reading is one
measurement at p about 0.1; a full-size GSM8K (1,319 items, paired) on
exact and on the all-on configuration decides the launcher default, and
calibration (GPTQ or QuIP#-style from the FP8 release) is the path to a
lossless default. Until then the flags stay opt-in. The batch-one and batch-four replays above ran with per-row
scaling; the group-scaled kernel streams at the same bandwidth, so the
throughput ratios stand and are re-measured with the full-size run.

## Evidence

`$SHARAR_RESULTS/reviews/2026-09-17-w3cuda/`: `w3-bench.json` and
`bench.log` (kernel sweep), `q-*.json` (round profiles with their window
commands and logs), `gate-w3cuda/`, `ab/`, `accuracy/`, the window scripts,
`debug_w3*.py` (the isolation tests that found the scale-indexing bug); the combined `results.json` is named in the [manifest](w3-cuda-results.json).
