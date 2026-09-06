# NVFP4 verification cost and narrow-tile study

The 2026-09-06 study uses real Qwen3.8-27B-NVFP4 inputs and checkpoint weights
on GB10. It separates packing from the GEMMs and tests narrower live MMA tiles.
These are extensions of FlashInfer's installed SM120 kernel, not claims of an
original GEMM algorithm. Both tile variants fail their screen; the measured
dispatch change also falls below its whole-round threshold. Serving is unchanged.
Experimental adapters stay in `bench/`; the SGLang comparison is unchanged.

## Protocol and baseline

The source baseline is `f6af0ef`. The pinned target/draft, DFlash2 k=7, FP8 KV,
FP32 recurrent state, 35,000 KV tokens and 12 state slots match the preceding
verification study. Each guarded run has exclusive GPU use, a 64 GiB cgroup
and host memory-drop budget, a 16 GiB available-memory floor, eight CPU cores
of quota and four OpenMP threads. The validated source-overlay image is
`412c4c2e9a4e1179cad974d72b5f3a1a943c33075a0b39801f0dd774af66751b`;
this is not a fresh dependency-image rebuild.

`bench/nvfp4_cost.py` captures one real eager speculative round after restoring
the same cached prefix. The four cells use contexts 128/8,192 and B=1/4.
Each saves all 130 packed input/scale/output triples outside git: 64 gate/up,
64 down, and two vocabulary projections. The target uses M=8/32; the draft
vocabulary projection uses M=7/28. Checkpoint revision and per-call tensor hashes
bind the subsequent tests to that work. Synthetic random weights are not used.

The isolated family replay gives these CUDA-event medians:

| Context / batch | Gate/up, 64 calls | Down, 64 calls | Vocabulary, 2 calls | Packing, all 130 calls |
|---|---:|---:|---:|---:|
| 128 / 1 | 28.164 ms | 14.570 ms | 6.198 ms | 0.415 ms |
| 128 / 4 | 30.526 ms | 16.186 ms | 6.818 ms | 0.933 ms |
| 8,192 / 1 | 28.201 ms | 14.614 ms | 6.215 ms | 0.419 ms |
| 8,192 / 4 | 30.660 ms | 16.191 ms | 6.822 ms | 0.933 ms |

Packing includes the current fused SiLU/quantizer. Its outputs already use the
required layout; there is no measured separate weight conversion between packing
and GEMM. Separate replay does not establish the same cost in a full model.
Do not add these medians to overlapping phase spans or convert forced acceptance
into useful-answer TPS.

Each graph has two warmups. Repeated-address timings use nine samples of ten
replays. The flushed condition updates a 128 MiB buffer outside each timed
single replay, with nine samples. A family traverses all 64 distinct checkpoint
matrices; repeated addresses do not mean that this working set fits in L2.
The flushed and repeated-address family timings are close. Packing and GEMM
are timed separately; weight/scales and arithmetic within a GEMM overlap.

The three-backend sweep covers early/middle/late layers and both vocabulary
calls in every cell: 96 output comparisons against the captured baseline.
All are byte-identical on these inputs. The sweep suggests cuDNN instead of
b12x for the sampled M=28/32 wide projections. This is a possible dispatch fix,
with a separate whole-round gate; isolated timings cannot promote it.

## Narrow-tile mechanism

The default b12x plan uses a 64-row MMA tile for these wide projections. The
experiment narrows the live tile to 16 rows for M<=16 and 32 rows for M<=32,
reducing MMA warps from eight to two/four. It retains packed FP4 weights and
the original 128x4 scale blocks, without a persistent expanded weight copy.

Direct construction initially fails the upstream scale-layout assertion that
M is divisible by 64. The refinement passes the unchanged full scale-block
extent to that helper while narrowing the live MMA work. The alternative
transposes the operand roles into the supported narrow-N path; row padding
and the contiguous output transpose are included in its measured cost.

The screen was frozen before candidate timings: at least 10% improvement in
a complete projection family and at least 2% projected whole-round improvement,
without more than 1% protected-shape regression. Numerical checks include
finite outputs, relative L2 <=0.001, max error / max reference magnitude <=0.01,
and identical vocabulary argmax. These are exploratory checks; serving would
still require the existing full parity, graph/state, stop/cache/failure and
HTTP gates. A component tolerance does not replace those contracts.

Each variant uses three ABBA blocks per family and for the complete 130-call
sequence. Each of the six measurements per implementation contains five samples
of five CUDA-graph replays. All input, scale and reference hashes are checked
before comparisons. Negative candidates need no serving flag or kernel fork.

| Context / batch | Transposed baseline → candidate, all GEMMs | Refined direct baseline → candidate, all GEMMs |
|---|---:|---:|
| 128 / 1 | 49.593 → 53.176 ms | 49.594 → 52.491 ms |
| 128 / 4 | 54.673 → 56.792 ms | 54.860 → 54.319 ms |
| 8,192 / 1 | 49.807 → 53.504 ms | 49.810 → 52.742 ms |
| 8,192 / 4 | 55.017 → 56.928 ms | 54.893 → 54.411 ms |

Each variant reproduces all **520 captured BF16 outputs byte-for-byte**.
The transposed variant is 3.5–7.4% slower. The refined direct variant is about
5.8–5.9% slower at B=1 and less than 1% faster at B=4. Reducing padded MMA work
does not produce the predicted projection gain on this hardware. Neither
variant proceeds to serving integration or claims a whole-round speedup.

## Whole-round dispatch screen

The separate dispatch hypothesis extends the existing cuDNN threshold from
16 to 32 rows. Above 32, dispatch remains unchanged. It predicts roughly 2%
round savings at B=4 and requires at least 2% in both natural-round context
cells, no more than 1% protected B=1 regression, and independent confirmation
before runtime/HTTP promotion. It is an ordinary dispatch change, not novelty.

`bench/nvfp4_round.py` captures separate target/draft graph sets for both
implementations, then alternates ABBA three times after an ABBA warmup. Its
16 cells cover both contexts/batches with natural acceptance and forced counts
1/4/8: **192 measured rounds and 64 warmups**. Prefix restoration and one-token
prefill occur outside timing. Both paths run the real draft, verifier, state
commit and feedback. Tiny diagnostic copies retain proposal/target IDs for
comparison after timing; they are present in both modes. Natural proposals,
target decisions, anchors, output IDs and committed lengths match throughout.

| Context / batch | Baseline natural round | cuDNN-through-32 natural round | Round reduction |
|---|---:|---:|---:|
| 128 / 1 | 111.017 ms | 111.049 ms | -0.03% |
| 128 / 4 | 128.065 ms | 126.031 ms | 1.59% |
| 8,192 / 1 | 114.733 ms | 115.400 ms | -0.58% |
| 8,192 / 4 | 132.345 ms | 130.633 ms | 1.29% |

The first window fails the 2% threshold in both B=4 natural cells. Forced B=4
rows range from about 1.0–1.7% improvement. One unchanged B=1 forced-count row
varies by -1.93%, illustrating why these small differences need confirmation
before promotion. No second promotion window or full serving gate is run for
this rejected candidate. Keep this evidence for a future combined change; do
not change the threshold after observing a smaller gain.

## Hardware traffic and the next mechanism

Nsight Compute 2025.3.1 collects **16 real GPU kernel launches**: cuDNN and
b12x for the middle-layer gate/up/down and both vocabulary shapes at B=1/4.
NVTX labels bind each counter row to its captured input. Kernel replay flushes
caches; clocks remain uncontrolled and recorded. These profiled durations are
separate from the uninstrumented component and whole-round timings.

The device reports 24 MiB of L2. `lts__d_sectors_fill_sysmem.sum` measures fills
from system memory into L2; multiplying sectors by 32 gives the byte counts
below. NVIDIA distinguishes cache-sector traffic from device-memory counters
in its [profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#quantities).
This counter set contains no `dram__` or `gpu__dram` metrics for this device.
It does **not** establish physical LPDDR bus bytes or a fraction of LPDDR peak.

| Projection | Packed weights + scales | Measured system-memory fills into L2 |
|---|---:|---:|
| Gate/up | 100.270 MB | 100.393–100.592 MB |
| Down | 50.135 MB | 50.382–50.734 MB |
| Vocabulary | 715.162 MB | 715.315–715.899 MB |

MB here is decimal. Across the 16 samples, fills are **1.000–1.012×** one packed
weight-and-scale footprint. This is evidence against a large redundant
system-memory read multiplier in these kernels. The measured fill rates are
178–226 GB/s under profiling; they are cache-fill rates, not whole-model speed
or a demonstrated memory-controller roofline. More efficient latency hiding
or different shapes can still justify a future kernel experiment.

The next priority is useful accepted tokens per target weight traversal and
preserving valuable draft/session state under pressure. Complete the natural
DFlash2/DSpark/repaired-MTP and long-session comparisons before choosing that
bounded implementation. Phase 8 separately measures prepacked loading and
reclaimable cache slabs. Unified physical memory alone does not remove a hot
projection's required weight reads or prove that mapped loading is faster.

## Validation and retained evidence

The final benchmark adapter is checked again against all 520 captured outputs
in eager mode and 32 representative CUDA-graph cases, all exact. The existing
CPU suite passes **315 tests, with six GPU-only skips**. No serving code changed,
so this rejected experiment does not claim a new full-model parity/HTTP gate.
The minimum host `MemAvailable` across the completed measurements is 80.45 GiB.
The exact original serving image, memory guard and health/model/canary checks
are restored after every GPU window.

The [evidence manifest](nvfp4-cost-results.json) records the full external bundle
location and SHA-256; detailed samples, sources and outputs stay outside Git.
Initial direct compilation and three profiler-setup failures are retained and excluded
from performance claims. The successful profiler run uses a temporary cache,
root's counter capability and the actual evidence group; it changes no host
driver policy. The GPU adapters use CuTe/FlashInfer, and the recorded CUDA
kernels verify execution on SM121. An unrelated Triton/FLA import warning in
the restricted profiling environment is retained in the raw log.

The initial sandboxed CPU run stalled in the asynchronous engine tests and was
terminated. The complete run executes outside that network-restricted sandbox;
its CPU/GPU settings are unchanged. These setup failures are not passing tests.

## Reproduction

Use exclusive GPU access and the host memory guard; these commands do not
stop or restore a service themselves:

```bash
python3 -m bench.nvfp4_cost --output /path/to/new-inputs
python3 -m bench.nvfp4_tiles --inputs /path/to/new-inputs \
  --output /path/to/new-transposed --variant transposed
python3 -m bench.nvfp4_tiles --inputs /path/to/new-inputs \
  --output /path/to/new-direct --variant direct --retain-full-scale-block
python3 -m bench.nvfp4_round --output /path/to/new-round-screen
```

Raw tensors, profiler reports, launch commands, memory/power samples and failed
attempts are retained under `$SHARAR_RESULTS/reviews/2026-09-06/nvfp4/`.
