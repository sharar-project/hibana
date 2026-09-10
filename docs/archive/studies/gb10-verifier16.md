# Current 16-row GB10 engine study

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

The best combined matrix candidate measured **1.014× decode speed** on the
frozen code 10K/1K workload, with identical output IDs and accepted work.
This is far short of the requested **1.5×–2×** improvement. The candidates
remain in benchmarks; serving dispatch and production are unchanged.

This study starts from `ad58fd7`, already pushed to `main`, and keeps the
target checkpoint, FP8 DFlash2 draft, 16-row verification and accepted-state
reconstruction fixed. The [preceding profile](gb10-fp8-profile.md) owns the
AIPerf/SGLang comparison. New timings below are native executor measurements,
not new HTTP or SGLang results.

## Exact matrix candidates

Nine captures cover code, design and reasoning near output positions 32, 384
and 768. Instrumentation clones operands inside the existing verification
graph. All three complete 1,000-token continuations match uninstrumented
references. Each capture contains 176 FP8 and 129 NVFP4 target operations.

| Candidate | Validation | Complete projection-sequence timing |
|---|---|---|
| cuBLAS FP8 for attention QKV, M=16, N=14336, K=5120 | All 144 captured attention-QKV matrices exact | All 176 FP8 calls: 34.087 → 33.490 ms, 1.75% reduction |
| CuTe FP4 with M tile 64, N tile 128, K tile 512 for MLPs | All 1,152 captured MLP matrices exact | All 128 MLP calls: 43.867 → 42.974 ms, 2.04% reduction |

These sequence timings stream every distinct weight matrix in order. They
are not sums of isolated medians. Five ABBA blocks use five flushed replays
per entry. They still exclude other verifier work and drafting.

The QKV-only full-generation comparison uses one loaded model and two
independent graph sets, one complete warmup per mode and an ABBA measured
sequence per workload. All output IDs and per-round emitted counts match.

| Workload | Baseline decode TPS | QKV candidate | Decode gain |
|---|---:|---:|---:|
| Code, B1 | 44.334 | 44.586 | 0.57% |
| Design, B1 | 34.354 | 34.521 | 0.49% |
| Reasoning, B1 | 56.602 | 56.877 | 0.49% |
| Code, B4 aggregate | 107.083 | 107.039 | −0.04% |

The combined QKV/MLP code pilot repeats the same warmup/ABBA protocol in a
fresh load. Decode rises **44.178 → 44.800 TPS**; total latency falls
**26.589 → 26.269 s**, or **1.012× end-to-end speed**. All six full
generations retain the same IDs and emitted work. Combined design/reasoning,
B4, independent-load confirmation and HTTP validation have not been run.
These small samples do not establish latency tails or wider task performance.

## Counter evidence and rejected alternatives

Nsight Compute 2025.3.1 records eight real-operand kernel calls with cache
flushing and unmodified GPU clocks. Profiling uses an isolated container with
the required counter capability; the host driver policy stays unchanged.

| Matrix | L2 system-memory fills | Profiled duration |
|---|---:|---:|
| GDN QKV | 52.54 MB | 0.260 ms |
| GDN Z | 31.57 MB | 0.180 ms |
| GDN output | 31.59 MB | 0.176 ms |
| MLP gate/up | 100.40 MB | 0.498 ms |
| MLP down | 50.46 MB | 0.258 ms |
| Attention QKV, PyTorch | 73.60 MB | 0.442 ms |
| Attention QKV, cuBLAS candidate | 73.59 MB | 0.350 ms |
| Vocabulary | 715.32 MB | 3.167 ms |

Each fill count is close to one weight footprint, including block scales
for FP4. These are `lts__d_sectors_fill_sysmem.sum × 32`, **not physical
LPDDR byte counters**. Nsight durations are diagnostic, not replacement
generation timings. Low reported occupancy alone does not quantify removable
work in kernels that dedicate warps to asynchronous memory transfers.

The attention candidate replaces the sampled SM89-family cuBLAS kernel with
an SM121 NvJet kernel. Its long-scoreboard stall ratio falls, while weight
traffic stays essentially unchanged. This supports a scheduling/latency
improvement, not a reduction in the model's weight bytes.

Other screens:

- Switching every FP8 projection to FlashInfer cuBLAS regresses GDN QKV.
  Expanding cuBLASLt custom options tests **940 supported tactics** across
  four representative shapes, without finding a larger exact-output gain.
- Eight FP4 tile/pipeline/scheduler configurations are tested on gate/up,
  down and vocabulary. Default b12x is slower than CUTLASS. K=64 is much
  slower; K=512 provides the small MLP win above. Vocabulary retains CUTLASS.
- Bit-based and hardware FP4 encoding both match **65,322** test values,
  covering every finite BF16 value and FP32 neighbors of quantization ties.
  Captured MLP operands also preserve packed bytes and scales. Neither gives
  a material quantizer improvement. The 2,048-row microbenchmarks repeat
  captured verification rows; they are not real prefill operand captures.
  Removing byte-encoding comparisons does not remove the larger surrounding
  activation traffic and SiLU work.

The first counter attempt failed before capture due to output permissions.
The retry collected a valid report with profiler exit zero, then failed its
final host-directory chmod. Both service restorations passed; the retained
cleanup note distinguishes that wrapper error from profiler failure.

## Spending verifier rows differently

The existing conditional candidate lattice is also screened without training.
One rule retains the entire native greedy chain; another assigns all rows by
descending conditional path probability. Both use draft scores only. Future
target tokens are read solely to evaluate coverage after selection.

| Total rows | Probability-tree coverage gain: code / design / reasoning |
|---|---|
| 16 | 8.53% / 18.54% / 8.82% |
| 32 | 19.28% / 32.02% / 18.97% |
| 64 | 24.90% / 41.74% / 24.65% |

The protected-chain 64-row rule gives 24.90% / 41.24% / 24.44% extra
coverage. At 128 rows it gives 29.62% / 50.35% / 33.57%. Neither screen
passes its prospectively recorded per-task advancement threshold. These
are fixed-native-history counterfactuals, not alternative rollout results.

An additional cost probe shares immutable committed history among flat
16-token paths and gives each path a private writable KV tail. At the three
continuation positions, median verifier wall times are approximately:

| Total target rows | Verifier time |
|---|---:|
| 16 | 89.3–89.9 ms |
| 32 | 93.4–94.3 ms |
| 64 | 105.1–106.4 ms |

All committed KV/SSM/convolution hashes remain unchanged. However, the first
path's argmax changes across physical batch shapes: three positions at 32
rows and two at 64 rows in the early probe; one at 64 rows in the middle
probe. The late probe matches. A real tree needs its own attention/recurrent
execution, winner commit and numerical validation. These flat-path timings
do not include that work or prove a tree speedup.

## Next work toward 1.5×–2×

The current roughly 107 ms round would need to fall to about 71 or 54 ms
at unchanged acceptance. The target projection inventory alone is 17.555 GB:
one read at the specified 273 GB/s takes 64.3 ms, before drafting, attention,
recurrence or other work. This conditional traffic budget explains why a
twofold result needs fewer target passes or fewer transferred weights in
addition to kernel tuning. It is not a measurement of bandwidth saturation.
[NVIDIA hardware specification](https://www.nvidia.com/en-us/products/workstations/dgx-spark/).

1. Evaluate a 32-row verification prototype only with explicit attention
   reduction controls and bounded recurrent branching. Measure real branch
   execution/commit cost before crediting the 19–32% coverage opportunity.
   The prior [numerical investigation](verifier-numerics.md) is relevant:
   changing B/A projection shape alone did not explain its verifier drift.
2. Profile fusion at the producer/consumer boundary in prefill: native
   normalization, quantization and GEMM. Encoding alone has now been tested
   and is insufficient. Preserve the phase-specific rounding established by
   the [target arithmetic repair](gb10-target-fix.md).
3. Keep richer draft supervision and calibrated mixed target precision as
   separate complementary tracks, with held-out task quality and complete
   generation measurements. Their gains cannot simply be multiplied by tree
   coverage or isolated kernel improvements.

The 1.5×–2× goal remains open. All GPU windows restore the original service
and memory guard. Raw commands, captures, counter reports, samples and source
snapshots live under `$SHARAR_RESULTS/reviews/2026-09-09-engine16`.
The [compact evidence manifest](gb10-verifier16-results.json) records their
summary hash; the retained artifact index covers 117 files. Minimum recorded
host `MemAvailable` across all windows was 69.71 GiB.
The CPU suite passes **400 tests, with eight GPU-only skips**. New tree tests
compare the selection rule against exhaustive enumeration of small conditional
trees, and check sibling allocation, deterministic ties and invalid input.
