# Calibrated W3 runtime profile and memory staging

2026-09-17. Base source `ddd82af1027060758cf776ff1b62f3b99f2d2936`.
Raw evidence: `${SHARAR_RESULTS}/reviews/2026-09-17-w3-runtime-profile/`.
The [compact manifest](w3-runtime-profile-results.json) identifies the full
results, raw profiles, experimental sources and validation logs.

## Profile

One GB10, calibrated W3 target layers, NVFP4 vocabulary head, step-200 adapted
FP8 DFlash2 draft, k=15, reconstructed accepted states and exact acceptance.
The target manifest is
`9585ba8c682de346e526aeed0e0c9ba5ec34b341d7e8bfb382da0b81f60991b8`.
The container image is
`sha256:c8b70c24bd7d6de8860f751af5463f5af33567c9cbff7c52084d2e6b48633bea`,
with experiment source mounted separately. The frozen family-v3 prompts are
`ring_buffer`, `deadline_queue`, `build_graph` and `stream_parser`, taking the
first B at batch size B. Each uses 10,000 input tokens, cold cache, 60,000 KV
pages, 24 state slots and maximum batch four.

After eight warmup rounds, the driver runs 32 rounds before profiler capture,
30 Nsight Systems rounds, then eight separate event-instrumented rounds.
Loading, graph capture and prefill are outside these windows. GPU busy time
is the union of timeline intervals; phase spans overlap and cannot be summed.

| Batch | Unprofiled round, ms | Profiled round, ms | W3 GEMM, ms | GPU idle, ms | Idle fraction |
|---|---:|---:|---:|---:|---:|
| 1 | 70.68 | 71.31 | 50.67 | 2.00 | 2.80% |
| 2 | 82.19 | 81.29 | 52.70 | 3.68 | 4.52% |
| 4 | 107.79 | 108.15 | 63.22 | 5.94 | 5.49% |

W3 matrix multiplication accounts for about 58–71% of round time. At batch
four the grouped 64-row kernel uses 255 registers per thread. This motivates
occupancy experiments, but register count alone does not establish the cause
of the bottleneck. Systems profiling does not measure achieved DRAM bandwidth.

Each round has one device-to-host copy, taking 1.2–1.6 microseconds on the GPU,
and 14 host-to-device copies totaling 8.8–10.9 microseconds. Most remaining
copies are device-to-device. The host acceptance call waits for preceding GPU
work; its 66–99 ms span is not removable CPU overhead. At batch four, GDN
recurrent/replay kernels total about 11.8 ms and attention about 9.4 ms.

## Unified memory

GB10's CPU and GPU share physical DRAM, but remain separate processors with
ordering and synchronization requirements. NVIDIA documents that ordinary
`cudaMalloc` allocations are not coherently CPU-accessible; shared
communication buffers require an appropriate allocation path. See the
[CUDA porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)
and [optimization guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html).

Sharar already retains the weights, KV cache, recurrent state and draft
features on the GPU. Acceptance counts cross back to Python for stop/budget
handling, state reconstruction and draft feedback scheduling. Keeping those
decisions on the GPU could remove some synchronization gaps; HTTP,
tokenization and request management would still use the CPU. Merely replacing
copies with shared pointers cannot remove the producer/consumer dependency.
The measured idle fraction is an upper bound on savings from removing all
current gaps under these particular fixed-batch workloads.

## Retained change: calibrated weight loading

The safetensors loader previously copied mapped file storage directly to CUDA.
Staging each tensor with a CPU `clone()` before `.to(device)` avoids the slow
direct path on this GB10, matching the existing draft loader. Checksum,
revision, coverage, shape and dtype checks stay in place. Only one projection
is staged at a time; tensor format and inference arithmetic are unchanged.

A largest-projection screen measured median 392.48 ms for direct copying and
4.91 ms with staging, with identical tensor bytes. The full-target ABBA check
used fresh model constructors and warm filesystem/JIT caches:

| Arm | Target construction, seconds |
|---|---:|
| Original, first | 63.981 |
| Staged, first | 11.970 |
| Staged, second | 12.019 |
| Original, second | 62.474 |

The medians are 63.227 and 11.994 seconds: **5.27× faster construction, or
81.03% less time**. All 304 calibrated matrices were fully hashed outside
timing and matched in all four runs. This includes original target loading and
calibrated W3 copies, but excludes drafter loading and graph capture. It is
neither a complete server-startup measurement nor a decode-throughput gain.
The integrated implementation was loaded once more in a fresh process:
14.264 seconds, with the same full tensor hash. That check verifies the code
in the checkout; the repeated 5.27× comparison above uses the controlled ABBA
runs and should not be read as a guarantee for every restart.

## Kernel experiments

Seven alternatives were screened against the original kernel: two warps,
one prefetched buffer, two 32-row tiles, launch bounds requiring three or four
resident blocks, and narrower column tiles with four or two warps. Every
variant used the six unique calibrated target shapes at 16, 32 and 64 rows,
the original split-K policy, and at least 160 MiB of independent weight copies
per shape. Twelve samples alternated forward and reversed variant order.
All tested outputs were bit-identical to the original.

Broad substitutions were not selected: most results were negligible or
regressed, with forced register caps substantially slowing large 64-row
grouped projections. Splitting the attention QKV projection into two 32-row
tiles improved that isolated 64-row shape from 188.37 to 145.49 microseconds.
It occurs only 16 times per round, implying about 0.69 ms before considering
whole-model interactions. The separate full-round screen tests this shape
only; an isolated kernel ratio is not a served speed claim.

The full-round screen ran native, tiled, tiled, native in fresh processes,
with 48 rounds in each of two successive timing windows and no phase probes.
All four prompts' output IDs and per-round emitted counts matched in all
four arms.

| Arm | First window, ms/round | Second window, ms/round |
|---|---:|---:|
| Native, first | 108.469 | 107.737 |
| Tiled, first | 105.034 | 104.692 |
| Tiled, second | 104.880 | 104.789 |
| Native, second | 105.604 | 105.620 |

The native repeat improved by 2.0% in the second window. The candidate's
average second-window speed advantage over the later control is only 0.84%, versus
1.85% over both controls' average. There is no interval across task families
or qualified HTTP gain for this candidate. Keep it as a follow-up experiment;
the serving kernel remains unchanged.

## Rejected scheduling experiment

The current scheduler may issue two prefill forwards between decode steps.
The candidate inserted a decode opportunity and output publication between
those forwards, preserving prompt chunk sizes. Both arms used the calibrated
target and selected drafter through real HTTP, with fixed 100,000 KV pages,
24 slots and batch capacity four. There was one excluded warmup and two
measured repeats for cold, three-turn warm, and mixed requests at concurrency
one, two and four (mixed only at two and four).

Cold/mixed requests emitted 256 tokens; each warm turn emitted 128 tokens.
Warm turns reused actual preceding output plus a controlled 128-token tool
suffix and 0.1-second delay. In mixed cells a fresh 10K prompt arrived before
the cached requests. These are synthetic fixtures, without real tool CPU
load or task-success claims. Both arms completed all 24 cells and passed four
natural-stop quality canaries.

At concurrency two, the cached request's maximum observed stream gap fell
from 1.72 to 0.91 seconds, but aggregate mixed throughput fell from 64.54 to
57.51 tokens/s, about 10.9%. Concurrency-four mixed throughput improved about
3.5%, with stream gaps falling from about 1.75 to 0.94 seconds. Warm
concurrency-four candidate repeats were inconsistent (345.74 and 168.39
tokens/s, versus 336.72 and 348.45 native). This anomaly remains unresolved;
it is retained in the report rather than discarded as noise. The candidate
fails throughput protection and is not integrated. Small request samples do
not establish production tail percentiles.

## Validation and disposition

CPU checks: 539 passed, eight skipped. Source distribution, wheel assets and
installed-runtime probes passed. GPU checks cover calibrated execution,
graph replay, both accepted-state strategies, stops, retraction, prefix/cache
ownership, mixed sampling and draft feedback: 277 passed on the first run.
Four frozen-canary cases initially failed because the isolated container
lacked their reference files. The original fixtures were copied into a
separate test directory; all four cases passed on rerun. Both logs are
retained, for 281 distinct passing GPU checks. The test fixtures include
different draft precisions and widths; the performance comparisons use the
selected FP8 k=15 draft explicitly.

The lowest host `MemAvailable` across the windows was 31.46 GiB, above the
16 GiB experiment floor. Each window restored the original service container
and memory guard. The loader fix and measurement/reporting changes are local
source changes; they have not been activated in the live image or pushed.
There is no qualified steady-state decode-throughput gain from these
experiments. The main decode bottleneck remains W3 matrix multiplication.
