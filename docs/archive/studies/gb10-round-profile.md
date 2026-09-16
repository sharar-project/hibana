# One decode round on the GB10, profiled: where the bytes and the time go

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

2026-09-15 and 16, main `658a01f` plus `bench/gb10_round_profile.py`. The
first profile of the current engine with the timeline of every kernel
inside the CUDA graphs, and the first hardware-counter measurement of the
round on this host. Protocol: the family prompt `ring_buffer` (10,000 input
tokens) at batch one and the first four family prompts at batch four; FP8
pinned DFlash2 at k=15, exact acceptance, cold prefix cache; eight rounds to
reach steady state, then a window of twenty verify rounds (timeline) or one
round (counters) bracketed by the CUDA profiler API.

How the counters were obtained, since every earlier study lacked them: the
host driver restricts counters to administrators (`RmProfilingAdminOnly: 1`),
so the profiling container ran as root with `CAP_SYS_ADMIN` and a private
copy of the JIT caches; that alone made the counters readable, no driver
change or reboot. Two further obstacles: Nsight Compute cannot replay
kernels inside captured CUDA graphs (`LaunchFailed`), so the counter pass
runs the engine's eager path (`--eager`); and its default kernel replay
snapshots device memory between passes, which over unified memory with tens
of gigabytes allocated made no progress in an hour, so the pass uses
application replay (the process re-runs once per metric pass, about 7
minutes per batch size). Finally the discrete-GPU `dram__` counters report
nothing on this integrated GPU; system-memory traffic is
`lts__d_sectors_fill_sysmem` (L2 fills from memory) and
`lts__t_sectors_aperture_sysmem_op_write`, 32 bytes per sector. Nsight
Compute locks clocks to base, so its durations run 10 to 15% long; the
tables below take bytes per launch from the counters and durations from the
graph timeline at serving clocks.

## Batch one: 104.7 ms per round, 3.75 tokens per round on this prompt

The GPU is busy 98.4% of the round (1.7 ms idle in four gaps). 1,421
kernels per round, all inside graphs; the median kernel lasts 3.3 us and
67% of launches are under 10 us, but those sum to under 3 ms. The counters
see 22.3 GB of system-memory traffic per round, 20.7 GB of it in the kernel
classes below; over the 104.7 ms round that is 213 GB/s, 89% of the
239 GB/s streaming ceiling.

| Kernel class (bytes per launch measured) | launches | MB per launch | median us | GB/s | of ceiling | ms per round |
|---|---:|---:|---:|---:|---:|---:|
| MLP gate/up, NVFP4, CuTe 16-row kernel | 64 | 101.6 | 442 | 230 | 96% | 28.3 |
| MLP down, NVFP4, CuTe 16-row kernel | 64 | 50.7 | 234 | 217 | 91% | 15.0 |
| GDN qkv projection, FP8, cuBLAS 128x64 | 48 | 53.0 | 242 | 219 | 92% | 11.6 |
| GDN z projection and attention o_proj, FP8, cuBLAS 64x88 | 64 | 31.9 | 159 | 200 | 84% | 10.3 |
| GDN out projection, FP8, cuBLAS 64x88 | 48 | 31.9 | 151 | 212 | 89% | 7.2 |
| vocabulary head, NVFP4 (target and draft) | 2 | 725 | 3,120 | 232 | 97% | 6.3 |
| attention qkv, FP8, cuBLAS 192x48 | 16 | 74.1 | 338 | 220 | 92% | 5.4 |
| draft gate/up and down, FP8 | 10 | 180 / 90 | 813 / 414 | 222 / 217 | 93% / 91% | 6.2 |
| attention, FlashInfer (10K context, FP8 KV) | 16 | 23.4 | 129 | 181 | 76% | 2.1 |
| GDN recurrent verify (`fused_sigmoid_gating`) | 48 | 4.1 | 42 | 96 | 40% | 2.0 |
| accepted-state replay (`_reconstruct`) | 1 | 304 | 1,320 | 230 | 96% | 1.3 |
| draft feature and k/v projections, BF16 | 20 | 13 to 263 | 68 to 1,178 | 198 to 224 | 83 to 94% | 2.5 |
| GDN b/a projection, BF16, 48 tiny GEMMs | 48 | 1.2 | 8.7 | 142 | 59% | 0.4 |

Every large GEMM moves exactly one weight footprint (the NVFP4 gate/up
launch reads 101.6 MB against 100.3 MB of packed weights and scales), so
there is no hidden re-read to remove, and every one streams at 84 to 97% of
the ceiling. The per-kernel SM throughput is 9 to 15% on all of them:
compute is nowhere near binding at 16 rows. The tiny kernels report more
bytes per second than the ceiling because they hit in L2; they cost about
3 ms in total.

What is left at batch one, in ms per round: the 64x88 cuBLAS class at 84%
(about 1.5 ms if it reached 228 GB/s), the out projection at 89% (0.6), the
recurrent verify kernel, which is latency-bound at 96 GB/s and 33% SM
throughput (about 1 ms), attention at 76% (0.5), and the idle gaps (1.7).
**Roughly 5 to 6 ms of 104.7, about 5%.** Nothing else is worth a kernel.

## Batch four: 134.4 ms per round, 17.0 tokens per round for the batch

The GPU is busy 96.7% (4.5 ms idle in 14 gaps). 27.3 GB of memory traffic
per round; the larger classes:

| Kernel class | launches | MB per launch | median us | GB/s | of ceiling | ms per round |
|---|---:|---:|---:|---:|---:|---:|
| MLP gate/up, NVFP4, CUTLASS 64-row path | 64 | about 101 of weights | 472 | about 212 | 89% | 30.2 |
| MLP down, NVFP4, CUTLASS 64-row path | 64 | about 51 | 250 | about 200 | 84% | 16.0 |
| GDN qkv projection, FP8, `sm89_xmma` 64x128 | 48 | 54.2 | 283 | 191 | 80% | 13.7 |
| GDN z projection and attention o_proj, FP8, cuBLAS 64x128 | 69 | 32.2 | 183 | 176 | 74% | 12.2 |
| attention, FlashInfer (4 x 10K context) | 16 | 86.1 | 483 | 178 | 75% | 8.1 |
| GDN out projection, FP8, cuBLAS 64x152 | 48 | 33.0 | 162 | 204 | 85% | 7.9 |
| GDN recurrent verify (`fused_sigmoid_gating`) | 48 | 16.1 | 122 | 132 | 55% | 5.9 |
| accepted-state replay (`_reconstruct`) | 1 | 1,228 | 5,312 | 231 | 97% | 5.3 |

At 64 rows the FP8 projections lose 8 to 12 points of efficiency against
batch one (the z and o_proj class drops to 74%), the NVFP4 MLPs move to the
CUTLASS path and lose 4 to 7 points, the recurrent verify kernel takes
5.9 ms at 55% of the ceiling with 49% SM throughput (it is doing sixteen
sequential steps per row), and attention over four 10K contexts reads
1.4 GB at 75%. The state replay reads and writes 1.2 GB (whole recurrent
slots for four requests) at the ceiling: it is bandwidth-bound, not
latency-bound, and only cheaper by touching less state. Idle time is
4.5 ms, a third of the old 16 ms estimate from short contexts.

What is left at batch four, in ms per round: the FP8 z and o_proj class
(about 3), the recurrent verify kernel (about 2.5), the NVFP4 64-row path
against the 16-row kernel's rate (about 3), attention (about 1.5), qkv
(about 1) and the idle gaps (4.5). **Roughly 15 ms of 134, about 11%.**

## What this decides

- **Kernel work at batch one is capped near 5%,** spread over four small
  items; the NVFP4 GEMMs, the head and most FP8 projections already stream
  at 89 to 97% of the ceiling. A custom 16-row NVFP4 GEMM, a persistent
  decode kernel (bounded by 1.7 ms of idle plus tails) and further work on
  the FP8 projections are not worth doing for batch one. The earlier
  Triton FP8 result (a clean 1.5% on round cost) is consistent with this:
  it collects most of what the 64x88 class leaves.
- **Batch four has about 11% of slack with names:** the 64-row FP8 dispatch,
  the recurrent verify kernel, the NVFP4 64-row path, attention efficiency
  and the idle gaps. These are the targets for roadmap item 3, and the
  recurrent verify kernel is the one piece of engine code among them.
- **For the three-bit decode-weight project:** the 16-row NVFP4 kernels
  stream one weight footprint at 217 to 232 GB/s. A dequantizing 3-bit
  kernel has to match that on 72% of the bytes to deliver the projected
  gain; the bar is measured, not assumed, and the bytes-per-launch method
  here is how to check a new kernel for hidden traffic.
- **Method, for next time:** root container with `CAP_SYS_ADMIN`, eager
  path, application replay, the `lts__` system-memory counters, bytes from
  the counters and durations from the graph timeline.

## Evidence

`$SHARAR_RESULTS/reviews/2026-09-15-profile/`: `profile_window.py`,
`run.sh`, `run_ncu.sh`, `run_ncu2.sh`, `nsys-bs{1,4}.nsys-rep` and their CSV
exports, `summary-nsys-bs{1,4}.json`, `timeline-bs{1,4}.json`,
`ncu-sysmem-bs{1,4}.csv` (the counter pass), `ncu-eager-bs{1,4}.csv` (the
pass whose `dram__` metrics came back empty), `roofline-merged-bs{1,4}.json`
(the tables above), the analysis scripts, and the combined `results.json`
named in the [manifest](gb10-round-profile-results.json).
