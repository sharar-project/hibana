# W3 MLP hardware counters and cache-policy screen

2026-09-17. Evidence root:
`${SHARAR_RESULTS}/reviews/2026-09-17-w3-mlp-profile/`.
The [compact manifest](w3-mlp-profile-results.json) identifies the full raw
reports, commands, source snapshots and checksums.

## Scope and decision

Profile the two large grouped-activation MLP projections at physical row
counts 16, 32 and 64: the usual target-verification shapes for request
batches one, two and four with k=15. **Keep the existing streaming weight
loads.** Two counter-guided cache-policy substitutions preserve outputs
but regress every measured shape. Neither is integrated or promoted to
the family gate.

The previously qualified [QKV specialization](w3-qkv-tiling.md),
[mapped-checkpoint staging](w3-runtime-profile.md) and source-keyed W3
extension build remain unchanged. This study makes no new serving speedup
claim.

## Method

Use actual layer-zero matrices from the calibrated target manifest
`9585ba8c682de346e526aeed0e0c9ba5ec34b341d7e8bfb382da0b81f60991b8`:
gate/up `[34816,5120]` and down `[5120,17408]`. Activations are seeded random
BF16 values, converted through the serving per-32-group FP8 quantizer.
This is a shape-matched microprofile, not captured user activations or a
complete model replay. Gate/up uses split-K=1 throughout; down uses 8, 2
and 2 splits respectively. Current CUDA source SHA256:
`1dce452331124d23dff06b0565fc2fde46ae95270dd7466b3fc39f7f9888f0f6`.

Nsight Compute 2025.3.1 runs in a temporary, network-isolated container with
`SYS_ADMIN` for the driver's restricted hardware counters. The host driver
policy is unchanged. Repository and checkpoint mounts are read-only;
compilation uses an isolated cache. The first two attempts failed before
collecting counters because of output-directory and checkpoint-file
permissions. The successful run uses byte-verified copies of just the two
required matrices. Logs of the failed setup attempts are retained.

The complete initial capture uses 19 kernel-replay passes per case, with
caches flushed between passes and GPU clocks left unchanged. Nsight
Compute's documented [cache/replay behavior](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control)
means this measures isolated cold-cache behavior. Its timings must not be
substituted for normal graph-replay timings.

## Cold-cache counter results

“Long scoreboard” means a warp is waiting for a dependency from a
local/global/texture load. Its share below is the counter's stalled cycles
per issued instruction divided by total warp cycles per issued instruction;
it is **not a fraction of wall-clock time that can simply be removed**.

| Matrix | Rows | Registers/thread | Achieved occupancy | Eligible warps/scheduler | Long-scoreboard share | Tensor active | Local spill requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gate/up | 16 | 102 | 29.52% | 0.152 | 85.86% | 4.73% | 0 |
| Gate/up | 32 | 184 | 15.65% | 0.177 | 80.09% | 9.47% | 0 |
| Gate/up | 64 | 255 | 15.91% | 0.265 | 60.27% | 14.31% | 217,600 |
| Down | 16 | 102 | 27.57% | 0.106 | 94.05% | 3.48% | 0 |
| Down | 32 | 184 | 13.83% | 0.109 | 85.66% | 6.40% | 0 |
| Down | 64 | 255 | 13.75% | 0.174 | 72.18% | 11.09% | 108,800 |

Register allocation limits theoretical occupancy to 33.33% at 16 rows and
16.67% at 32/64 rows. The 64-row kernels also have measured local-memory
spill requests. These observations support investigating memory-load
latency hiding and register lifetime; they do not establish that simply
raising occupancy improves performance. The earlier register caps and
broad row-tile substitutions already failed their timing screens.

The profiler's `gb10b` metric catalog exposes no `dram__` counters. Its
headline “Memory Throughput” reports on-chip resources here, not achieved
DRAM bandwidth. Low Tensor Core activity or that headline percentage cannot
be used to claim a large available speedup. Likewise, packed-weight bytes
divided by elapsed time is useful effective bandwidth, not measured total
memory-controller traffic.

## Unprofiled cache-policy experiment

Hypothesis: bypassing L1 for bit-plane loads might preserve cached
activations; normal caching might improve reuse. Change only the three
bit-plane loads from streaming/evict-first `__ldcs` to either `__ldcg`
(bypass L1) or `__ldca` (normal caching). Keep scales, arithmetic, split
factors and dispatch identical.

Each graph rotates through at least 160 MiB of independently stored
weights, larger than L2, and each timing replays four graphs. Twelve
variant sweeps alternate forward/reverse order. All six cases are checked
for exact BF16 output equality before timing. These are one-process
screening medians, not confidence intervals or serving measurements.

| Matrix | Rows | Current streaming, µs | Bypass L1, µs | Normal cache, µs |
|---|---:|---:|---:|---:|
| Gate/up | 16 | 307.03 | 465.12 | 468.63 |
| Gate/up | 32 | 304.52 | 477.57 | 477.26 |
| Gate/up | 64 | 357.41 | 557.18 | 537.91 |
| Down | 16 | 150.25 | 220.60 | 221.48 |
| Down | 32 | 158.94 | 250.75 | 252.21 |
| Down | 64 | 216.03 | 319.62 | 309.20 |

Both substitutions take 43–59% more time. This is sufficient to reject
these particular implementations without an expensive family evaluation.
It does not isolate every compiler or cache effect, or prove a cause for
each regression. All experimental sources stay in the raw evidence root.

The current kernel's effective packed-weight bandwidth is approximately
236/238/203 GB/s for gate/up and 241/228/168 GB/s for down at 16/32/64 rows.
The batch-four decline is a more useful reason to target MLP kernel work
than an assumed opportunity to eliminate CPU–GPU data copies.

## Follow-up and limits

A follow-up application-replay attempt with software source counters
stalled at the final 64-row down projection. It was stopped, the service
was restored, and no partial measurements from it are used. Raw logs
retain this profiler failure separately from the successful captures.

A hardware-only retry completed **17 application-replay passes** without
forced cache flushing. Register counts and spills remain present, and the
same broad occupancy/load-wait pattern persists:

| Matrix | Rows | Achieved occupancy | Long-scoreboard share | Profiled duration, µs |
|---|---:|---:|---:|---:|
| Gate/up | 16 | 29.91% | 85.91% | 486.02 |
| Gate/up | 32 | 15.99% | 76.55% | 446.78 |
| Gate/up | 64 | 16.01% | 54.12% | 541.06 |
| Down | 16 | 28.87% | 88.57% | 315.30 |
| Down | 32 | 13.80% | 83.07% | 347.97 |
| Down | 64 | 13.74% | 69.87% | 402.37 |

At 64 rows, L1 local-load/local-store counters report 870,400/882,492
sectors for gate/up and 435,200/428,132 for down. These are requests at
the cache interface, not bytes transferred from DRAM. L2 read/miss
sector counts and all metric units are preserved in the raw export.

Even this capture's durations exceed the unprofiled graph timings above.
Removing forced cache flushing does not make isolated instrumented launches
equivalent to continuous graph replay. Treat the counters as diagnostic
evidence of register/spill/load behavior, not a quantified serving latency
breakdown. Specific load attribution remains unresolved because the source
instrumentation attempt failed; this report does not blame activation
loads alone.

Future kernel experiments should preserve the useful weight-streaming
policy while testing selective activation/scale prefetch or reuse with
lower register lifetime. First attribute the waits to specific loads;
the aggregate scoreboard counter alone cannot distinguish weights,
activations, scales and spills. Recheck exact outputs and states, then
use the paired family and HTTP gates before changing a default. None of
the profiler's advisory speedup estimates is a forecast for Sharar.
