# GB10 agent checkpoint retention

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

2026-09-10 benchmark study against `6987f1a`; publication decision 2026-09-11.
The target and draft weights, k=15 verification, and production scheduler
configuration are unchanged. The original production container is restored
after each guarded measurement window; this source change is not deployed.

The cached executor includes the qualified checkpoint-retention change and
its memory accounting. The short-resume scheduler remains an explicit
benchmark experiment; its mixed-throughput and streaming regressions rule
out promotion. Publishing the source and evidence does not deploy a new
production image.

## Confirmed native result

Two independent baseline loads and two corrected candidate loads give
**3.240× higher resumed completion throughput**, from 11.337 to 36.733 TPS.
The six-family total falls from 135.483 to 41.815 seconds; individual family
ratios range from 2.80× to 4.35×. All 12 candidate probes match their
immediate-hit oracles' complete outputs, emitted work and retained prefix.
Both implementations' immediate-hit outputs also agree across loads.
The baseline falls back on all 12 interleaved probes; the candidate on none.

Cold decode changes are −0.12% for code BS1, +0.06% for design BS1, −0.14%
for reasoning BS1 and −0.10% for BS4. All seven complete cold continuations
are identical across baseline, candidate and the enlarged-LRU control.
There is no demonstrated cold throughput gain or material regression.

| Warm interleaved configuration | Feature cap | Live features after probe | After flush | Completion TPS |
|---|---:|---:|---:|---:|
| Baseline LRU | 80 MiB | 80 MiB | 80 MiB | 11.337 |
| Enlarged LRU diagnostic | 460 MiB | 460 MiB | 460 MiB | 36.731 |
| Checkpoint retention | 460 MiB | 120 MiB | 0 MiB | 36.733 |

The enlarged-LRU diagnostic is one fresh load, whereas baseline and candidate
have two each. It matches the warm recovery in this trace. Sufficient feature
capacity explains the throughput rescue; coupled ownership supplies reliable
lifetime and reclamation. The change is not a further 3.24× gain over a
sufficiently sized LRU, nor additive to the earlier duplicate-save rescue.

## Repeated HTTP result at 24 slots

Each variant has two independent loads and four timed blocks per workload.
The second window reverses the order: scheduler, retention, baseline.

| Workload | Baseline TPS | Retention TPS | Retention + experimental scheduler TPS |
|---|---:|---:|---:|
| Four cold requests | 37.291 | 37.452 | 38.429 |
| Repetitive four-session cache stress | 37.010 | 223.355 | 271.813 |
| Two cold requests + two repetitive resumes | 25.316 | 64.703 | 59.997 |

The repeated-text limitation below applies to the two warm rows. Cold HTTP
throughput changes by +0.43% with retention; native fixed-partition controls
establish neither a cold gain nor a material regression. Retention has zero
feature fallbacks, versus 24 across the baseline's 48 stress-session requests
and eight across its mixed resumes. One of the candidate's 48 stress-session
requests loses its entire target prefix under recurrent-state pressure.
Its first-token latency is 5.20 seconds; pooled stress-session p95 rises from
0.318 to 0.828 seconds. This tail is retained in the aggregate result.

In the mixed trace, resumed p95 first-token latency changes from 8.695 seconds
with FIFO retention to 2.101 seconds with the scheduler experiment. Mean
completion changes only from 11.442 to 11.391 seconds, while mixed throughput
falls 7.27% and the maximum resumed stream gap grows from 0.961 to 3.654
seconds. The experiment's higher pooled repetitive-session throughput is
also affected by the single target-prefix miss in the FIFO run. It is not a
demonstrated general scheduling gain.

## Conversation check at 96 slots

The additional matched baseline/candidate loads use 96 recurrent slots and
100,000 KV pages. Each has two timed blocks after warmup; these are one fresh
process per implementation, not two independent process repetitions.

| Non-repetitive four-session conversations | Baseline | Retention |
|---|---:|---:|
| Aggregate completion throughput | 28.779 TPS | 78.482 TPS |
| Timed-block throughput | 29.458 / 28.131 TPS | 78.874 / 78.093 TPS |
| Per-request p95 completion latency | 27.757 s | 7.180 s |
| p95 first-token latency | 0.339 s | 0.275 s |
| Maximum streaming gap | 0.509 s | 0.451 s |
| Feature fallbacks / 24 requests | 12 | 0 |
| Full target-prefix misses | 0 | 0 |

This is a **2.727× throughput improvement** on the simulated conversation
trace. Each implementation produces 24 distinct complete output sequences,
with no short periodic cycles; saved text was inspected for task-relevant
responses. Actual previous outputs feed the next turn, so trajectories and
batch membership can differ. This is not an identical-output comparison or
an execution-based test of the generated code.

The 96-slot repetitive control also has no target-prefix misses or snapshot
evictions. The candidate retains 880 MiB of live features after either the
stress or conversation block, below its 1,900 MiB cap. The larger state pool
removes the observed small-pool pressure from these short traces; arbitrary
longer sessions can still exceed the pool. Cold HTTP changes from 38.065 to
38.355 TPS (+0.76%), consistent with no material cold regression.

## What changed

Projected draft features now belong to the recurrent slot for their exact
checkpoint. A target checkpoint cannot lose those features to an independent
feature LRU. Freeing its slot also releases its feature tensor, including
duplicate insertion, interior snapshot eviction, cancellation, retraction,
deferred flush, and cold-tier parking. Restore binds imported features to the
newly allocated slot. Single-request prefill allocates the prompt endpoint
before saving its features, preserving branches from that endpoint after a
longer completion.

The cap is one feature window per usable recurrent slot, reserved before
memory planning. With 24 slots and a 2,048-token DFlash2 window, that is
460 MiB, versus the baseline's 80 MiB. This trades additional bounded capacity
for reliable speculation on retained prefixes. KV eviction and recurrent
snapshot selection still use the existing radix policy. The change does not
make evicted checkpoints reusable or introduce session priorities.

At the production state capacity of 96 slots, the feature cap is 1,900 MiB
(1.855 GiB). Automatic planning reserves the larger cap and its temporaries
before allocating KV, so the available KV page count can decrease. The
explicit benchmark geometries keep KV capacity fixed between variants;
they do not measure that automatic-planner tradeoff under a nearly full KV
working set.

`cache.draft_features` reports entries, live bytes, capacity, hits and
fallbacks. Live tensor bytes are distinct from the PyTorch allocator's reserved
memory. GB10's managed cold tier shares physical DRAM with the GPU; parking is
not additional physical capacity.

## Measurement protocol

All loads use the pinned production image, Qwen3.8-27B NVFP4 target, FP8
DFlash2, reconstruction, depth 15 and maximum batch four. The primary
comparisons use 24 recurrent slots; the additional conversation/capacity
check uses 96. The restored production service uses the BF16 draft, so these
are FP8 source benchmarks rather than measurements of a deployed change.
Native comparisons use 60,000 KV pages; HTTP comparisons use 100,000 for every
variant so four primed histories plus two fresh arrivals fit without KV
capacity confounding the scheduling screen. Model loading, graph capture and
explicit shape warmup are outside timing.

Native cold controls generate 1,000 tokens for code, design and reasoning at
BS1, and four distinct task families at BS4. Six independently primed
10K-token families each have an immediate-hit oracle and a 256-output probe
after two unrelated 10K requests. Complete output IDs, emitted work, retained
prefix length, fallback counts, timing and feature bytes are saved. The
fallback baseline's ordinary-decode continuation can differ numerically from
the healthy speculative oracle; a speedup against that fallback is not an
identical-output comparison or a cold-kernel improvement.

The native enlarged-LRU control uses the same 460 MiB cap as the candidate.
It distinguishes the performance benefit of sufficient capacity from the
ownership and cleanup benefit of coupling features to checkpoints.

HTTP traces exercise four cold requests; four interleaved sessions with three
turns each; and two fresh long requests followed by two cached resumes at
100 ms. Later session inputs include the actual previous output IDs and a
fixed 128-token tool-result fixture, with a 100 ms pause between turns. One
full trace block at 32 outputs warms the actual HTTP shapes and is excluded;
two subsequent blocks are measured per independent load. These are controlled
serving traces, not real tool execution or agent task-success measurements.

Output inspection found that repeating the tool text to fill 128 tokens
encourages repetitive model continuations and unusually high draft acceptance.
The large HTTP agent-turn ratio is therefore a **cache stress result**, not
evidence of comparable real-agent throughput. The raw fixture and output
repetition audit are retained. Natural-stop canaries check basic functionality
separately; they do not turn this stress fixture into a task-quality benchmark.

The scheduler experiment ranks short cached suffixes using CPU-only radix
inspection, applies two-second aging, and temporarily limits prefill chunks
to 512 tokens while such a suffix is waiting. It lives in `bench/` and does
not change the serving scheduler. Its independent comparison must account
for throughput as well as resumed TTFT and long-request latency.

The short-resume policy is **not promoted**. In both experimental loads it
delivers the resumed first token earlier, then interrupts that stream with
roughly 3.6-second prefill gaps. Completion stays near the FIFO result. Moving
the wait from before the first token into the response does not establish
better completed-turn throughput, and worsens streaming behavior.

The 24-slot retention trace also exposes a separate limit: recurrent-state
pressure can evict a session's target checkpoint, requiring full prefill
even though features remain valid for every surviving checkpoint. One such
miss explains the slow second block of `retention_http_1`. Zero feature
fallbacks must not be mistaken for zero target-cache misses. The additional
96-slot check matches production's state capacity, while retaining the same
100,000-page KV geometry between its baseline and candidate.

That check also includes `agent_chat`: distinct family/turn test reports,
snapshot-rendered chat boundaries, and the actual prior output IDs. Reports
arrive as user-relayed fixtures through `/generate`; no external tests or
tools actually run. Each of four lanes has three 128-output turns and a
100 ms pause. Its excluded warmup uses the full 128-output length. Decoded
continuations are retained to inspect response diversity. These remain
fixed-budget simulated conversations, not complete agent tasks.

## Evidence and limitations

Raw protocols, commands, complete outputs, resource samples and immutable
source snapshots are under
`$SHARAR_RESULTS/reviews/2026-09-10-agent-retention`.
The initial `source-retention` prototype precedes the prompt-end ownership
correction and is excluded from final candidate pooling. The amendment and
original failed test logs are retained alongside the corrected validation.

The final CPU suite passes **447 tests with eight skips**. The affected GPU
suite passes **45 tests**, including BF16/FP8 branch restoration,
long batched prefill, accepted-token feedback and cold-tier ownership. CPU
validation and final restoration details are recorded in the
[compact evidence manifest](gb10-agent-retention-results.json). All eight HTTP loads complete their four natural-stop canaries
(32/32); all recorded streams and background requests complete successfully.
The final service check found the original container stopped after the
measurement supervisor ended; it was explicitly restarted with the original
command and memory guard. No candidate image was deployed.
Final health is 200, the memory guard is active, and no experimental container
remains. Minimum sampled `MemAvailable` across final validation/measurement
windows is 65.92 GiB. These service observations describe the end of benchmark
validation; production deployment is separate from source publication.

The native logs precede a correction to the generic recorder, which labeled
physical rows as if every request speculated. Analysis reconstructs their
ordinary and verification work from the recorded per-request `spec_rounds`
and emitted counts. The maintained recorder and subsequent HTTP runs report
those paths separately, with CPU regression coverage. Native completion
timing, output comparisons and engine source are unaffected by this
instrumentation correction.

Free-running HTTP changes admission order, batch membership and sometimes
floating-point continuations. Client/server pairs and output counts are
checked exactly within each run, while native fixed-partition continuations
are compared across implementations. A few controlled families and short
tool pauses do not establish performance under arbitrary long-lived agent
traffic, CPU tool contention, or managed-memory pressure.
