> Archived experiment: reproduce from commit `56605f0`. Its model, executor,

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.
> scheduler hooks and benchmark runner have been removed from the current tree.

# Mixed target forward on GB10

The shared-projection serving experiment is **archived**. It combines an
arriving prefill with independent, already-ready speculative verification
requests. The bounded [prefill metadata optimization](prefill-plan.md) improves
ordinary latency, but mixed execution still fails first-token protection.
The CLI option, model/executor hooks and benchmark runner have been removed
from the current tree. The implementation below describes the archived commit;
further mixed scheduling is not on the active speed path.

## Implementation and ownership

At the archived commit, `bench/mixed_forward.py` enables the experiment through
`Executor.enable_mixed()`. `mlp` shares only MLP work. Ordinary serving always
leaves mixed execution off. The executor allocates one additional 256 MiB attention
workspace on enablement. Turning the mode off retains this workspace so the
benchmark can compare equal persistent pools/workspaces and report temporary
allocation peaks separately. A server started with `off` never
allocates it. The existing fixed prefill policy is required.

The implementation uses the existing scheduler, proposer contracts and
StateManager. `SpecDecoder.prepare` reserves and drafts; its target result can
then come from an ordinary verification graph or a mixed eager forward.
`SpecDecoder.finish` uses the common acceptance, stop, state-commit and feature
feedback path. No second sampling or state-ownership implementation is added.

Each model layer exposes dense work separately from its attention/GDN core.
Full mode shares NVFP4 MLPs and FP8 projections; the small BF16 GDN gate
projection keeps its separate row shapes. Prefill and verify keep separate
residuals, feature storage and attention plans. Cold prefill still uses BF16
ragged attention; cached prefill and verification retain paged FP8 attention.
Verification never writes committed recurrent state before acceptance.

StateManager checks distinct live private slots, valid prefill extents, pending
verification positions and absence of previous reservations before any writes.
Page-pressure preflight remains retryable. Any later failure poisons both
groups; a late `MemoryError` becomes non-retryable because releasing reservations
cannot undo in-place prefill writes or proposer progress.

There is at most one mixed pass per scheduler step, bounded by 4,096 prefill
rows and the existing active-request limit. It consumes one decode round.
A newly completed prefill joins the next verification round, since drafting
its first block depends on its newly sampled token. Cancellation is reaped
between the scheduler's two possible prefill chunks, before authorization.
Unsupported groups, including sampled MTP/n-gram requests, keep the ordinary
fallback. Decode-only execution retains its verification graphs.

See [the contribution contract](../../interfaces.md#mixed-prefill-and-verification-experiment).

## Frozen experiment

The pre-implementation brief and `bench/workloads/mixed_v1.json` pin the target,
DFlash2 k=7, direct reconstruction, 35,000 target pages, 12 state slots,
four active requests and the existing 2,048-token chunk/4,096-token budget.
The control runtime is `0d570cf543ca58e11b161fb88dd172b413e054e3`.

Four selected cases submit the existing `local-v1` chat or cached session
request immediately after one or three JSON requests have each delivered
their first token. Every measured candidate run must actually execute a mixed
pass; missed overlap is an eligibility failure, never a discarded outlier.
The selected metric is mean completion of the existing requests. Arrival
completion, TTFT and delivery gaps are protected at no more than 5% regression;
selected completion must improve at least 10%. Natural stopping, sampling and
budgets remain fixed. Changed output lengths remain visible.

`bench/mixed_forward.py` measures real HTTP without profiler or CUDA-event
instrumentation. It records all outputs, cache observations, delivery events,
mixed-step counts, source fingerprints and allocation peaks. Both modes retain
the extra workspace. A scheduling-only `separate` ablation keeps ordinary
verification graphs; `mlp` isolates the sharing family. The full `local-v1`
manifest protects ordinary decode, long contexts, prefix reuse, sampled
requests, cancellation and small budgets.

## Results

The final independent run has one warmup plus three measured ABBA blocks per
case: six measured samples per mode. Every candidate sample includes the
arriving request in a mixed pass. Median changes from ordinary serving:

| Arrival / existing requests | Existing completion reduction | Arrival completion change | Arrival TTFT change |
|---|---:|---:|---:|
| Short / 1 | 7.1% | -7.3% | +23.8% |
| Short / 3 | 3.5% | +1.3% | +20.0% |
| Cached tail / 1 | 7.8% | +1.4% | +15.3% |
| Cached tail / 3 | 20.7% | +5.8% | +14.6% |

Existing requests' largest delivery gaps fall 21–30%; arrival gaps change by
less than 1.1%. Those gains do not override slower arrival TTFT or the cached
tail's completion regression. All existing JSON requests emit the same 30-token
answer, but some arriving continuations change length with the GEMM shape.

Both 88-group full comparisons and the 40-group pilot pass their automatic
selected checks. Codex review passes all 88 rubric-scored arrivals in the final
run. This is a small local quality check, not independent human review or broad
model evaluation. The pilot's arrival TTFT regresses 13–20%; the first full
comparison regresses 16–20%. Concurrent completion savings vary substantially:
the cached-tail/B=3 result is 3.1%, 11.8% and 20.7% across these runs. Keep every
sample and group rather than choosing the most favorable result.

The standalone candidate passes **654 CPU/GPU tests**, with one skip for an
unsupported MTP verification width. A separate CPU-only run passes **301 tests**
with six GPU-only skips. The new GPU cases cover DFlash2 reconstruction, DSpark
snapshots and MTP reconstruction, B=1/3, cold/cached prefixes, MLP/full sharing,
reduced verification widths, sampled fallback, stopping and continuation.
Block/logit comparisons retain the existing numerical criteria. CPU recurrence
oracles cover every acceptance/budget/EOS position, RNG, failures, shared prefix
loans and cancellation between chunks; GPU checks protect untouched slots/pages.
The lowest controlled block cosine is 0.999973. Complete forward states are
not bitwise equal: the lowest prefill final-norm cosine is 0.8829, while prefill
top-1 agreement remains at least 92.97%. Verification blocks have too few rows
for the 20-position confidence gate; controlled block tests, the strong-margin
next-token rule and real continuation checks remain necessary. Do not describe
these shape-changing forwards as numerically identical. All 29 required prefill
confidence checks pass (minimum agreement 92.8%), as do all 51 required
strong-margin next-token checks.

The protection comparison completes 78 HTTP cases in each mode without request
or automatic-check failures. All non-mixed output IDs match between modes and
against the historical `64e02c9` runtime. Six mixed-arrival continuations change.
Codex rubric review records 22 passes/four misses in the control and 20 passes/
six misses with mixed execution. The four sampled-chat misses omit recomputation
in both modes. Two additional mixed-chat responses describe reuse of common
code patterns without identifying shared-prompt computation. They fail the
existing rubric; shorter answers are not a quality-qualified speedup.

Protected completion changes range from -3.79% to +2.71% outside the mixed
and cancellation groups. The mixed session is 13.82% slower, with changed
output length. Some TTFT medians also exceed 5% in unchanged execution paths;
these small, sequential protection runs do not isolate noise from drift.
The single cancellation sample completes correctly but cannot support a latency
claim. Both quality and latency reject promotion independently.

The historical parity oracle also passes all required comparisons. The first
pipeline's `python -m sharar.cli` invocation did not execute a command; the
recorded parity result comes from the corrected `python -m sharar parity` run
with its summary written beside a read-only reference proxy.

The final selected-case confirmation records exact request membership as well
as counters. Earlier runs counted all mixed passes, which could include sharing
among the initial requests. Final eligibility requires the named arrival itself
to share a pass; timing thresholds and workloads are unchanged.

## Why this can save work and still miss the serving gate

A separate fixed-work diagnostic prepares identical requests and times one
prefill plus one existing verification round, including preparation, state
publication and feedback. Three measured ABBA blocks give these wall medians:

| Case | Separate pair | Mixed pair | Reduction |
|---|---:|---:|---:|
| Short / 1 | 242.95 ms | 184.83 ms | 23.9% |
| Cached tail / 3 | 275.28 ms | 200.60 ms | 27.1% |

Both modes emit identical existing-request IDs in every measured fixed-work
sample: eight verification outputs after the initial prefill token.

This measures a pair of forwards, not request completion or TTFT. The ordinary
server can publish the prefill's first token before the following verification.
The mixed server must finish the combined pass first. Existing requests benefit
from that shared work, while the newcomer can wait longer.

Separate profiler runs show GPU activity union falling from 203.79 to 131.01 ms
and 230.88 to 147.04 ms respectively. Instrumented mixed spans still have about
31–35% uncovered device time. These include Python/profiler overhead and are
not a bandwidth measurement or a promise of recoverable time.

The short mixed trace contains 388 stream synchronizations, 384 under GDN
prefill. Of these, 144 occur while rebuilding/uploading the same three small
metadata tensors across 48 layers; another 240 occur under the chunked GDN
function. The first group spends 52.66 ms inside synchronization calls, which
includes waiting for previously queued GPU work. That number must not be
reported as a potential 52.66 ms speedup. CPU-scope accounting filters out
duplicate GPU annotation events; raw traces are retained.

## Bounded follow-up and archive decision

The [one-plan metadata change](prefill-plan.md) improves ordinary TTFT by
18–29% with unchanged output IDs. Repeating the frozen mixed-v1 comparison
still fails its 5% protection limit. Six measured samples per mode/cell give:

| Case | Existing-request completion reduction | Arrival completion change | Arrival TTFT change |
|---|---:|---:|---:|
| Short, one existing request | 12.3% | −7.5% | +18.3% |
| Short, three existing requests | 12.1% | −0.3% | +21.8% |
| Cached tail, one existing request | 11.4% | +2.0% | +16.9% |
| Cached tail, three existing requests | 5.8% | −0.6% | +24.4% |

All four arrival TTFT gates fail; the final cell also misses the 10%
existing-request completion target. Faster ordinary prefill is retained.
The `--mixed-forward` serving option is removed, and no further scheduling
policy is planned around this prototype. Its reusable model/verification
interfaces, state tests and explicitly enabled HTTP benchmark remain.

Fresh equal-budget SGLang measurements are separate from this experiment.
Projection-sharing savings alone establish no competitive lead.

## Reproduction and evidence

Run `python3 bench/mixed_forward.py --output NEW_DIRECTORY --protected` in the
validated GB10 runtime under exclusive GPU access and the 16 GiB host guard.
`--pilot` is explicitly insufficient for promotion. Preserve independent runs,
including failed quality and missed overlap. Runtime dependencies are unchanged.

Raw frozen sources, commands, images, test logs, model numerics, HTTP outputs,
rubric reviews, traces and host memory/clock samples are under
`$SHARAR_RESULTS/reviews/2026-09-05/mixed-forward/`.
The [evidence manifest](mixed-forward-results.json) records the full external bundle
location and SHA-256; detailed samples, sources and outputs stay outside Git.


The standalone candidate image is
`sha256:d1a67054a02bf836668adec5e281f27a5f8a82627bad375b03af0a3585e2845e`,
built by copying source onto the already validated `sharar:direct-state` base.
An attempted clean offline Dockerfile build could not fetch uncached APT
dependencies; it is not counted as a successful dependency rebuild. The final
membership confirmation mounts updated benchmark code over the candidate;
every serving-runtime file matches the image that passed the complete suite.

The lowest sampled host `MemAvailable` is 67.28 GiB, above the 16 GiB floor.
Final selected allocation peaks are 27.84–27.93 GiB; both modes reach the same
28.86 GiB allocator reservation with equal configured pools/workspaces. These
are CUDA allocator figures, not total physical-memory or DRAM-traffic counters.
The original serving image and memory guard are restored and checked after
each isolated run. No production image or default is promoted.
