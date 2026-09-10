# Direct accepted-state reconstruction

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

This Phase 7D experiment removes the final FP32 output buffer and its copy into
committed recurrent slots. It follows [accepted-feature batching](accepted-batch.md).
The pinned target, recurrence arithmetic, proposer feedback and serving defaults
are unchanged. Reconstruction remains selectable; full snapshots remain the
production default and executable correctness reference.

## Mechanism and ownership

Verification retains compact raw K/V and gate/beta inputs without changing the
committed recurrent state. After greedy or sampled verification applies EOS and
output budgets, `StateManager.commit_reconstructed`:

1. Validates every count against its reservation, the current verified width and
   convolution window. Validates distinct, non-dummy slots and committed page lengths.
2. Prepares retained page tensors, then uploads all slots/counts once, asynchronously.
3. Reconstructs the accepted prefix directly into each authorized slot. The kernel
   uses the same recurrence and reduction order as the retained-snapshot oracle.
4. Copies the corresponding small convolution windows and publishes retained
   pages/committed lengths. The engine updates output histories and invokes the
   existing batch proposer feedback only after these commits.

Each program owns one layer/value-head/K-by-32-value tile. It reads its complete
source tile before storing its result; the full K reduction never reads another
program's V tile. Unique request slots prevent cross-row aliasing. Active rows
alone launch programs, so padded capacity and the dummy slot are not destinations.
No additional GPU synchronization or persistent buffer is introduced.

Failures after mutation mark all affected requests uncacheable. A post-write
`MemoryError` is converted to a non-retryable runtime failure, because freeing
reserved pages cannot restore the old recurrence. Existing failure cleanup owns
release; no output tokens are published by this commit method. The separate-output
kernel layout remains available for tests, without allocating scratch in serving.

## Capacity and validation

At B=4 the removed buffer is exactly **576 MiB**, independent of verify width.
The prospective budget and actual retained-tensor accounting both remove it.
This is a reduction in allocated storage; physical DRAM traffic is not measured.

| B=4 profile | Previous verify buffers | Direct reconstruction |
|---|---:|---:|
| MTP, k=3 | 633.28 MiB | 57.28 MiB |
| DSpark/DFlash2, k=7 | 690.56 MiB | 114.56 MiB |

These totals include compact recurrence inputs and convolution windows. The
production default still uses full snapshots. Freed allocation is capacity
headroom; it does not by itself establish better cache hit rates or throughput.

The focused kernel suite passes 48 cases: B=1/2/3/4; widths 1/2/4/5/8/16;
eager and CUDA graph replay; every accepted prefix; mixed counts; nonordered
slots and padded capacity. Direct writes match full verify snapshots bitwise,
and every unselected slot remains unchanged. The old separate-output layout
also matches the same oracle.

The CPU suite passes 236 tests, with six GPU-only skips. New transaction cases
cover invalid final rows before any write, duplicate/dummy/out-of-range slots,
unverified replay tails, stop/budget selection, exact page-tail reclamation,
partial write failure, uncacheable rollback and complete resource recovery.

The final standalone image passes **548 GB10 checks** in 972.92 seconds, with
no failures or skips. These include the full-snapshot versus direct-reconstruction
committed-state oracle on the real checkpoint, MTP and external-draft lifecycle
coverage, greedy/sampled requests, EOS/budget truncation, cache-hit continuation,
flush, retraction and page/slot recovery. Timing is measured separately below;
passing correctness checks alone does not establish an end-to-end gain.

## Measurement design

Baseline source: `d0861582914cbc15e23187d28d54b655115a9dc1`. Separate processes
measure old and new DFlash2 execution at context 128/8,192 and B=1/4, with k=7,
reconstruction, 35,000 target pages and 12 state slots. The frozen HTTP manifest
includes coding, chat, JSON, tool use, reasoning, sampled output, reused prefixes,
mixed arrivals, cancellation and short budgets.

`bench/draft_profile.py --compare-state BASELINE_SOURCE --output NEW_DIRECTORY`
alternates candidate/control/control/candidate on one fresh model and graphs.
The control imports the baseline reconstruction kernel and StateManager commit,
with a retained FP32 output allocated in both modes to equalize allocation for
timing. Separate-process profiles establish the storage difference. Each cell
uses one warmup block and three measured blocks, six eight-round samples per
method, with fresh requests and equality checks on emitted IDs.

CUDA event spans diagnose reconstruction/commit cost; they include queue gaps.
The direct-commit span includes reconstruction, so these nested spans must not
be added together. Uninstrumented wall time supplies the speed comparison.
This is a small fixed-work pilot, not a tail-latency or competitive-speed claim.

Final interleaved medians:

| Context / batch | Control ms/round | Direct ms/round | Round-time reduction |
|---|---:|---:|---:|
| 128 / 1 | 111.39 | 109.58 | 1.62% |
| 128 / 4 | 130.41 | 125.70 | 3.61% |
| 8,192 / 1 | 114.28 | 115.05 | -0.67% |
| 8,192 / 4 | 137.41 | 132.03 | 3.91% |

All fixed-work emitted IDs match across methods and repetitions. Long-context
B=1 does not show a consistent win: its direct samples span 112.20–115.22 ms
and control samples span 113.52–115.89 ms. The separate-process comparison
shows 3.2–3.8% lower B=4 round time, but long-context B=1 is 2.78% slower there.
Keep that variation visible instead of claiming a universal speedup. Both
comparisons remain within the protected 5% latency-regression limit.

In the separate-process traces, B=4 reconstruction plus commit falls from
10.47–10.62 ms to 5.69–5.76 ms. The remaining small convolution copy is still
required. CUDA allocation falls by exactly **576 MiB in every fixed-work cell**.
Every two-round trace records two stream synchronizations and three profiling
harness device synchronizations in both modes; direct commit adds none.

The separate-process final-image comparison completes 78 HTTP cases per mode
without request or automatic coding/tool/JSON/reasoning failures. Every non-mixed
output token sequence matches, including sampled, cancellation and short-budget
cases. Two mixed-arrival continuations differ: `mixed-chat-2` grows from 67 to
73 tokens and `mixed-session-2` from 66 to 67. Those are not equal-token work.
Main HTTP completion median changes range from -2.65% to +1.19%; mixed groups
range from -0.88% to +0.25%. The short-budget group improves 1.94%. Three
repetitions and one cancellation case do not support population-tail claims.

Assistant review of all 26 rubric-scored outputs per mode records 22 passes
and the same four sampled-chat misses: the retrieval analogy omits explicit
recomputation. The changed mixed responses pass their existing rubrics. This
is not independent human review or a broad model-quality evaluation.

DSpark separately completes all 78 frozen HTTP cases in the final image,
without request or automatic-check failures. Its assistant review records
22 passes and four sampled-chat recomputation misses. This validates the
shared state path; a DSpark before/after speed comparison is not claimed.

Local raw evidence is under
`/home/yke/sharar-results/reviews/2026-09-05/direct-state/`, including source
snapshots, commands, CUDA traces, HTTP records, numerical gates, test XML and
memory/power logs. Run GPU work exclusively under the 16 GiB watchdog and restore
the original serving container and memory guard afterward.
The minimum sampled host `MemAvailable` during final validation is 76.20 GiB.
The original production image and memory guard were restored afterward;
health/model/canary checks passed.
The final standalone image is
`sha256:051f5dd546def8616227e79c9897fb1d69be27330cc936b42825f0c0514da60a`.
The [evidence manifest](direct-state-results.json) links the full archived bundle by SHA-256.

## Next mechanism

Profile opportunities to share target projections across prefill and verification
under mixed arrivals. The prior independent prefill-throttling pilot degraded
session latency; establish useful shared work before adding scheduling heuristics.
Equal-memory cache pressure, broad sampled quality and fresh SGLang comparisons
remain open promotion gates before a default change or Phase 8 capacity work.
