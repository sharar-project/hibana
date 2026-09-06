# Shared prefill and verification projections

This Phase 7D profile tests whether independent prefill and verification rows
should share target weight applications on GB10. The experiment lives entirely
in `bench/mixed_profile.py`; it does not implement a mixed serving path.
The control is runtime source `64e02c935d6c2d98790e919617cd88d096beb2be`, with
the pinned Qwen3.8 NVFP4 target, native DFlash2 k=7, direct reconstruction,
35,000 target pages, 12 state slots and a four-request active limit.

## What can share work

All 64 target layers apply the same dense weights to prefill and verify rows.
The probe covers 64 complete NVFP4 MLPs (gate/up, fused SiLU quantization,
down) and 176 FP8 mixer projections (GDN QKV/Z/output and full-attention
QKV/output). Their loaded tensor sizes are respectively 9,625,928,704 and
7,214,204,608 bytes. These are logical weight sizes, **not measured DRAM reads**.
GDN's small BF16 B/A projection, norms, rotary/gating, attention, recurrence,
target logits, draft generation, feature feedback and state commits are excluded.

Admission and dependencies constrain the possible combination:

- One new prefill leaves at most three existing verification requests. If
  three requests are themselves prefilling, only one verification slot remains.
- A request's first verify depends on its completed prefill, sampled pending
  token and proposer initialization. It cannot share that same prefill pass.
- Current verification graphs capture B=1/2/4. Three active requests execute
  32 rows at k=7; only 24 rows are useful. The final baseline includes all 32;
  the packed probe uses the 24 active rows.
- Prefill GDN updates its own committed recurrent state, while verification
  preserves committed state until acceptance. Their distinct kernels and
  transaction boundaries must survive any sharing of dense work.

The observed `local-v1` mixed workload has a 2,172-token session. Its first
2,048-token chunk may be batched with the short chat/JSON prompts. A final
124-token chunk overlaps three already-running requests. Arrival timing can
change admission grouping even with frozen client delays; the report retains
every group and request ID. The scheduler can execute two prefill forwards
before decoding, which must be included in any client-gap prediction.

## Method and decision gates

The initial brief is frozen before measurement. A second, recorded amendment
adds actual B=3 graph padding and the observed session-tail shape. It also
removes the same terminal token IDs as `local-v1` before grading HTTP text;
the first pilot had incorrectly included `<|im_end|>` in its automatic graders.
Raw first-pilot results are retained. Timing and numerical gates are unchanged.

Real model forwards capture each operation's inputs. Prefill grid inputs are
the first 2,048 session activations sliced to smaller row counts; the separate
124-row case captures the actual continuation after a committed 2,048-token
prefix. Verification capture eagerly replays the graph's exact padded inputs
before acceptance. B=1 probes use the first eight active rows of that capture.
This capture pass is diagnostic and is never used for speed measurements.

Each timing graph visits every layer's weights. The separate graph performs
the full prefill sweep followed by the full verification sweep. The packed
graph concatenates the two input partitions for each operation, runs it once,
and splits output views. Concatenation costs are included. The numerical pass
checks every operation and each active partition separately, so many prefill
rows cannot hide damaged verification rows or dummy-row mistakes.

Each cell has three ABBA repetitions with five graph replays per sample.
Both graphs are resident during timing; all inputs are immutable. HTTP phase
observations are separate and instrumented. CUDA event spans can include queue
gaps; neither these spans nor the logical byte inventory measure physical
memory traffic. All-layer sweeps avoid making the decision from a single hot
layer, but detached activation replay still omits cross-layer dependencies
and the real mixed execution graph.

The frozen feasibility gate is at least 10% reduction for a named dense family
at a prefill size no larger than 512, with finite outputs and relative RMS
error at most 1% for every operation/partition. This authorizes a prototype,
not promotion. Production still needs full target numerical/quality and
ownership gates, at least 10% selected completion improvement, no more than
5% protected regression, equal memory and independent repeated confirmation.

## Results and decision

**Proceed to the bounded prototype.** The independent confirmation passes
the feasibility gate for both dense families. Final all-layer event medians:

| Prefill rows / active verify requests | MLP separate → packed ms | Reduction | FP8 separate → packed ms | Reduction |
|---|---:|---:|---:|---:|
| 128 / 1 | 97.34 → 52.73 | 45.8% | 72.29 → 37.85 | 47.6% |
| 128 / 3 | 102.31 → 54.21 | 47.0% | 71.82 → 38.43 | 46.5% |
| 512 / 1 | 125.80 → 85.87 | 31.7% | 88.13 → 53.69 | 39.1% |
| 512 / 3 | 130.40 → 87.19 | 33.1% | 87.50 → 56.70 | 35.2% |
| 2,048 / 1 | 330.96 → 294.85 | 10.9% | 202.13 → 182.96 | 9.5% |
| 2,048 / 3 | 335.49 → 298.74 | 11.0% | 201.48 → 183.25 | 9.1% |
| Actual 124-row session tail / 3 | 102.41 → 53.99 | 47.3% | 71.18 → 37.72 | 47.0% |

The first independent process, without B=3 padding in its baseline, found
46.0% MLP and 48.1% FP8 reduction at 128/1. This supports the short-row result
without attributing the whole gain to removal of dummy verification rows.
The final B=1 confirmation preserves that baseline shape. B=3 results include
both sharing and elimination of unused dense rows; they are not a pure
weight-reuse ablation.

At B=3, fitting 488 prefill plus 24 verification rows into 512 rows reduces
packed MLP time from 87.19 to 81.02 ms compared with appending to 512 prefill
rows; FP8 falls from 56.70 to 53.74 ms. It also processes fewer prefill tokens,
so these numbers do not prove better prompt throughput or justify a new
scheduler policy. The effect is smaller at 2,048 rows. Keep both layouts in
future experiments rather than assuming row padding is free.

All **26 final GPU cells** and all 24 first-pilot cells pass the numerical gate.
The final maximum per-operation/partition relative RMS error is 0.00465% for
MLP and 0.00820% for FP8; the first pilot's maximum was 0.0662%. These small
differences still require full-model numerical/quality validation before using
mixed shapes in serving. The actual session-tail MLP comparison is bitwise;
the FP8 comparison includes small B=3-versus-padded-B=4 arithmetic differences.

The final HTTP observations contain one warmup plus three measured repetitions
of four arrivals. All six measured automatic coding/JSON checks pass; the six
chat/session outputs satisfy their rubrics in Codex review. This is a baseline
diagnostic, not a human production approval or candidate quality comparison.
Measured large prefill groups take about 796 ms and the session tail about
155 ms; target verification event spans have a median of 88.71 ms. The coding
stream's largest observed delivery gap is 923–931 ms. There are only two useful
overlap opportunities per final measured run, followed by decode-only work.

**No whole-request speedup is established.** Independently timed family savings
must not be added and reported as model latency. In particular, this profile
does not prove a 10% completion gain on the current mixed manifest. Freeze a
selected short-arrival/session-tail workload before implementing the prototype,
retain `local-v1` as protection, and measure the resulting HTTP behavior.

## Reproduction and proof

Run `python3 bench/mixed_profile.py --output NEW_DIRECTORY` in the validated
GB10 runtime with exclusive GPU access. The source tree is mounted read-only
over image `sharar:direct-state`; no dependency or runtime-source change is
needed. The complete CPU suite passes **240 tests**, with six GPU-only skips.
The new CPU cases check eligibility/dependencies, partition-sensitive numerical
gates, nonfinite rejection and executed-versus-active padding rows.

The [evidence manifest](mixed-projection-results.json) records the full external bundle
location and SHA-256; detailed samples, sources and outputs stay outside Git.
Full operation checks, frozen briefs/source trees, reports,
memory samples and clocks/power logs are under
`$SHARAR_RESULTS/reviews/2026-09-05/mixed-projection/`. The first-pilot grading
mistake remains visible there; the confirmation uses the corrected text path.
The lowest observed host `MemAvailable` was **77.45 GiB**, above the 16 GiB
guard. The original serving image and memory guard are restored and verified
after both isolated runs. No new serving image or default is promoted.

## Next implementation boundary

Build one bounded, opt-in mixed target forward. Start with a shared MLP as an
ablation, then share FP8 projections through the same per-layer boundary.
Avoid introducing another scheduler, proposer API or state manager.

1. `SpecDecoder` prepares proposals and verification metadata before the target
   pass, then accepts its result afterward. Scheduler admission identifies
   already-ready independent requests; it never includes the new prefill's
   own dependent verification.
2. Model layers expose common dense projection/MLP work separately from the
   existing prefill and verification attention/GDN cores. Preserve the cold
   BF16 ragged attention path and the paged prefix/verify paths independently.
3. `Executor` owns bounded packed row buffers, graph lifetimes and feature
   slices. Padding is explicit, and inactive rows cannot own slots or pages.
   Retain graph execution for protected decode-only workloads.
4. `StateManager` remains the only authority for reservation, accepted-prefix
   commit and ownership publication. A failure after either mode mutates state
   must poison affected requests; it cannot retry a partially completed pair.

Before end-to-end timing, prove per-request state/feature routing, every
accepted/EOS/budget position, cached-prefix continuation, seeded sampling,
cancellation and failure cleanup against the separate-forward control.
Mixed GEMM shape changes require a target oracle; passing this local projection
comparison alone does not establish identical model continuations.

The row-budget probes are a design input, not a scheduler recommendation.
Shrinking every prefill chunk can erase the sharing gain by repeating overhead
and delaying long sessions. Preserve the existing 2,048-token policy initially;
test any budget change as a separate ablation. Whole-request and stream-gap
improvements must be measured directly. No competitive lead over SGLang or
novelty claim follows from this profiling experiment.
