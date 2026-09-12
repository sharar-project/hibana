# Plan: 1.5x cold 10K/1K TPS on GB10

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

**Current priority:** the user has deferred training and asked for engine work
at both BS1 and BS4. Follow [the engine-only next steps](../gb10-engine-next-2026-09-10.md)
first. The training sequence below remains a deferred research option.

The [completed BS4 study](gb10-bs4-engine.md) keeps k=15 and the scheduler
defaults. Narrower verification and the scheduling candidates do not establish
a throughput improvement; the runtime-count kernel prototype remains experimental.

2026-09-10. **BS1 focus.** Updated after the family evaluations, integrated
GPU validation and final HTTP comparison. The [implementation report](gb10-bs1-speedups.md)
records **+1.13–1.27% HTTP decode** and **+0.99–1.05% end-to-end throughput**,
with all outputs identical and all 16 quality canaries passing. Historical snapshots under
`sharar-results/reviews/` retain the original plan.

The measured exact kernel gain is **1.0143x across 18 task families**. The
custom FP8 kernel, two compressed-draft configurations and top-32 selector
have all been screened; none establishes a reliable aggregate throughput
gain under the declared gate. The previous 1.12–1.18x engineering stack was
an extrapolation and is withdrawn. **1.5x remains an open research target.**

## Baseline and the target

Use one GB10, the pinned Qwen3.8-27B NVFP4 target, **FP8 DFlash2 draft,
k=15**, greedy BS1, cold 10,000 input tokens and exactly 1,000 output tokens.
The baseline is approximately 44.2 / 34.2 / 56.4 decode TPS and
37.6 / 30.1 / 46.1 end-to-end TPS for code / design / reasoning. Code emits
about 4.735 tokens per 107 ms round.

The restored production container uses the BF16 draft. Deploying the already
validated FP8 draft is a separate configuration change; it provides no gain
against this plan's FP8 baseline. BS4 optimization and warm-session cache
retention are separate objectives.

For code, 1.5x decode means **66.3 TPS**. At roughly 105.5 ms after the exact
bundle, that requires about **7 emitted tokens per round**, almost 48% more
than the baseline, or substantially less work per round.

A 1.5x **end-to-end** gain is harder: 37.6 -> 56.4 TPS means finishing 1,000
outputs in about 17.7 seconds. With roughly 4 seconds of cold prefill still
present, decode must approach 72.7 TPS, or about 7.7 tokens per 105.5 ms round.
Keep these two targets separate in benchmark reports.

## Completed measurements and decisions

All family results below compare each task against the mean of the opening
and closing native controls. Intervals are paired bootstrap 95% intervals
across the 18 families; they do not establish quality on unseen workloads.

| Step | Measured result | Decision |
|---|---|---|
| Protocol and controls | 21/21 native outputs repeat in both windows; family timing drift −0.16% in the kernel window, −0.05% in the draft window | Complete |
| Exact attention QKV + K512 MLP kernels | Family decode **+1.43% [ +1.37%, +1.49% ]**, round cost −1.41%, 21/21 outputs identical | Integrated at the three M=16 shapes; 53 GPU tests and 1,296 captured-matrix checks pass; HTTP results in the report |
| M=16 Triton FP8 | Family decode +4.68% [ −1.02%, +11.07% ]; round cost −2.07%; all 21 outputs change | Remains experimental |
| Exact bundle + Triton FP8 | Family decode +5.59% [ −0.14%, +12.03% ]; round cost −2.92% | Remains experimental; the QKV savings overlap |
| NVFP4 draft body, W4A16 | Family decode +2.51% [ −4.33%, +10.11% ]; round cost −2.87% | Remains experimental |
| NVFP4 draft body + FP8 feature projection | Family decode +3.01% [ −3.47%, +9.80% ]; round cost −3.21% | Remains experimental |
| Top-32 support | Family decode −0.79% [ −4.03%, +2.48% ]; round cost +0.29% | Keep top-16 |

The draft configurations reduce round cost, but the broad acceptance and
throughput intervals cannot establish a useful improvement across requests.
They do not receive production-quality qualification on the strength of a
favorable point estimate. The three-task held-out subsets are too small to
justify choosing a variant from a favorable subgroup.

The code benchmark emits EOS at output index 98 in the baseline. Its forced
continuation is a useful exactness/performance control, but a poor standalone
judge of changed draft or target numerics. The family protocol is essential.

## Corrected kernel budget

The previous **−6.0 ms** FP8 estimate was based on isolated random-operand
winners. Replaying all 176 real FP8 matrices with their captured inputs gives
**34.573 -> 32.532 ms**, a **2.040 ms** saving, consistent with whole-model
rounds. Using the isolated screen's 2-warp/3-stage attention QKV winner yields
32.500 ms: only another **0.033 ms**. Missing per-shape tuning does not explain
the discrepancy. The original path is already much faster in the real
sequence than the projected 38.2 ms.

The exact QKV path alone saves approximately 0.703 ms in that FP8 sequence.
It is already part of the integrated bundle, so adding its saving to a full
FP8 replacement would double-count the same projections. No unvalidated
acceptance estimate belongs in a stack of guaranteed gains.

The measured streaming reader reached 238.8 GB/s. This is a result from that
harness, **not a proven physical bandwidth ceiling**. At 239 GB/s, streaming
17.555 GB of target projections would take about 73.5 ms; 4.735 tokens per
such traversal would be about 64.5 TPS. That arithmetic assumes every byte
must be fetched and the same bandwidth applies. It illustrates the small
byte-preserving budget, rather than proving that no other kernel can win.

## Remaining BS1 work, in order

1. **Use the validated exact bundle as the source baseline.** Future controls
   must identify whether it is enabled before graph capture. Preserve the
   FP8-draft baseline and report both decode and end-to-end throughput. A
   production image change is separate from source validation.
2. **Bound one post-sync overlap experiment.** The profile exposes roughly
   0.7–1.3 ms of host gap, so its total possible round benefit is around 1%.
   A correct prototype needs device-side acceptance and stop-truncated commit
   counts, pinned nonblocking transfer and an event, with reconstruction
   queued before the host wakes. Preserve stop/length behavior, committed
   recurrent state, buffer lifetime and fallback paths. A naive `.tolist()`
   after reconstruction waits for the whole stream and loses the overlap.
   This change is not implemented or counted as a gain in the current report.
3. **Build a representative draft-training corpus before training.** Use at
   least 500 independent long-context requests, with repository/task-family
   provenance and disjoint train/validation/test groups. Keep the frozen
   21-task screen as evaluation data. The default starting mix is code,
   design and reasoning; weight it to the actual BS1 workload when known.
4. **Train for proposal support and early acceptance.** Collect current
   target histories and the draft's required intermediate features. Improve
   the first proposal and coverage of likely target tokens, then validate
   sequential acceptance and complete generations. Avoid another score/rank
   selector fitted only to existing frozen candidates.
5. **Advance only on held-out throughput and quality.** Measure draft runtime
   as well as emitted tokens per round. Require a family throughput interval
   above 1, representative long natural completions, quality canaries and a
   final paired HTTP run. Stop or revise the mechanism if it fails; no gain
   from training is currently measured or promised.

The capture analysis finds approximately flat conditional acceptance by
position (about 0.79–0.83 after the first proposal), rather than a cliff at
position eight. First-position code acceptance is about 0.70, and 17.6% of
first rejections overall (29% on code) name a token absent from top-16 support.
The perfect-selection ceilings on native histories are 7.85 / 7.29 / 9.66
emitted tokens per round. Those are oracle ceilings on fixed histories,
not achievable policies or forecasts of a retrained draft.

## Work to defer

Do not repeat unchanged history-lookup, score/rank-only selector, uniform
FP4 target conversion or always-tree projects. Prior screens found the
history-lookup code gain came from the post-EOS artifact, the selector failed
held-out coverage, later target FP4 layers failed quality, and tree coverage
has not translated into sufficient complete-generation throughput.

Target compression with calibrated corrections remains a possible larger
research direction, but changes target numerics and needs separate quality
qualification. A smaller target, higher concurrency, or a warm-cache cell
changes the objective and cannot be reported as a win on this fixed BS1 cell.

## Evidence

- Current implementation and benchmark report: [gb10-bs1-speedups.md](gb10-bs1-speedups.md).
- Current raw runs: `$SHARAR_RESULTS/reviews/2026-09-10-bs1-speedup/`.
- Prior kernel family window: `$SHARAR_RESULTS/reviews/2026-09-10-family-eval/candidates-1/`.
- Original bandwidth screen: `$SHARAR_RESULTS/reviews/2026-09-10-family-eval/bandwidth-results.md`.
- Prior profiles and numerical investigations: [gb10-fp8-profile.md](gb10-fp8-profile.md),
  [gb10-verifier16.md](gb10-verifier16.md), [gb10-next-speedups.md](gb10-next-speedups.md).
