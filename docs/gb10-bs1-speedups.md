# BS1 throughput: implementation and measurements

2026-09-10. Completed. The integrated exact kernel bundle improves measured
BS1 HTTP decode throughput by **1.13–1.27%** and end-to-end throughput by
**0.99–1.05%**, preserving every output ID. The 1.5x target remains open.

This study follows [the 1.5x plan](gb10-1p5x-plan.md). The baseline is the
optimized **FP8 DFlash2 draft, k=15**, against the pinned Qwen3.8-27B NVFP4
target on one GB10. It is not the BF16-draft configuration in the restored
production container. Compare configurations explicitly when reporting gains.

## Final HTTP result

Four fresh server loads in native / integrated / integrated / native order;
four timed samples per variant/workload, with warmups excluded from averages.

| Workload | Decode TPS, before → after | Gain | End-to-end TPS, before → after | Gain |
|---|---:|---:|---:|---:|
| Code | 44.185 → 44.738 | **+1.25%** | 37.620 → 38.016 | **+1.05%** |
| Design | 34.241 → 34.628 | **+1.13%** | 30.167 → 30.466 | **+0.99%** |
| Reasoning | 56.418 → 57.135 | **+1.27%** | 46.114 → 46.594 | **+1.04%** |

Mean request latency falls from 26.582 → 26.305 s for code,
33.148 → 32.823 s for design, and 21.685 → 21.462 s for reasoning.
TTFT remains approximately 3.97 s. Every load uses the same 211 / 272 / 165
verification rounds for code / design / reasoning, so the gain comes from
lower execution cost, not different draft acceptance.

All **36 fixed-token requests** pass the cold-prefix and 10K/1K usage audit;
all outputs match across warmups, repetitions and variants. All **16 natural
quality canaries** pass. Both integrated loads improve every workload over
the opening control; the closing control confirms the baseline. These are
small repeated HTTP measurements, supported by the separate family study
below, rather than a claim of general quality coverage.

## Family results

Ratios below use both native controls, not only the opening baseline. The
kernel window's closing control is 0.16% slower across families; the draft
window's control is 0.05% slower. All 21 native outputs repeat exactly in
each window. Confidence intervals describe variation across the 18 tasks.

| Candidate | Family decode gain, 95% interval | Round cost change | Identical outputs / 21 | Decision |
|---|---:|---:|---:|---|
| Exact QKV + K512 MLP | **+1.43% [ +1.37%, +1.49% ]** | **−1.41%** | **21** | Integrated; HTTP validation passes |
| Triton M16 FP8 | +4.68% [ −1.02%, +11.07% ] | −2.07% | 0 | Experimental; throughput gate fails |
| Exact bundle + Triton FP8 | +5.59% [ −0.14%, +12.03% ] | −2.92% | 0 | Experimental; throughput gate fails |
| NVFP4 draft body | +2.51% [ −4.33%, +10.11% ] | −2.87% | 0 | Experimental; throughput gate fails |
| NVFP4 body + FP8 feature projection | +3.01% [ −3.47%, +9.80% ] | −3.21% | 0 | Experimental; throughput gate fails |
| Top-32 candidate support | −0.79% [ −4.03%, +2.48% ] | +0.29% | 8 | Keep top-16 |

The exact bundle's native benchmark prompts move from 44.13 / 34.26 / 56.45
to 44.70 / 34.71 / 57.24 decode TPS for code / design / reasoning, preserving
all output IDs. The numerical FP8 candidate instead measures
38.87 / 36.60 / 51.28 TPS on those prompts. Changed continuations make their
acceptance gains highly workload-dependent; the favorable aggregate point
estimate does not establish an improvement on unseen requests.

The draft-body candidates quantize 20 q/o/gate-up/down matrices from the BF16
checkpoint to NVFP4 weights with BF16 activations. Other draft matrices remain
BF16, with the second candidate additionally using FP8 for the feature
projection. Both are compared against the optimized FP8 draft. Their weight
kernel oracle passes, but that is a quantized-matmul check, not evidence of
unchanged draft quality. Neither candidate received a production-quality
qualification after failing the aggregate throughput gate.

## Why the FP8 estimate was too large

The prior 6 ms claim extrapolated isolated random-operand winners to a full
round. The new diagnostic captures the **actual 176 FP8 projections and
7,214,202,880 weight bytes**, including 64 output projections with the same
5120x6144 shape. CUDA graph replay of the whole distinct-weight sequence gives:

| Path | Matched original sequence | Candidate sequence | Saving |
|---|---:|---:|---:|
| Exact attention QKV only | 34.562 ms | 33.860 ms | 0.703 ms |
| Triton M16, original configuration | 34.573 ms | 32.532 ms | 2.040 ms |
| Triton M16, tuned attention QKV | 34.574 ms | 32.500 ms | 2.073 ms |

The original path is already faster here than the projected 38.2 ms.
The candidate is close to the projected 32.2 ms, so the exaggerated gap comes
mainly from the isolated baseline estimate. This experiment does not isolate
every cause of the microbenchmark discrepancy. It does establish that using
the omitted 2-warp/3-stage QKV configuration recovers only 0.033 ms for the
whole sequence, not the missing 4 ms. These measurements agree with the
approximately 2 ms saving in real model rounds.

The numerical candidate matches only 73 of 176 captured projection outputs
exactly; worst normalized RMS error is 7.16e-5. Small local differences are
enough to change long greedy continuations in this recurrent target. Keep
these kernels in `bench/`; do not count their savings as an exact speedup.
Their attention QKV work also overlaps the exact bundle, so those savings
cannot be added twice.

## Correctness validation

- **420 CPU tests passed**, 8 skipped.
- **53 GPU tests passed**: 9 real-weight projection/graph tests, 4 full-model
  lifecycle comparisons, and 40 existing target-numerics, NVFP4-rounding and
  accepted-feedback checks.
- **1,296 captured matrices match exactly**, across code/design/reasoning at
  output positions 32, 384 and 768.
- Projection graph tests change input values, amplitude and zero inputs at
  M=16/32/64. Complete-round comparisons cover BS1 length and forced-stop
  truncation, BS3 padded verification and BS4, checking output IDs, committed
  SSM/convolution state, private ownership and final page/slot recovery.

Tensor equality uses `torch.equal`; this checks values and does not distinguish
the sign of zero. The existing NVFP4 rounding tests separately check packed
boundary bytes. These are bounded numerical and lifecycle checks on the
pinned runtime, not a proof of equivalence for every possible tensor or future
dependency version.

## Implemented kernel dispatch

`sharar/kernels/verify_gemm.py` and `sharar/kernels/ops.py` specialize three
target projection shapes at M=16:

| Projection | N | K | Implementation |
|---|---:|---:|---|
| Attention QKV | 14,336 | 5,120 | FlashInfer FP8 `bmm_fp8`, cuBLAS backend |
| MLP gate/up | 34,816 | 5,120 | CuTe NVFP4, 64x128 tile, K tile 512 |
| MLP down | 5,120 | 17,408 | CuTe NVFP4, 64x128 tile, K tile 512 |

These cover 16 attention QKV and 128 MLP calls per BS1 verification round.
The MLP specialization requires the existing `cutlass` backend and BF16
output. Other row counts and projection shapes use the existing dispatch.
No checkpoint format or draft weights change. Each call owns its output;
the normal graph-capture warmup populates the compiler cache. The CuTe
compiler interface is private to the pinned FlashInfer 0.6.17 runtime.
The benchmark reuses the host's compilation cache; cold compiler startup and
an AOT image build are not measured by the generation TPS results.

`VERIFY_GEMM_ENABLED` is an internal benchmark control. It must be set before
graph capture; changing it afterwards does not alter captured graphs. The
HTTP comparison starts a fresh process for every variant.

## Measurement protocol

- One exclusive GPU window at a time, fixed image digest and frozen source
  snapshots. The harness stops the idle service, checks available memory
  throughout the run, enforces a timeout, and restores the original service
  and memory guard after each window.
- Native family evaluations: 18 synthetic task families (8 code, 5 design,
  5 reasoning), plus the three standard benchmark prompts. Each request is
  cold, greedy, 10,000 input tokens and exactly 1,000 output tokens with EOS
  ignored. Candidates are bracketed by fresh native processes. Final ratios
  use each task's mean opening/closing baseline, then a paired geometric mean
  and a bootstrap 95% interval across tasks.
- Exact integration: captured real operands, changed inputs under CUDA graph
  replay, complete speculative rounds and committed recurrent state, plus
  normal stopping and slot/page cleanup. Larger batches exercise fallback
  dispatch.
- HTTP: native / integrated / integrated / native, FP8 draft in every load.
  Each load runs code, design and reasoning through AIPerf, with one excluded
  full-generation warmup and two timed requests per workload. This gives
  24 timed requests and 12 warmups. Four natural-stop quality canaries run
  after each load. The client audits token IDs and usage.
- FP8 diagnostic: CUDA graphs replay the actual 176 distinct FP8 weight
  matrices and frozen quantized inputs, both as one sequence and grouped by
  shape. Three ABBA blocks per candidate use five cold replays per entry.
  These kernel timings are reported separately from generation throughput.

The code benchmark reaches EOS around output index 98 in the baseline. Its
remaining forced output is not a representative natural completion. Changes
that alter output IDs therefore need the family comparison and independent
quality evidence; a favorable number on this one prompt is insufficient.
The 18 families are a bounded synthetic screen, not a representative training
corpus or a comprehensive quality evaluation.

## Scope and remaining work

The exact dispatch is enabled in source. The numerical FP8, compressed-draft
and top-32 paths remain benchmark candidates. No draft training, target
checkpoint conversion, host-overlap implementation or BS4 throughput tuning
was performed in this execution. The [revised plan](gb10-1p5x-plan.md) states
the remaining work and corpus requirements; the earlier extrapolated
1.12–1.18x engineering stack is withdrawn.

The original production container/image and launch settings are restored,
`/health` returns 200, the memory guard is active, and all three temporary
containers are gone. No production image was built or promoted. The automatic
memory planner sizes spare KV capacity at startup: it now has 498,112 rather
than 498,395 pages (about 8.8 MiB less). The initial restoration audit treated
that as a fixed configuration field; inspection confirmed automatic sizing
in `Ledger.plan`, and the final audit separately records the pool change and
checks the original static configuration.

Minimum available memory was 76.73 GiB during draft screening, 80.63 GiB
during GPU validation, and 80.31 GiB during the HTTP window. All windows
completed within their guards.

## Evidence and reproduction

Evidence root: `$SHARAR_RESULTS/reviews/2026-09-10-bs1-speedup/`.
The prior kernel-family window is
`$SHARAR_RESULTS/reviews/2026-09-10-family-eval/candidates-1/`.

- `protocol.json`, source manifests and `source-baseline/` /
`source-integrated/` identify the configurations.
- `draft-screen/` contains native / NVFP4 body / body plus FP8 feature
  projection / top32 / native evaluations and output IDs.
- `gpu-tests.xml`, `cpu-tests.xml`, and `kernel-validation/report.json`
  contain correctness and diagnostic results.
- `http-abba/` contains client commands, server source hashes, raw AIPerf
  artifacts, token audits, warmup/timed results and quality canaries.
- `summary.json` aggregates the complete runs; `artifact-index.json` hashes
  the evidence files. `window.py`, `gpu_validation.py`, `http_coordinator.py`
  and `summarize.py` reproduce the guarded validation and aggregation.
- `final-health.json` records restoration and resource minima;
  `final-source-manifest.json` records the delivered source. Runtime and test
  code match the GPU-tested snapshot. Later edits are documentation and a
  benchmark protocol label clarifying that tensor equality uses `torch.equal`.

Do not run a GPU benchmark alongside the serving container. The recorded
window harness handles exclusive access and restoration. Run
`bench.gb10_bs1_kernels` with the frozen capture directory only inside such
a window. It does not measure end-to-end serving performance.
