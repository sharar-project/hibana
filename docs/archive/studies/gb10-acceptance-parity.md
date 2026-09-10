# GB10 acceptance parity investigation

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

2026-09-08. Follow-up to the [optimization measurements](gb10-optimization-results.md).
Evidence: `${SHARAR_RESULTS}/reviews/2026-09-08-acceptance-parity/`.

**Result:** the controlled comparison does not reproduce a large draft
acceptance deficit. Reference arithmetic variants were implemented and tested,
but none qualifies as a serving speedup. The most promising norm variant fails
the strict JSON canary and regresses reasoning; serving retains its existing
arithmetic. The original production image is not rebuilt or deployed.

## Finding: the generation-round gap was mostly a trajectory comparison

The earlier code benchmark used identical 10,000 input IDs and 1,000 output
tokens, but the engines generated different text. SGLang produced a repeated
tool-call sequence; Sharar produced a written design explanation. The measured
264/266 versus 205 verification rounds therefore did **not** establish a
comparable deficit in draft prediction accuracy. The completion-time comparison
remains valid for those complete configurations and continuations.

An instrumented run of the pinned SGLang image reproduces its **205 counted
verification rounds**. It records committed tokens, target predictions, target
features, draft KV, hidden states and proposals. Startup warmup captures are
identified by their absolute positions and excluded from the 10k replay. One
additional in-flight round after the requested output budget is retained in the
raw trace and excluded from the reference benchmark's 205 rounds.

Sharar then follows **128 of SGLang's exact token histories and verification
boundaries**. It computes its own target features and recurrent state from those
histories. Its proposals are compared with the recorded continuation; its
target predictions are recorded separately. Committing the reference tokens is
deliberate teacher forcing, not ordinary generation or a latency benchmark.

| Arithmetic | Draft prefix matches, total | Reference total | Prefix matches + one correction per history | Target top-1 agreement on committed inputs |
|---|---:|---:|---:|---:|
| Original Sharar | 481 | 489 | 609 vs 617 | 596 / 617 |
| FP32 residual-sum normalization | 477 | 489 | 605 vs 617 | 592 / 617 |
| Normalization + reference FP16 weight-loading intermediate | 489 | 489 | 617 vs 617 | 595 / 617 |

The original gap is **1.6% in matched draft tokens**, or **1.3% including one
correction per history**, on this controlled sample. Matching aggregate totals
does not mean matching individual rounds: the last variant is better in 12
histories, worse in 13 and equal in 103. The normalization correction alone does
not improve this fixed-history score. These results do not support attributing
the earlier approximately 30% extra generation rounds to a weak selector, or
making selector retraining the next priority on that evidence alone.

This is a sample from one code continuation, not a universal parity result.
Original Sharar's first target token is also different (ID 2 versus ID 40),
before a draft has influenced the continuation. The target's numerical path
therefore already separates the generated histories at the start.

## Confirmed arithmetic differences and implementation

The source comparison uses SGLang image
`sha256:60166a99b6610e2b4b3febbeb6411e4cbc9338e996ab6c80d3339f86a7b5b9c3`,
source `5f55db35e`, the same target revision
`319f741cce68d7914884900c138a1fbb70a42f30` and DFlash2 revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4` as the previous benchmark.
The instrumented package is a local copy; the original image is unchanged.

**Implemented as an opt-in experiment: target residual normalization.** SGLang's actual CUDA norm,
tested on identical BF16 operands, normalizes `x.float() + residual.float()`.
Sharar rounds that sum to BF16 before computing both its variance
and normalized numerator. The experimental kernels normalize the FP32 sum while
still writing a BF16 residual stream. Normalized activations still round to
BF16 before FP8 or NVFP4 quantization. Both `fused_norm.py` and the optional
`fused_norm_nvfp4.py` expose this rule through `fp32_residual_sum=True`.
The default is `False`, preserving the existing serving arithmetic after the
quality failure below. Only the benchmark scripts expose a configuration switch,
`--target-residual-sum fp32`; there is no serving CLI flag. The experiment does
not change the checkpoint, KV format, acceptance rule or state precision.

**Measured, retained as diagnostics:**

- Draft RMSNorm casts before its weight multiply in Sharar; SGLang casts after
  that multiply. On a captured primitive input, the FP32-weight equation matches
  SGLang exactly, while the original equation differs by 0.27% normalized RMS.
- SGLang fuses residual addition with draft normalization, convolution and SiLU
  multiplication at different rounding boundaries, and fuses QKV projection.
- SGLang's captured draft KV uses FP8 E4M3; Sharar's draft KV uses BF16. SGLang's
  draft RoPE table is FP32; Sharar casts its factors to BF16 before rotation.
- SGLang's FP8 weight loader dequantizes through FP16 before requantization;
  Sharar uses FP32. A CPU scan of the checkpoint finds 4,345,873 differing
  packed values among 7,214,202,880, concentrated in a few projections. This
  variant was also executed on the GPU. It remains a diagnostic because it
  regresses the measured code workload. Source comments no longer claim that
  the resulting weight bytes are identical.

Proposal/anchor indexing, accepted-prefix feedback, feature-layer indexing,
the greedy acceptance rule and the configured sliding-window extent agree in
the inspected implementations. Four captured 10k draft states were additionally
replayed with shared SGLang KV, isolating draft arithmetic from target features.
The FP32-score selector variant reproduces SGLang's selected paths on its own
captured hidden states, despite differences in top-k tie ordering. This does
not assert bitwise parity of the full draft backbone.

The source comparisons are reproducible from the pinned files retained in the
evidence directory:

| Area | Sharar | SGLang source under `python/sglang/` |
|---|---|---|
| Draft layers, norms, RoPE, convolution | `model/parallel_draft.py` | `srt/models/dflash.py` (`DFlashDecoderLayer`, `DFlashDraftModel`) |
| Candidate support and conditional selection | `ParallelDraftModel.selector` | `DFlash2DraftModel`, `dflash_worker_v2._selector_lattice` |
| Anchor, position and accepted-feature feedback | `proposers/parallel.py` | `srt/speculative/dflash_worker_v2.py` |
| Greedy commit boundary | `engine/spec.py`, `engine/rejection.py` | `dflash_worker_v2._commit_accept`, `srt/speculative/dflash_utils.py` |
| FP8 weight requantization | `weights/modelopt.py` | `srt/layers/quantization/utils.py::requantize_with_max_scale` |

The target residual equation was checked against the installed CUDA operation
on captured operands, not inferred solely from Python wrappers.

## Single-sample native screens

These are exploratory full generations, not paired serving results. Each cell
has 10,000 input IDs and exactly 1,000 output tokens; graph capture and warmup
are excluded. Wider arithmetic changes can change the generated continuation.
The complete failed and successful trials are retained.

| Variant | Code seconds / rounds | Design seconds / rounds | Reasoning seconds / rounds |
|---|---:|---:|---:|
| Target normalization only, BF16 draft | 32.085 / 240 | 32.814 / 247 | 25.608 / 187 |
| Draft arithmetic changes, BF16 KV | 35.769 / 267 | 35.106 / 263 | 23.134 / 163 |
| Draft arithmetic changes, FP8 KV | 36.851 / 276 | 32.434 / 239 | 23.940 / 169 |
| Both normalization paths, FP8 draft KV | 34.491 / 258 | 31.992 / 235 | 26.097 / 188 |
| Target normalization + FP16 weight-loading intermediate | 36.437 / 277 | 32.552 / 246 | 25.426 / 185 |
| Same target changes, FP8 draft matrices | 34.684 / 278 | 31.200 / 247 | 24.508 / 188 |
| Target normalization only, FP8 draft matrices | 35.076 / 282 | 28.356 / 223 | 25.988 / 201 |
| FP16 weight-loading intermediate only | 36.256 / 274 | 29.854 / 220 | 27.121 / 200 |

The FP8 draft improvement from the earlier target arithmetic does not transfer
uniformly to the FP32-sum target. BF16 remains the draft default. Reference
weight loading and the wider draft arithmetic variants are not enabled in
serving. No training, new quantized checkpoint or altered acceptance rule is
introduced.

## Paired serving validation

The experiment compares the current fused-convolution BF16 draft with
the original target normalization against the FP32-sum target normalization.
The planned load order was **old / FP32 / FP32 / old**, with an excluded full-generation
warmup for each workload in each load, followed by one measured sample. Every
request uses the frozen IDs, greedy sampling, ignored EOS and a flushed cache.
Each load also runs the four answer canaries. Pool sizes, prefill chunks,
verification width and checkpoint revisions are held fixed.

**The sequence stopped after the first FP32 load failed its quality gate.**
There is only one measured sample per arithmetic/workload; the planned paired
experiment is incomplete. Its raw observations are:

| Workload, B1, 10k/1k | Existing target | FP32-sum experiment | Latency change |
|---|---:|---:|---:|
| Code | 35.481 s | 32.490 s | −8.43% |
| Design | 34.625 s | 33.308 s | −3.80% |
| Reasoning | 24.106 s | 26.207 s | +8.72% |
| Answer canaries | 4 / 4 | 3 / 4 | Strict JSON fails |

The failed response wraps the requested JSON in Markdown fences, so the
unchanged strict parser rejects it. Removing the fences in evaluation or
relaxing the prompt would hide the failure. The exact-instruction, arithmetic
and long-context needle canaries pass. The experimental source was returned to
an opt-in mode after this result. These observations are not a quality-qualified
speedup, and do not establish a tail-latency or statistically significant gain.

An independent fresh-load check then runs the unchanged `local_v1.json` suite
under each arithmetic mode: **78 requests each, 156 total**, including warmups,
cache primers, mixed arrivals, cancellation and short output budgets.

| Validation | Existing target | FP32-sum experiment |
|---|---:|---:|
| Request errors | 0 / 78 | 0 / 78 |
| Automatic output failures | 0 | 12 |
| Fresh-load answer canaries | 4 / 4 | 3 / 4 |

The 12 failures are four cached-code trials and eight JSON trials, including
three warmups. All wrap the requested raw code or JSON in Markdown fences.
The candidate therefore fails on two independent model loads. The suite's
manual text rubrics remain separate from these automatic counts; no production
promotion or broader answer-quality claim is made. The prompts, output parsers,
repetitions and cache expectations are unchanged for both modes.

## Consequence for optimization work

The final **uninstrumented** SGLang refresh takes **30.352 and 30.657 seconds**
(mean **30.505 s**) on the exact frozen B1 code input and 1,000 output tokens,
with an excluded full-generation warmup. The current BF16 Sharar control's
single measured **35.481 s** is **16.3% longer**. This is a small-sample complete
configuration comparison, with different generated continuations, not an
identical-output or useful-task completion guarantee. The rejected norm
candidate does not establish a new competitive result. The earlier explicit
FP8-draft measurements remain in the previous report; FP8 is not the default.

The fresh SGLang load reports **201 rounds**, versus **205** in the retained
instrumented trace and earlier uninstrumented reference. Its response hash
differs from that earlier reference. These runs are kept separate: the
same-history comparison uses the recorded 205-round trace, and the fresh
latencies describe the new continuation. Pinning the configuration did not
guarantee the same generated trajectory across these processes; the precise
cause of this additional numerical variation has not been isolated.

The large generation-round difference is not evidence of a comparably large
draft porting loss. Retain the same-history comparison for subsequent changes,
and expand it to held-out continuations before making general parity claims.
Matching more reference rounding points can change generated content without
improving fixed-history proposal agreement or preserving output requirements.

There is no new quality-qualified serving speedup from this investigation.
The earlier kernel profile still points to target matrix traffic and complete
round cost as major constraints. Calibrated target compression and a cheaper
draft remain research paths, with the failed conversion and quality screens
in [the optimization report](gb10-optimization-results.md) setting their limits.
Neither copying reference arithmetic nor retraining the selector has been
shown here to produce a wide win.

## Regression checks and reproducibility

- All seven original FP32-sum regression tests fail on the original kernel and
  pass with the experimental equation. That proves the arithmetic difference;
  it does not prove better model quality.
- **108 final GPU tests pass**, covering both residual rounding modes, graph
  replay, fused quantization, draft lifecycle in BF16 and FP8, and accepted-state
  reconstruction/commit behavior. The earlier always-on FP32 candidate passed
  100 tests before the quality gate prevented promotion.
- The final full CPU suite passes **349 tests**, with eight GPU/snapshot-dependent
  cases skipped, in 9.84 seconds. All **eight focused replay tests** subsequently
  pass, including three added padding-boundary cases. The initial suite stalled
  in the network-isolated sandbox; host-local access resolves the async-test stall.

`bench/gb10_acceptance_replay.py` validates the reference trace and performs the
teacher-forced comparison. It rejects inconsistent token histories, wrong
checkpoint revisions and insufficient lookahead. In an exclusive guarded GPU
container with this evidence directory mounted at `/proof`:

```bash
python3 -m bench.gb10_acceptance_replay \
  --reference /proof/reference-code.json \
  --output /proof/new-replay.json --rounds 128
```

Use a separate process and `--target-residual-sum fp32` for the experimental
equation. The default uses the existing BF16 residual-sum target, BF16 draft
and exact fused draft convolution. The replay never imports SGLang. The
reference exporter, instrumented source copy and reference JSON stay outside
the engine tree; no reference library is a runtime dependency.

Raw tensors, continuations, source snapshots, launch commands and per-window
resource logs are retained outside git. Each GPU window restores the original
Sharar service and memory guard. The production image is not rebuilt or deployed
by this investigation.

The final service check confirms the original container ID and image digest,
HTTP health 200, an active memory guard, and the pinned model/profile plus
deterministic chat canary in `tools/verify-sharar.sh`. Final host available
memory is **56.97 GiB**; the last SGLang benchmark window stays above **63.25
GiB**, against the 16 GiB guard floor. The shipped replay reproduces both
128-history totals exactly, including after testing padding that guarantees
the recorded rejection boundary.

[Machine-readable measurements and evidence archive](gb10-acceptance-results.json).

Metric caveat: some large-tensor cosine values in the early `replay-report.json`
were accumulated incorrectly in FP32 and exceed one. Those entries are not used
for conclusions. Subsequent screens use FP64 accumulation; the early NRMSE
values and discrete token/path counts are independent of that cosine metric.
