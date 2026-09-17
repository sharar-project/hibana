# Task accuracy: exact acceptance, the relaxed default and 3-bit decode weights

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

2026-09-16, main `afab63c` plus the working tree. The judged comparisons in
the [relaxed acceptance](relaxed-acceptance.md) and [3-bit decode](w3-decode.md)
studies rule out large quality losses but are same-model pairwise
judgments. This adds two standard task benchmarks with an objective score,
run identically for the three configurations through
`bench/accuracy_eval.py` against the OpenAI-compatible server: **GSM8K**
(the first 300 test problems, exact match on the final number after a
step-by-step answer) and **HumanEval** (all 164 problems, pass@1 with the
reference tests executed in a sandboxed subprocess). Greedy, thinking
disabled, concurrency four, the acceptance rule pinned per request. Exact
and relaxed ran on the production server; the 3-bit arm ran in an exclusive
window with `SHARAR_W3_DECODE=1` and exact acceptance, so it measures the
quantization alone.

| Arm | GSM8K (300) | gained / lost vs exact | p (McNemar) | HumanEval (164) | gained / lost | p |
|---|---:|---:|---:|---:|---:|---:|
| Exact acceptance, NVFP4/FP8 target | **95.0%** | | | **90.9%** | | |
| Relaxed default (top-2 within 1.0 logit) | 94.7% | 5 / 6 | 1.00 | 91.5% | 4 / 3 | 1.00 |
| 3-bit decode weights (exact acceptance) | 95.7% | 3 / 1 | 0.62 | 91.5% | 3 / 2 | 1.00 |

Per-item results are paired: on GSM8K the relaxed default solved five
problems exact missed and missed six exact solved; the 3-bit target solved
three and missed one. Completion lengths are within 2% across arms. At this
size a real accuracy change of about three points would show; smaller
differences would not, and the discordant counts above are what noise looks
like.

Throughput during the runs (aggregate tokens per second at concurrency
four, short prompts, both benchmarks): exact 220 and 260, relaxed 246 and
285, 3-bit 231 and 282. These are not the controlled cells of the other
studies (batch size varies as requests finish); the 3-bit path only serves
16-row rounds, which is why its gain here is smaller than the batch-one
gate's 1.17x.

## Reading

Neither the relaxed default nor round-to-nearest 3-bit decode weights
change GSM8K or HumanEval accuracy measurably. Together with the judged
comparisons this is enough to keep the relaxed default on and to ship the
3-bit path as an opt-in; it is not enough to call 3-bit lossless, and a
larger or harder evaluation (full GSM8K, a coding benchmark with hidden
tests, long-context tasks) remains the bar for making it the default.

## Evidence

`$SHARAR_RESULTS/reviews/2026-09-16-accuracy/`: `results/<arm>-<task>.json`
(per item: prediction, gold, correctness, tokens, sandbox error), the
window logs, and the combined `results.json` named in the
[manifest](accuracy-benchmarks-results.json). Datasets under
`$SHARAR_RESULTS/datasets/accuracy/` (GSM8K test set and HumanEval as
published by OpenAI).
