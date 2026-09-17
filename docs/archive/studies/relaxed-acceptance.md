# Relaxed speculative acceptance: opt-in speed mode, batch one and batch four

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

2026-09-15, main `24a1c77` plus the working-tree change. First item of the
[research directions](../../../roadmap.md#research-directions-2026-09-15):
accept a drafted token not only when it is the target's argmax but also when
the target ranks it within its top-k and within a logit margin of the argmax.
This is deliberately lossy, so it is a per-request setting
(`accept_top_k`, `accept_margin`), off by default; the default path is
unchanged and every exact request in a mixed batch stays exact.

## What was built

- `SamplingParams.accept_top_k` / `accept_margin` (validated; requires
  temperature 0), parsed on every generation route; `Request.spec_relaxed`
  and the per-request `spec_relaxed_tokens` count the emitted tokens that
  were accepted by the relaxed rule rather than as the argmax;
  `spec_stats.relaxed_accepts` totals them.
- `SpecDecoder._relaxed_matches` computes, from the logits the verify graph
  already materializes, each drafted token's rank and logit gap against the
  argmax and applies the per-row thresholds; `_accept` ORs it into the exact
  match and downloads the relaxed count in the same single sync. Nothing
  runs when no request in the batch asks for it.
- Four CPU tests (`tests/unit/test_relaxed_accept.py`): parsing and
  validation, the margin boundary on the tiny target, counters, pools, and a
  mixed batch whose exact rows stay exact. The full suite passes (526).

## Protocol

One exclusive window (`serve_window.py`, production and the guard stopped
and restored), FP8 pinned DFlash2 at k=15, the working tree mounted
read-only. `bench/draft_ab.py` replayed the 29 repository-coding test prompts
(`traces-test`) with their recorded output budgets, greedy, at batch one and
at batch four (`--concurrency 4`, closed loop: a lane sends its next prompt
when its previous one completes), and the 24 newest held-out ShareGPT
prompts at batch one. Two settings: top-2 within 1.0 logit (the drafted
token has at least 37% of the top probability) and top-3 within 2.0 logits
(at least 14%).

## Results

Aggregate decode throughput is total tokens over total wall time; tokens per
round come from the server's counters; "relaxed" is the share of emitted
tokens that were not the target's argmax.

| Cell | Setting | Before TPS | After TPS | Change | Tokens per round | Relaxed tokens | Identical outputs | Median first divergence |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Coding, batch one | top-2, 1.0 | 45.47 | 49.91 | **+9.8%** | 4.884 to 5.358 (+9.7%) | 4.05% | 3 / 29 | token 13 |
| Coding, batch one | top-3, 2.0 | 45.47 | 55.31 | **+21.6%** | 4.884 to 6.037 (+23.6%) | 7.02% | 3 / 29 | token 5 |
| Coding, batch four | top-2, 1.0 | 122.27 | 129.19 | **+5.7%** | 17.65 to 19.00 per batch round (+7.7%) | 3.97% | 3 / 29 | token 22 |
| ShareGPT, batch one | top-2, 1.0 | 35.47 | 40.72 | **+14.8%** | 3.714 to 4.310 (+16.0%) | 4.96% | 0 / 24 | token 6.5 |

Paired per-request throughput ratios (medians): 1.042 and 1.237 at batch
one on coding, 1.090 at batch four, 1.136 on ShareGPT. Batch-four aggregate
gains are diluted by the closed loop's uneven tail (29 prompts over four
lanes), which is why the batch-round tokens rose 7.7% while aggregate
throughput rose 5.7%.

## Reading the numbers

- The speed side is what the byte budget predicts: rounds are fixed cost,
  so throughput follows accepted tokens per round almost one for one, and
  the extra acceptance costs nothing (round time unchanged).
- The price is explicit and small in count: at top-2 within 1.0 logit, one
  emitted token in 25 is a draft token the target ranked second with at
  least 37% of the top probability; the rest are the target's own argmaxes
  conditioned on that prefix. At top-3 within 2.0 logits it is one in 14 at
  a 14% floor. Outputs therefore diverge from the exact greedy text early
  (median token 5 to 22), as any change to a near-tie does; that says
  nothing about quality by itself.
- What this study does not establish is task quality. The relaxed share
  is a lossiness count, not a judgment. Before any default changes, the two
  settings need a judged comparison on real traffic (the target model
  itself can score pairs offline in a training window). The family gate's
  cost interval shows nothing here because cost does not move; the decision
  is acceptance versus quality.

## Judged comparison

A second window generated natural-length completions (no forced budget,
1,024-token cap, greedy) for the same 53 prompts (29 coding, 24 ShareGPT)
under exact acceptance and three relaxed settings, at concurrency four, and
then had the target model judge each relaxed answer against the exact answer
for the same prompt, in both presentation orders, with a rubric that weighs
correctness first and ignores length. A setting "wins" a prompt only when
the judge prefers it in both orders; split or tied verdicts count as ties.
The judge ran with thinking disabled and a 700-token budget after a first
pass with thinking on left a third of the pairs without a verdict; the two
passes agree in direction where both have verdicts.

| Setting | Aggregate TPS at concurrency 4 | Relaxed tokens | Relaxed wins | Exact wins | Tie or split | Sign test p |
|---|---:|---:|---:|---:|---:|---:|
| top-2, 0.5 logit | 130.9 to 139.6 (+6.6%) | 2.5% | 11 | 18 | 24 | 0.26 |
| top-2, 1.0 logit | 130.9 to 144.3 (+10.3%) | 3.9% | 13 | 14 | 26 | 1.0 |
| top-3, 2.0 logits | 130.9 to 163.7 (+25.1%) | 7.4% | 11 | 15 | 27 | 0.56 |

Completion lengths and the share of answers that ended at EOS are the same
across arms (mean 835 to 842 tokens, 20 to 22 of 53 finished). Where the
judge did prefer the exact answer, its notes describe different analyses
(one answer found a bug the other missed), not broken text: a near-tie
early in the answer sends the two continuations down different but equally
fluent paths.

What this establishes and what it does not: on 53 prompts with a same-model
judge, no setting is distinguishable from exact acceptance. Pooled over the
three settings the decisive verdicts lean 47 to 35 toward exact (p about
0.2), so a small real preference for exact is plausible and a large one is
excluded: with about 27 decisive pairs per setting, an exact win share above
roughly two in three would have shown. A same-model judge may also favor
the model's own most-likely text. Detecting a 55/45 preference needs about
400 decisive pairs, which means a few hundred prompts per setting.

## Task accuracy

With the default setting (top-2 within 1.0 logit): GSM8K (300 problems)
94.7% against 95.0% exact, HumanEval (164) 91.5% against 90.9%, neither
difference significant (paired McNemar p 1.0 for both):
[accuracy benchmarks](accuracy-benchmarks.md).

## Decision

Merge as an opt-in feature. For interactive use, top-2 within 1.0 logit is
a defensible default: +10% at batch one and concurrency four, +15% on chat
prompts, 4% of tokens non-argmax, and no detectable quality difference at
this size. Keep exact acceptance for benchmarks, for parity work, and for
requests whose traces feed drafter training (a relaxed round teaches the
drafter its own choices; traces mark such rounds with `rx`, so the trainer
can skip them). The wider setting buys another 10 to 15% and also showed no
detectable loss, but one emitted token in 14 at a 14% probability floor is
a larger bet than the evidence here supports as a default.

The parameters are documented in the [API guide](../../api.md).

## Evidence

`$SHARAR_RESULTS/reviews/2026-09-15-relaxed-accept/`: `serve_window.py`,
`relax_client.sh`, `compare.py`, `ab/*.json` (per request tokens, seconds,
tokens per round, relaxed tokens, output hashes and IDs), the final server
info, `judge_client.py`, `judge_report.py`, `judge/*.json` (completions,
both judging passes with every verdict and note), and the combined
`results.json` named in the
[manifest](relaxed-acceptance-results.json).
