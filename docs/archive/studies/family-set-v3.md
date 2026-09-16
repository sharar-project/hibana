# The 101-family set: the gate's resolution at three percent

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

2026-09-16, main `658a01f` plus the working tree. Follow-up to the
[family gate study](family-gate.md), which measured that any change moving
continuations carries a ±7% interval on tokens per round with 18 families,
because a task's acceptance swings by 11 to 17% either way when its text
changes. The fix is arithmetic: the interval shrinks with the square root of
the task count. This adds 83 families and measures what the larger set
resolves.

## The set

`bench/workloads/family_v3.json` holds the original 18 families and 83 new
ones written on 2026-09-15 in the same form (a specification of a few
sentences, a question asking for complete code, a design or a derivation
with tests or worked arithmetic): 40 code, 31 design and 30 reasoning
families in all, split train/validation/test by a fixed rule (roughly
60/20/20 per category). `bench/family_inputs.py freeze --protocol
bench/workloads/family_v3.json --extra-code-prompt` materializes each to
exactly 10,000 input tokens with the synthetic record padding and appends
the four benchmark prompts: `family-eval-v3`, 105 tasks, SHA-256
`b72bdecd...`, now the gate's default (`--set family-eval-v2` and `v1`
remain selectable and hash-checked). The set was frozen once and has not
been tuned against any result. Every one of the 101 families runs the full
1,000 greedy tokens without emitting EOS under exact acceptance.

## The run

`tools/family-gate.py --set family-eval-v3 --order N,C1,N` with
`env:SHARAR_STABLE_NUMERICS=1` as the candidate, the same continuation-only
control the 18-family run used: three passes of 105 tasks, about 48 minutes
each, all tasks ok, in one exclusive window.

| Comparison, 101 families | Decode TPS | Round cost | Tokens per round | Identical outputs |
|---|---|---|---|---:|
| Native repeat vs native (noise floor) | x0.9984 [0.9982, 0.9986] | x1.0016 | x1.0000 | 105 / 105 |
| Stable numerics vs native | x0.9952 [0.9658, 1.0236] | x1.0017 [1.0015, 1.0019] | x0.9969 [0.9674, 1.0254] | 0 / 101 |

Family means: native 48.0 TPS, 5.05 tokens per round, 105.2 ms per round;
the repeat 47.9 / 5.05 / 105.3.

The tokens-per-round interval is now **±3.0%** (half-width), against ±7%
on 18 families, with the same per-task spread (log standard deviation
0.152; 0.17 on code families, 0.13 on design and reasoning). The noise floor
is 0.15% in throughput with identical outputs on every task, and the round
cost still resolves to 0.02%. Stable numerics reads −0.3% on acceptance
[−3.3%, +2.5%] and +0.17% on round cost, cost-neutral and acceptance-neutral
at this resolution, as the design of the change predicts.

## What this changes

- Acceptance claims of about 3% and above are now testable in one gate run
  (three passes, two and a half hours). Draft-capacity work, training
  recipes and relaxed-acceptance variants can be judged on the family set
  instead of being reported as unresolved.
- Claims of 1 to 2% still are not: that needs about 400 to 1,000 families
  or, for drafter changes, the per-request replay over hundreds of real
  traces. The 18-family results in earlier studies keep their ±7%
  reading; do not re-interpret them.
- The cost side is unchanged: round-cost decisions were already exact and
  remain so.

## Evidence

`$SHARAR_RESULTS/reviews/2026-09-15-family-gate-v3/v3-baseline/`: the
three pass logs and reports, `comparison-C1.json`, `comparison-noise.json`,
`manifest.json`, `resources.jsonl`, and the combined `results.json` named
in the [manifest](family-set-v3-results.json). The frozen set is under
`$SHARAR_RESULTS/datasets/family-eval-v3/`.
