# The family gate: one command, a fixed code prompt, and what it can resolve

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

2026-09-15, source `1feb8c6` plus the working tree (lookup branch opt-in,
stable numerics opt-in, the gate tooling). Roadmap item 2: replace the
bit-exact output gate and the three-prompt benchmark as decision tools with a
paired family evaluation, fix the code prompt, make the shape-stable kernels
available, and re-judge the shelved candidates with it.

## What changed

- **`tools/family-gate.py`** runs `bench.gb10_family_eval` passes in exclusive
  windows (one fresh container per pass, working tree mounted read-only,
  production and the memory guard stopped and restored), pairs every
  candidate against the first native pass and reports the noise floor from
  the last one. Candidates are hooks in `bench/gate_hooks.py` or
  `env:KEY=VALUE`; `|bf16` pairs a BF16-draft candidate with a BF16 native
  pass. `bench/family_inputs.py freeze` regenerates the frozen inputs from
  `bench/workloads/family_v1.json` and the benchmark prompts, byte-identical
  to the 2026-09-10 file (SHA-256 `0c0da566...`); the gate verifies the hash.
- **The code benchmark prompt.** The old prompt made the model answer with a
  `<tool_call>` and end its turn at output token 98, so 902 of its 1,000
  tokens were a forced post-EOS continuation of tool calls. Three tool-free
  variants were probed on the GB10; all ran 1,000 greedy tokens with no EOS.
  The one that asks for explanation, a complete module, a full pytest module
  and an extension in that order is now `code` (`gb10-coding-v2`, 36.4 TPS
  and 3.84 tokens per round against the old prompt's 44.9 and 4.73, which
  were inflated by the degenerate tail). The old prompt stays as `code_v1`;
  both digests are pinned in `tests/unit/test_benchmark_cli.py`. The family
  set `family-eval-v2` (22 tasks, SHA-256 `fc733283...`) adds the new prompt
  as `benchmark_code_v2` and keeps the old one; it is the gate's default.
- **Shape-stable numerics in main.** The M-invariant b/a projection, FP8
  window padding and constant rows-per-block from `feat/stable-numerics` are
  merged behind `SHARAR_STABLE_NUMERICS=1`, default off, with their CPU and
  GPU tests; `bench.gate_hooks:stable_numerics` or the environment variable
  selects them for a pass.

## The gate run

Six passes over the 22 tasks (10,000 input tokens, 1,000 greedy output tokens,
cold prefix cache, k=15 DFlash2), in this order: native (FP8 draft), stable
numerics, Triton FP8 at sixteen rows (`fp8_m16`), native again, native with
the BF16 draft, and the NVFP4 draft body (W4A16, paired with the BF16 pass).
Paired per-task log-ratios over the 18 families with bootstrap 95% intervals:

| Comparison | Decode TPS | Round cost | Tokens per round | Identical outputs |
|---|---|---|---|---:|
| Native repeat vs native (noise floor) | x1.0011 [1.0007, 1.0014] | x0.9991 [0.9987, 0.9995] | x1.0000 | 22 / 22 |
| Stable numerics vs native | x1.011 [0.934, 1.094] | x1.0010 [1.0005, 1.0015] | x1.012 [0.935, 1.095] | 0 / 18 |
| `fp8_m16` vs native | x1.041 [0.984, 1.105] | **x0.9847 [0.9842, 0.9852]** | x1.025 [0.969, 1.088] | 0 / 18 |
| NVFP4 draft body vs BF16 draft | x1.065 [1.012, 1.117] | **x0.9136 [0.9132, 0.9140]** | x0.973 [0.924, 1.020] | 0 / 18 |
| NVFP4 draft body vs FP8 draft | x1.025 [0.957, 1.101] | **x0.9714 [0.9708, 0.9720]** | x0.996 [0.929, 1.070] | 0 / 18 |
| BF16 draft vs FP8 draft | x0.963 [0.910, 1.022] | x1.0633 [1.0626, 1.0639] | x1.024 [0.967, 1.086] | 0 / 18 |

Absolute family means: native 48.2 TPS, 5.08 tokens per round, 105.4 ms per
round; the repeat 48.3 / 5.08 / 105.4. Every pass completed all 22 tasks with
clean pools. Under stable numerics the old code prompt no longer emits a tool
call at all (no EOS in 1,000 tokens), a reminder that its behavior was a
near-tie, not a property of the task.

## What the gate resolves, and what it cannot

The machine is quiet: two native passes agree to 0.1% in throughput with
identical outputs on all 22 tasks. Round cost is independent of the
continuation, so the gate resolves it to about 0.05%: `fp8_m16` saves
1.53% per round, the NVFP4 draft body saves 8.6% against the BF16 draft and
2.9% against the FP8 draft, the BF16 draft costs 6.3% more per round than
FP8. Those are decisions.

Accepted tokens per round are not resolvable at this size. Any change to
arithmetic or to the drafter changes every continuation (0 of 18 outputs
identical in every candidate pass, first differences within the first few
dozen tokens), and a changed continuation swings a task's acceptance by
tens of percent in either direction. The per-task log standard deviation of
the tokens-per-round ratio is 0.11 to 0.17 across these comparisons, so the
95% half-width on 18 families is about 7%; it would take about 100 families
for 3%, 400 for 1.5%, and 1,000 for 1%. The stable-numerics pass shows this
in its purest form: nothing about the drafter changed, round cost is
unchanged, and the family estimate still reads +1.2% with an interval from
−6.5% to +9.5%.

Decision rule from here on:

1. Judge a candidate's **cost** on the round-cost interval (families row).
   That is where kernel and precision work shows, and the gate resolves it
   to a tenth of a percent.
2. Treat a candidate's **acceptance** effect as unknown unless the interval
   excludes 1 at the size run. A point estimate inside ±7% on 18 families is
   noise; do not select among variants on it, and do not sum it with a cost
   gain. When acceptance is the claim (a drafter, a selector, a wider
   block), grow the task set until the interval is useful, or measure on
   many real traces with `bench/draft_ab.py`, whose per-request pairing
   averages the same drift over hundreds of requests.
3. Bit-identical outputs are neither required nor evidence: the native
   arithmetic already moves with the row count.

## Re-judged candidates

- **`fp8_m16`** (Triton weight-streaming FP8 at sixteen rows, all 176 target
  projections): round cost −1.53% [−1.58, −1.48], acceptance unresolved
  (+2.5% [−3.1, +8.8]). Under the old bit-exact gate it was shelved because
  0 of 21 outputs matched; under this gate it is a clean cost win. Worth
  integrating into `ops.fp8_linear` at M=16 (the kernel and its validation
  live in `bench/archive/gb10_fp8_m16.py`).
- **NVFP4 draft body** (W4A16 draft matrices): −8.6% round cost against the
  BF16 draft and −2.9% against the FP8 draft, acceptance unresolved (−2.7%
  [−7.6, +2.0] and −0.4% [−7.1, +7.0]). Cheaper than FP8 on bytes; whether it
  gives up acceptance needs a large-trace A/B before it replaces FP8, and it
  would need FP8-style recalibration after each training session.
- **Stable numerics**: cost-neutral (+0.10% per round), acceptance
  unresolved. Remains the research baseline, not a serving change.
- **The 16-row tree and the SiLU boundary** were not re-run: the tree lives
  on branches that predate the draft-training merges and has no hook; the
  SiLU change was a target-arithmetic edit judged on the old code prompt.
  Both would need porting before the gate can see them.

## Evidence

`$SHARAR_RESULTS/reviews/2026-09-14-family-gate/`: `code-prompt-probe/`
(the three prompt variants), `gate-1/` (six pass logs and reports, the
comparison JSONs including the two extra draft pairings, `manifest.json`
with the exact container commands, `resources.jsonl`), and the combined
`gate-1/results.json` named in the [manifest](family-gate-results.json).
The frozen datasets are under `$SHARAR_RESULTS/datasets/family-eval-v1/`
and `family-eval-v2/`.
