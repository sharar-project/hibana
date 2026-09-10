# Sixteen-row draft-tree coverage study

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

This follows the failed [two-path screen](two-path-study.md) and
[target-verifier kernel screen](atlas-verifier-study.md). It tests whether a
fixed tree can recover more useful tokens from each traversal of the target
weights on GB10. Selection and measurement live in `bench/tree_study.py`;
serving execution, proposers, kernels and memory ownership are unchanged.
The study **fails its frozen completion gate**: projected reductions are
**7.71% and 7.68%**, below the required 10%. No tree serving mechanism is added.

## Results and decision

Both windows complete all twelve control/capture pairs: **48 generations,
36,864 output tokens and 8,852 native speculative request-rounds**. All 24
pairs match in output IDs, round count, committed length and final target
KV/SSM/convolution hashes. All twelve cases also match outputs and final state
across the two loads. The audit rebuilds every one of the 4,426 captured trees;
4,390 rounds have complete lookahead and 36 terminal rounds are excluded.
Frozen input files reproduce byte for byte.

Each window's coverage rises from 9,151 to 10,878 potential tokens
(**18.87%**). After output token 512, it rises from 3,022 to 3,457
(**14.39%**). The sustained-coverage and per-cell regression gates pass, but
the completion gate fails in both windows:

| Load window | Native completion sum | Estimated tree completion, 5 ms allowance | Estimated reduction | Optimistic reduction, zero allowance |
|---:|---:|---:|---:|---:|
| 1 | 351.973 s | 324.850 s | 7.71% | 10.35% |
| 2 | 352.178 s | 325.145 s | 7.68% | 10.32% |

The following groups combine two tasks and two loads, four controls per row,
at **one concurrent request**. Native decode TPS is measured per request as
post-prefill output tokens divided by decode time. Tree completion reductions
are projections; no tree-rollout TPS is measured. These new coding tasks are
not a new matched SGLang/Atlas comparison.

| Input tokens | Output tokens/request | Native decode TPS | Extra coverage | Estimated completion reduction, 5 ms | Optimistic reduction, 0 ms |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 512 | 32.24 | 21.64% | 10.67% | 13.71% |
| 8,192 | 1,024 | 39.12 | 16.45% | 7.95% | 11.33% |
| 16,384 | 512 | 33.07 | 20.16% | 8.05% | 10.58% |
| 16,384 | 1,024 | 38.00 | 17.95% | 7.96% | 10.92% |
| 31,616 | 512 | 31.49 | 21.68% | 6.48% | 8.33% |
| 31,616 | 1,024 | 37.88 | 18.80% | 6.64% | 8.97% |

Measured flat-path target costs, taking medians over fourteen samples per
context and shape across the two loads:

| Input tokens | Eight target rows | Sixteen target rows | Difference |
|---:|---:|---:|---:|
| 8,192 | 88.68 ms | 90.54 ms | 1.86 ms |
| 16,384 | 89.79 ms | 91.93 ms | 2.14 ms |
| 31,616 | 91.81 ms | 94.72 ms | 2.91 ms |

The estimate uses each window's own medians, not the pooled table. Beyond this
flat-path proxy, only **0.66 / 0.60 ms of additional overhead per round** would
fit a 10% projected completion reduction in windows 1/2. That small residual
budget does not establish that real tree execution is viable. Cold prefill
also remains unchanged, averaging about 3.06 / 6.61 / 14.55 seconds at the three
context lengths. The apparent 8K/512 subgroup win does not pass the full gate.

A post-hoc diagnostic partitions the 4,390 included rounds: 918 already accept
all seven draft tokens; 1,612 recover the first rejection with the selected
tree; 1,348 have the first rejected target token in the lattice but outside the
tree; and 512 miss that token in the lattice entirely. This explains remaining
coverage opportunities without retuning the frozen selector or relabeling the
failed screen. The two-path study used different tasks, so its smaller observed
coverage gain is not a controlled head-to-head comparison with this tree rule.

The next bounded verifier study is a fixed-operand numerical trace for the
repeated eight/sixteen-row difference described below, alongside the pending
fixed-b/a prefill control. Locate the cause and compare against an arithmetic
reference before choosing a numerical change. Do not integrate the tree or
add an adaptive selector after this failed screen. Actual useful-turn speed,
task quality, B4 behavior and winner-state commit remain unproven for trees.

## Prospective protocol

The protocol and exact token IDs were frozen at **2026-09-06 22:01:07 UTC**,
before loading the model or collecting outputs. Two new standalone coding
tasks ask for a versioned transactional map and a range reservation allocator.
Historical repository source and deterministic synthetic telemetry fill exact
**8,192, 16,384 and 31,616 input tokens**. Each task/context runs at **512 and
1,024 output tokens**, one request at a time. Inputs differ from the previous
two-path tasks. These are fixed-work greedy continuations with EOS ignored;
they do not score whether the generated programs solve the tasks.

Each independently loaded window pairs twelve native controls with twelve
captured continuations. Window 1 runs control then capture in forward cell
order; window 2 reverses both orders. Every run starts with an empty cache and
the same 2,048-token chunked prefill. Output IDs, round count, committed length
and final target KV/SSM/convolution hashes must match within each pair. State
hashing happens after timing. Every release and flush must recover all 34,999
usable KV entries and eleven usable recurrent slots.

The tree has an implicit pending-input anchor, all seven nodes of the native
greedy chain, and eight extra nodes. Starting with this backbone, it repeatedly
adds the available child with the largest full-prefix draft probability.
Conditional probabilities use temperature-one softmax over each row's sixteen
candidate logits. Scores accumulate in log space; ties use the lexicographic
full candidate-rank path. The predecessor rank selects the next conditional
row. Two prefixes ending in the same token still own distinct target histories.
The selector receives no target outputs. These proposal probabilities are a
heuristic, not calibrated target probabilities or a novelty claim.

For each captured native round, coverage is the matching tree-prefix length
plus one target correction or bonus token, up to eight. The complete greedy
backbone must explain the actual native emitted count. Only terminal rounds
without eight-token lookahead are excluded. Full-continuation coverage and
coverage after output token 512 are reported separately. This observation on
the native trajectory does not execute an alternative tree continuation.

## Cost and advancement gates

The cost control reuses `bench.two_path_study.verify_cost`: two flat eight-row
target paths read the same immutable prefix and recurrent slot, with eight
extra private writable KV entries. Each context/window has two warmups and
seven paired samples, alternating physical batch order, with three graph calls
per sample. The measurement includes planning, metadata upload and replay.
Committed-history hashes must remain unchanged. Cross-shape argmax changes are
recorded rather than assumed absent.

This eight-versus-sixteen-row comparison is a **shape-cost proxy**, not measured
tree attention, recurrent branching or winner commit. The frozen estimate is:

```text
coverage_ratio = tree_coverage_tokens / native_coverage_tokens
extra_ms = max(0, median_two_path_wall_ms - median_one_path_wall_ms)
estimated_completion = native_prefill_seconds
    + (native_decode_seconds + native_rounds * (extra_ms + allowance_ms) / 1000)
      / coverage_ratio
```

The main screen reserves **5 ms per native round** for unmeasured tree overhead;
an optimistic zero-allowance estimate is reported separately. Five milliseconds
is a sensitivity budget, not a measured bound. Advancement requires all of:

- At least **10% lower summed estimated completion in each window** with the
  five-millisecond allowance.
- No individual cell with more than **5% estimated regression**.
- At least **10% extra coverage after token 512 in each window**.
- Native/capture output and committed-state equality, complete cleanup and
  the resource gates.

Even a pass only authorizes an isolated verifier prototype. It must measure
actual tree attention/recurrent cost, winner-only state/KV commit and correct
draft feedback, then demonstrate at least 10% useful-turn completion reduction
with task quality, state integrity, protected concurrency-four behavior and
memory checks before serving integration. B4 candidate behavior and semantic
answer quality are not evaluated by this single-request screen.

## Repeated numerical sensitivity

At 8K, the physical eight-to-sixteen-row cost control changes target argmax at
position four (zero-based), changing the first chain's emitted sequence in
both loads. At 16K, position six changes after the first rejection; the first
chain's emitted sequence is preserved. At 31,616 tokens these cost controls
show no first-chain argmax changes. Shared committed history remains unchanged.

This is batch-shape sensitivity, not evidence that either result is more
accurate, and not an observed native control/capture failure. It also means
the native-trajectory coverage ratio cannot establish a sixteen-row rollout's
acceptance or quality. The earlier [prefill trace](agent-context.md) isolated
shape-sensitive BF16 b/a projections, but that does not prove the same cause
here. A fixed-operand verifier trace must locate the first divergence and
control gating projections, attention dispatch and vocabulary row shape
separately before changing target arithmetic.

## Reproduction and evidence

The source baseline is `763de77`. Runtime source matches the validated image
except the previously committed chat admission repair, which this direct
executor benchmark does not call. The image is
`sha256:4a7206b66ead662a1dd2a5de811ef270f4dd7a27926273b2e5b812bbe0bade9b`.
The pinned target is RadixArk Qwen3.8-27B NVFP4 revision
`319f741cce68d7914884900c138a1fbb70a42f30`, with incoai DFlash2 revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`, seven draft tokens and accepted-state
reconstruction. Pools remain 35,000 KV entries and twelve recurrent slots.

Each isolated container has eight CPUs, four OMP threads, a 64 GiB memory
limit, a 64 GiB maximum host-available-memory drop and a 16 GiB host floor.
Only one model uses the GPU at a time. The guard restores the original
production image and memory guard, then checks health, model and the exact
canary after each window. No default promotion occurs in this study.
Both restorations and a final independent health/profile/canary check pass.
Minimum host `MemAvailable` is 81.85 GiB in both windows; maximum measured drops
are 31.27 and 31.21 GiB. The original production image remains
`sha256:b893cdc2e3171bc2b977949305535e1cc62ebb41cee7d607638d628588d50b1b`.

Full protocols, inputs, source hashes, raw lattices, state hashes, target-cost
samples, controls and resource logs live under
`$SHARAR_RESULTS/reviews/2026-09-06/tree-coverage`. Eleven new CPU cases cover
exhaustive small-tree selection, prefix closure, deterministic ties, distinct
converging histories, correction/bonus accounting, sustained observations,
terminal exclusions and invalid lattices. The complete CPU suite passes
**330 tests, with six GPU-only skips**.

Freeze inputs with `python -m bench.tree_study --freeze INPUTS --protocol PROTOCOL`
in the pinned environment. Run each window with
`python -m bench.tree_study --inputs INPUTS --output OUTPUT --window 1` (then 2)
inside the retained resource guard. The raw audit rebuilds every selected tree
and coverage total and recomputes the frozen estimates from uninstrumented
controls. Captured-run timings never enter reported speed estimates.
The [compact manifest](tree-coverage-results.json) indexes the complete sealed
bundle and summary. Raw lattices, outputs and performance samples stay outside
Git. The previous matched SGLang comparison and deployment status are unchanged.
