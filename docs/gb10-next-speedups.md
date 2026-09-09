# GB10 speed priorities after the acceptance investigation

The latest [target arithmetic repair](gb10-target-fix.md) resolves the first-token
gap and matches every accepted layer output, top-10 logit and committed state
in a 65-block common-history replay. Full independent generation remains a
separate gate. The dated cost studies below retain their original source and
continuations; consult the new report for current AIPerf results.

The earlier [target-path investigation](gb10-target-path.md) identifies
FP4 quarter-tie and SiLU product-rounding differences. It retains the exact
tie repair; the SiLU candidate improves code but regresses the other prompts,
and the QKV/Z fusion screen saves only 0.58%. The fresh trace remains dominated
by matrix work. Compare corresponding target intermediates to complete parity
before treating a changed continuation's acceptance gain as an engine speedup.

2026-09-08. The subsequent [exact-arithmetic fusions](gb10-engine-speedups.md)
reduce code B1 10k/1k latency from 35.144 to 33.803 s in paired AIPerf loads,
with identical output IDs and 264 rounds. Design and reasoning improve too.
The earlier SGLang AIPerf code result is 30.904 s on a different continuation
and 205 rounds. A wide matched-output win remains unproven.

## What limits the engine

The [10k profile](gb10-10k1k-audit.md) measures approximately 120 ms per code
round, including about 92 ms of target verification and 19.5 ms of drafting.
Separate trace measurements find approximately 4 ms outside GPU activity.
Phase spans and kernel sums come from diagnostic rounds and should not be
treated as an exact additive accounting of the HTTP run.

The target's projection operands occupy 17.555 GB, including 7.214 GB of FP8
matrices. At NVIDIA's specified 273 GB/s, one read of those target operands
alone costs 64.3 ms under ideal conditions. That is a conditional traffic
floor, not a measurement of physical LPDDR saturation. The device is already
executing native Tensor Core kernels; its advertised sparse FP4 peak does not
describe this dense, small-token-block workload.
[NVIDIA hardware specification](https://www.nvidia.com/en-us/products/workstations/dgx-spark/).

The [earlier NVFP4 counter study](nvfp4-cost.md) finds approximately one packed
weight footprint in system-memory fills for sampled GEMMs. Narrower NVFP4
tiles regressed B1. The [target FP8 kernel study](atlas-verifier-study.md)
measured only 1.97–2.90% whole-round savings at B1. There is no demonstrated
large hidden reread or launch-overhead problem to remove.

Sharar's historical round cost is already lower than the reference's rough
per-round estimate, but its code continuation needs more rounds. The
[same-history investigation](gb10-acceptance-parity.md) finds a small aggregate
proposal deficit on SGLang's recorded histories. It does not establish that
either draft is close to the best achievable predictions on Sharar's histories.

## New offline check: room for better selection

The existing `acceptance-screen/report.json` retains full candidate lists and
the actual Sharar continuation. For every one of its first 96 histories per
task, walk the continuation until the next token is absent from that position's
top-16 candidate list. Add one correction token. This gives the best possible
prefix from that fixed candidate support if every choice were known in advance.
It uses future target tokens and therefore is an oracle diagnostic, never an
implementable serving policy or an achieved speedup.

| Task, 96 histories each | Current mean prefix + correction | Perfect-selection ceiling |
|---|---:|---:|
| Code | 3.479 | 6.583 |
| Design | 4.031 | 6.813 |
| Reasoning | 7.198 | 10.406 |

These are early-history samples, not averages over complete alternative
generations. The captured native paths reproduce every recorded emitted count.
There is substantial available support; recovering it with learned predictions
is unproven. This motivates a training feasibility experiment, not a claim
that a selector porting bug caused the reference gap.

A cheap alternative was also screened without loading a model: replace the
locally greedy choice with a backward dynamic program maximizing expected
matching-prefix length under temperature-one conditional draft probabilities:

```text
V_t(previous) = max_j p_t(j | previous) * (1 + V_(t+1)(j))
```

The policy reads only draft scores; target continuations are used solely for
evaluation. A small exhaustive enumeration checks the recurrence. The protocol
fixes temperature one, uses every captured cell in both historical tree-study
windows, and requires 10% aggregate coverage improvement per load before
advancing. This is an eight-row historical configuration, separate from the
sixteen-row oracle sample above.

Across 4,390 retained round observations, including repeats across two loads,
matched prefix-plus-correction totals change from **18,302 to 18,428 (+0.688%)**.
Each load changes 9,151 to 9,214. The worst cell regresses 2.55%. The candidate
fails the advancement screen; no GPU kernel, serving option or rollout claim
is added. It shows that lookahead on the existing scores recovers little of
the oracle ceiling on these data.

## Ranked work

1. **Resolve target-execution parity before attributing the code round gap
   to the draft.** The exact fusions now save 3.82% on the same code output.
   The target FP8 tile screen changed continuations despite passing four
   canaries, so it did not advance to a combined FP8 profile under the
   exact-output gate. First-token probes still locate the cross-engine code
   divergence before drafting. Trace the first divergent target computation
   under identical prefill partitions, including packed weights, norm round
   points and GEMM backends. Preserve the earlier failed norm/weight variants;
   a faster changed continuation does not qualify as an exact-work speedup.

2. **Test target-specific selector training for the larger algorithmic gain.**
   Collect complete 10k/1k native traces, including late-generation histories,
   with the current target weights, numerical path and draft precision. Split
   training and held-out tasks before tuning. Extend the support-ceiling check
   to these traces. Start by freezing target and draft backbone and training
   the small conditional selector on the target's actual continuations, with
   an objective emphasizing accepted prefixes. Measure prefix length per
   complete round cost. Only expand to backbone distillation if held-out
   support or selector learning shows that it is necessary. The existing
   oracle ceiling does not guarantee a learnable or generalizing gain.

3. **Calibrate a more compact target as a separate model optimization.**
   The FP8 projections account for roughly 36 ms of B1 kernel time. The earlier
   FP4 conversions ran 1.4–1.7x faster locally, but five of ten projections
   failed their numerical screen. Use representative activation calibration
   and layer sensitivity, preserving FP8 in sensitive projections. NVIDIA's
   Model Optimizer supports calibrated NVFP4 and mixed-precision selection;
   compatibility and quality must be established for this particular checkpoint.
   Even converting all the FP8 operands would reduce the logical target
   projection inventory only from about 17.56 to 14.40 GB once FP4 block scales
   are counted. It is not a stand-alone 2x mechanism. Compare SGLang using the
   same resulting checkpoint to distinguish model gains from engine gains.
   [NVIDIA quantization workflow](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/README.md).

New draft weights can also benefit SGLang. Include a same-draft reference
comparison where supported before claiming an engine advantage from training.

## Concrete speed budget

For a single request, approximate completion time as:

```text
completion = prefill + 999 / mean_emitted_per_round * complete_round_time
```

Using a 3.75 s prefill and the earlier 30.904 s SGLang AIPerf reference:

| Goal | Required round cost at 3.8 tokens/round | Required tokens/round at 100 ms |
|---|---:|---:|
| Match reference latency | 103.3 ms | 3.68 |
| 25% lower reference latency | 73.9 ms | 5.14 |
| Half reference latency | 44.5 ms | 8.54 |

These are conditional budgets, not predicted results. A combined target of
100 ms per round and 5.3 emitted tokens per round would give approximately
22.6 s completion. That illustrates why both cheaper rounds and better
predictions matter for a wide win. At unchanged acceptance, the half-latency
round budget is below the ideal target-weight-read floor alone.

## Evidence

The offline selection check above used CPU-only analysis of retained captures.
The later GPU implementation and measurements are linked separately. Raw per-history results and
the reproducible script are under
`${SHARAR_RESULTS}/reviews/2026-09-08-speed-plan/`:

- `screen.py`: SHA-256 `889e6f326ef855703f83ff797d48ddd8b6913e1d5a8df298ae9fcdacf2f3545b`
- `protocol.json`: SHA-256 `a9399cda2ae6b0a8b91a499e5943e7aa509ef6d0b03bcf06e831ca63ca7ae4dc`
- `results.json`: SHA-256 `4c83f0da51126196be8e08211a4533f2b4332376c0c88e034230460762c0c108`

The results retain source hashes, exclusions and per-history counts. Original
GPU measurements remain bound to the dates, shapes and source versions in their
respective reports; the linked fusion report owns the subsequent GPU results.
