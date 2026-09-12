# Current performance evidence

Updated 2026-09-11. Numbers below describe controlled GB10 source benchmarks,
not a newly deployed production image. Raw evidence, immutable inputs and
source identities are linked through each report's compact manifest.

## DFlash2 configuration

Use **FP8 DFlash2, k=15, accepted-state reconstruction** as the optimized
measured baseline. Fifteen proposals give sixteen target verification rows per
request. The launcher still defaults to BF16 draft precision; select FP8
explicitly as described in [operations](operations.md).

| Proposals verified | Rows per request | BS4 pooled family decode TPS |
|---|---:|---:|
| 15 | 16 | 132.70 |
| 7 | 8 | 129.25 |
| 3 | 4 | 104.80 |

The [width study](archive/studies/gb10-bs4-engine.md) held draft generation at
15 and truncated target verification. It does not measure a fully shortened
drafter or prove that k=15 wins every prompt. Smaller widths changed outputs
and produced mixed BS1 results; no adaptive width policy was qualified.

## Trained drafter

Training the DFlash2 drafter on recorded traffic changes speed only. On a
repository-coding proxy workload, 300 steps on 359 traced requests raised
held-out tokens per round 5.367 to 5.739 and served throughput on held-out
prompts 46.3 to 52.3 tokens/s (+12.9%, batch one, FP8 draft, three repeats);
a 1,000-request serve-then-train loop raised acceptance while serving from
27.5% to 31.5% of proposals and throughput 129 to 145 tokens/s at concurrency
four. See the [draft training report](archive/studies/gb10-draft-training.md)
and [draft adaptation](draft-adaptation.md).

## Integrated changes

| Change and workload | Measured result | Qualification |
|---|---|---|
| Exact BS1 QKV/MLP bundle | +1.43% family decode | 18 families; exact output preservation |
| Resumed BS1 checkpoint retention | 11.34 → 36.73 completion TPS (3.24×) | Six families, two independent loads per variant; candidate matches healthy immediate-hit oracles |
| Four simulated conversation sessions, 96 slots | 28.78 → 78.48 aggregate TPS (2.73×) | Two blocks in one fresh process per variant; three fixed 128-output turns per lane |
| Retention, cold controls | Approximately unchanged | Native BS1 −0.14% to +0.06%; BS4 −0.10% |

See the [BS1 report](archive/studies/gb10-bs1-speedups.md) and
[retention report](archive/studies/gb10-agent-retention.md) for complete results.
Warm recovery comes with sufficient feature capacity: an enlarged independent
LRU also recovered the tested throughput. Checkpoint ownership supplies
predictable lifetime and reclamation. Caps are 460 MiB at 24 state slots and
1,900 MiB at 96; automatic KV capacity may decrease. A small-pool trace had one
full target-prefix eviction and worse latency tails. These are not additive
cold-kernel gains or measurements of arbitrary agent productivity.

## Rejected candidates and remaining limits

The short-resume scheduler reduced mixed throughput 7.27% and lengthened
streaming gaps. Smaller BS4 widths, mixed forwarding, numerical FP8 replacement
and selector/compression screens failed their stated gates. Their evidence is
in the [decision archive](archive/README.md); their reported microbenchmark
savings cannot be added into a throughput forecast.

The 1.5× cold BS1/BS4 goal with a fixed target remains open.

## Limits

The target's arithmetic depends on execution shape even with unchanged
weights: prefill features of a fixed prefix change with the total prefill
length and chunking (up to 30% relative RMS in deep layers, about one prefix
argmax in ten between a 26- and a 27-token prefill), and greedy continuations
can therefore change with batch shape, verification width or the drafter,
because those move the round boundaries a token is computed in. Each
configuration is deterministic run to run. Evidence: the
[draft training report](archive/studies/gb10-draft-training.md); the finding
is on the roadmap as a target-numerics task. HTTP task trajectories and batch
membership can differ; synthetic fixed-output traces do not establish
generated-code quality or completed tool tasks
([quality limitations](archive/studies/greedy-quality.md)).

For new measurements, use [the benchmark guide](../bench/README.md). Keep cold
requests, warm resumes, decode TPS and end-to-end TPS distinct.
