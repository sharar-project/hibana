# Current performance evidence

Updated 2026-09-17. Numbers below describe controlled GB10 source benchmarks,
not a newly deployed production image. Raw evidence, immutable inputs and
source identities are linked through each report's compact manifest.

## Current calibrated W3 profile, 2026-09-17

With the selected FP8 drafter and exact acceptance, unprofiled decode rounds
take 70.68, 82.19 and 107.79 ms at batch sizes one, two and four on the frozen
10K-context profile. W3 GEMM accounts for 58–71% of round time; GPU idle gaps
account for 2.8–5.5%. These are round costs, not tokens/s across workloads.

Staging mapped calibrated tensors in CPU storage before CUDA loading reduces
warm-cache target construction from 63.23 to 11.99 seconds, with all 304
matrices byte-identical. This excludes drafter loading and graph capture and
does not improve steady decode. Broad kernel substitutions and extra decode
between prefill forwards did not qualify for serving. See the
[profile, controls and negative results](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-runtime-profile.md).

The subsequent narrow 64-row QKV specialization improves cached 10K
batch-four family decode by **0.66% [0.61%, 0.71%]** across two paired process
blocks covering all 101 families, with identical outputs and round work.
Batch-one/two controls show no material change. HTTP comparisons do not
establish a general serving-throughput gain; their cache pressure and run
variation are recorded in the [qualification study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-qkv-tiling.md).
The specialization and source-keyed W3 build cache are local source changes
awaiting deployment.

The [MLP hardware-counter follow-up](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-mlp-profile.md)
profiles the large gate/up and down matrices at 16/32/64 rows. Cold-cache
captures show memory-load waits and 255-register kernels with local spills
at 64 rows. The profiler does not expose direct DRAM counters on this
device. Two cache-policy substitutions were 43–59% slower in unprofiled
graph-replay screens and were rejected; no additional MLP serving gain is
claimed.

## DFlash2 configuration

Use **FP8 DFlash2, k=15, accepted-state reconstruction** as the optimized
measured baseline. Fifteen proposals give sixteen target verification rows per
request. Since 2026-09-17 the launcher defaults to FP8 draft precision (with
the 3-bit layer and draft copies); see [operations](operations.md).

| Proposals verified | Rows per request | BS4 pooled family decode TPS |
|---|---:|---:|
| 15 | 16 | 132.70 |
| 7 | 8 | 129.25 |
| 3 | 4 | 104.80 |

The [width study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/gb10-bs4-engine.md) held draft generation at
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
four. See the [draft training report](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/gb10-draft-training.md)
and [draft adaptation](draft-adaptation.md).

## Integrated changes

| Change and workload | Measured result | Qualification |
|---|---|---|
| Exact BS1 QKV/MLP bundle | +1.43% family decode | 18 families; exact output preservation |
| Resumed BS1 checkpoint retention | 11.34 → 36.73 completion TPS (3.24×) | Six families, two independent loads per variant; candidate matches healthy immediate-hit oracles |
| Four simulated conversation sessions, 96 slots | 28.78 → 78.48 aggregate TPS (2.73×) | Two blocks in one fresh process per variant; three fixed 128-output turns per lane |
| Retention, cold controls | Approximately unchanged | Native BS1 −0.14% to +0.06%; BS4 −0.10% |

See the [BS1 report](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/gb10-bs1-speedups.md) and
[retention report](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/gb10-agent-retention.md) for complete results.
Warm recovery comes with sufficient feature capacity: an enlarged independent
LRU also recovered the tested throughput. Checkpoint ownership supplies
predictable lifetime and reclamation. Caps are 460 MiB at 24 state slots and
1,900 MiB at 96; automatic KV capacity may decrease. A small-pool trace had one
full target-prefix eviction and worse latency tails. These are not additive
cold-kernel gains or measurements of arbitrary agent productivity.

## Against SGLang, 2026-09-16

On the standard 10K/1K batch-one cell with identical prompt IDs, mean
end-to-end output TPS over code, design and reasoning: SGLang DFlash2 30.4,
Sharar exact 36.5 (1.19x), the relaxed default 44.2 (1.44x), 3-bit decode
weights 47.4 (1.59x), both together 52.3 (1.75x); see the
[comparison](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/sglang-comparison-2026-09-16.md) and the
controlled numbers behind it in the gate studies. The CUDA 3-bit kernel with
the head and draft (2026-09-17, decode x1.488 on the gate against the first
build's x1.174) projects to about 1.9x end-to-end exact and 2.1x with the
relaxed default; the [inventory](optimizations.md#combined-effect-on-the-standard-cell)
shows the arithmetic, a rerun replaces it. Measured on the code workload
after the 2026-09-17 restart (layers and draft copies, FP8 draft): 43.8
end-to-end and 53.0 decode with exact acceptance pinned, 53.1 and 67.2 with
the served relaxed default, against SGLang's 27.6 and 31.1 (1.93x end-to-end
as served).

## Rejected candidates and remaining limits

The short-resume scheduler reduced mixed throughput 7.27% and lengthened
streaming gaps. Smaller BS4 widths, mixed forwarding, numerical FP8 replacement
and selector/compression screens failed their stated gates. Their evidence is
in the [decision archive](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/README.md); their reported microbenchmark
savings cannot be added into a throughput forecast.

The 1.5× cold BS1/BS4 goal with a fixed target remains open.

## Limits

The target's arithmetic depends on execution shape even with unchanged
weights: prefill features of a fixed prefix change with the total prefill
length and chunking (up to 30% relative RMS in deep layers, about one prefix
argmax in ten between a 26- and a 27-token prefill), and greedy continuations
can therefore change with batch shape, verification width or the drafter,
because those move the round boundaries a token is computed in. Each
configuration is deterministic run to run. The cause is three kernels whose
arithmetic depends on the row count: cuBLAS's BF16 GEMM for the GDN b/a
gates, cuBLASLt's FP8 GEMM for the 5120x6144 output projections at 27 to 32
rows, and the gated norm's rows-per-block choice, with FlashInfer's prefill
schedule a rare fourth. M-invariant replacements for the first three make a
prefix bit-exact across prefill lengths for under 1% prefill cost ([target
numerics study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/target-numerics.md)); since 2026-09-15 they
ship in main behind `SHARAR_STABLE_NUMERICS=1`, off by default, as the
research baseline for row-count experiments: the one-time shift in greedy
outputs changed served acceptance on a small prompt set, and chunked and
cached-prefix prefills still take different attention paths. HTTP task trajectories and batch
membership can differ; synthetic fixed-output traces do not establish
generated-code quality or completed tool tasks
([quality limitations](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/greedy-quality.md)).

For new measurements, use [evaluation](evaluation.md) (served builds) or [profiling](../profiling/README.md) (in-process). Keep cold
requests, warm resumes, decode TPS and end-to-end TPS distinct.
