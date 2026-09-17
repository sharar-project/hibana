# Sharar versus SGLang on the 10K/1K batch-one cell, 2026-09-16

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

The standard benchmark (`tools/benchmark.sh --workload all`: AIPerf 0.12.0,
10,000 frozen input IDs per workload, 1,000 output tokens, greedy,
concurrency one, cold prefix cache, one warmup and three timed requests) on
identical prompt IDs for six configurations, each in its own exclusive GPU
window. SGLang is the pinned DFlash2 reference (`lmsysorg/sglang:dev-qwen38-27b-dflash2`
with the server command recorded on 2026-09-08: FP8 KV, DFLASH with 16 draft
tokens, torch compile); the non-speculative SGLang container is shown only
as context. Sharar is main `afab63c` plus the working tree (relaxed
acceptance and the 3-bit decode path), FP8 pinned DFlash2 at k=15.

End-to-end tokens per second include prefill; decode is `999 / (latency - TTFT)`.

| Configuration | Code e2e / decode | Design e2e / decode | Reasoning e2e / decode | Mean e2e | vs SGLang DFlash2 |
|---|---:|---:|---:|---:|---:|
| SGLang DFlash2 (reference) | 27.6 / 31.1 | 28.3 / 32.1 | 35.3 / 41.4 | 30.4 | 1.00x |
| SGLang without speculation | 11.5 / 12.1 | 11.5 / 12.1 | 11.5 / 12.1 | 11.5 | 0.38x |
| Sharar, exact acceptance | 32.0 / 36.5 | 30.7 / 34.9 | 46.8 / 57.5 | 36.5 | **1.19x** |
| Sharar, relaxed default (top-2 within 1.0 logit) | 38.4 / 45.3 | 37.8 / 44.4 | 56.5 / 72.7 | 44.2 | **1.44x** |
| Sharar, 3-bit decode weights | 48.3 / 59.6 | 49.8 / 62.0 | 44.2 / 53.5 | 47.4 | **1.59x** |
| Sharar, 3-bit + relaxed | 50.1 / 62.4 | 57.4 / 74.2 | 49.3 / 61.3 | 52.3 | **1.75x** |

Time to first token is 3.95 to 3.97 s for every Sharar arm and 4.19 to
4.21 s for SGLang; the ratios above are means over the three workloads of
per-workload ratios (decode: 1.22x, 1.53x, 1.71x, 1.93x).

Read the per-workload columns with the usual caution: these are three
prompts, and any arm that changes the target's arithmetic or the acceptance
rule changes each continuation, so single workloads swing (the 3-bit arm
runs code 75% faster and reasoning 25% faster than SGLang). The controlled
numbers are the 101-family gate's: relaxed acceptance about +10% tokens per
round, 3-bit decode x1.174 [1.137, 1.214] on decode throughput with
acceptance unchanged ([relaxed acceptance](relaxed-acceptance.md),
[3-bit decode](w3-decode.md)); task accuracy is unchanged for both
([accuracy benchmarks](accuracy-benchmarks.md)). This table shows those
gains landing on the standard cell against the reference engine.

## Evidence

`$SHARAR_RESULTS/reviews/2026-09-16-compare/`: one benchmark result
directory per arm (frozen inputs, per-request AIPerf artifacts, summaries),
`sglang-dflash2-command.json` and server log, the window scripts, and the
combined `results.json` named in the [manifest](sglang-comparison-2026-09-16-results.json).
