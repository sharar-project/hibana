# GB10 TPS validation and push recommendation

2026-09-09. The existing optimized source was measured again without runtime edits.
Recommendation: push a review branch/PR with FP8 drafting kept opt-in. These results
do not qualify a production default change or an identical-output win over SGLang.

## Fresh AIPerf measurements

AIPerf 0.12.0; one GB10; concurrency one; exactly 10,000 frozen input IDs and
1,000 emitted output IDs; greedy temperature zero and seed 42. Each task has
one excluded full-generation warmup and five timed requests per engine.
Prefix cache is flushed before every request. Loading, compilation, warmups,
cache flushes and AIPerf process startup are excluded from request timing.

Sharar uses the existing explicit FP8 draft option. SGLang uses its pinned
BF16 DFlash2 draft profile. Both use the same target checkpoint, frozen input IDs,
FP8 target KV, FP32 recurrent state, 16-row verification, maximum batch four,
60,000 target KV pages, 24 state slots and 2048-token prefill chunks.
Both run through the same loopback-to-Unix relay in separate offline GPU windows.

End-to-end output TPS is `1000 / mean request latency`, including prefill.
Decode TPS is `999 / mean(latency - TTFT)`, matching AIPerf’s per-user decode metric.
All rates below are output tokens per second; input tokens are not added.

| Task | Sharar end-to-end TPS | SGLang end-to-end TPS | TPS gain | Sharar decode TPS | SGLang decode TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 37.65 | 24.77 | +52.01% | 44.20 | 27.62 |
| Design | 30.14 | 29.18 | +3.30% | 34.20 | 33.23 |
| Reasoning | 46.09 | 39.34 | +17.15% | 56.37 | 47.09 |

Across the equal mix of these three tasks, pooled end-to-end output TPS is 36.84
for Sharar versus 29.98 for this SGLang load (+22.88%).
This divides total output tokens by summed request durations; gaps between
separate client invocations are excluded. It is not a saturation measurement.

| Task | Sharar latency, s | SGLang latency, s | Sharar TTFT, s | SGLang TTFT, s | Sharar TPS min–max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 26.5598 | 40.3746 | 3.9574 | 4.2082 | 37.57–37.75 |
| Design | 33.1796 | 34.2740 | 3.9687 | 4.2112 | 30.13–30.15 |
| Reasoning | 21.6969 | 25.4187 | 3.9737 | 4.2045 | 46.03–46.13 |

## Reference reload variation

The preceding investigation measured two other fresh loads of the same pinned
SGLang profile. Their continuations differed across processes. They remain
separate here; each preceding load has two timed samples per task.
The conservative comparison uses the fastest task mean among those two loads
and this new five-sample load, without pooling different continuations.

| Task | Prior C TPS | Prior D TPS | New SGLang TPS | Sharar gain vs fastest SG mean |
| --- | ---: | ---: | ---: | ---: |
| Code | 28.52 | 31.19 | 24.77 | +20.71% |
| Design | 28.79 | 27.18 | 29.18 | +3.30% |
| Reasoning | 40.62 | 38.18 | 39.34 | +13.46% |

## Correctness and repository readiness

* All 36 fresh requests were independently re-audited from raw AIPerf streams:
  exact payload and prompt IDs, 1,000 raw output IDs, consistent usage, zero
  cached tokens, length finish, HTTP success and complete stream termination.
* Each engine produces identical IDs and text across all six requests per task
  within its current load. Sharar’s correspondence to the preceding FP8 load
  is recorded per task below.
* Full cross-engine greedy output/text equality remains unresolved.
  The earlier 65-block / 307-accepted-row target replay is a common-history
  numerical diagnostic, not proof of full free-generation parity.
* All four current Sharar quality canaries pass. That is not a broad quality evaluation.
* Fresh CPU checks: 373 passed, 8 skipped.
* The earlier 128 targeted GPU tests apply to the same source: all 216 files
  in that validated source manifest match. No GPU regression rerun was needed
  for this documentation-only follow-up.
* The repository evidence check initially failed on three earlier JSON files.
  All six recent measurement summaries now follow the manifest schema/path
  format without dropping measurements;
  original files and hashes are retained in the evidence directory. The check
  and `git diff --check` now pass.

| Task | Sharar matches previous FP8 load | First cross-engine different output ID (zero-based) |
| --- | --- | ---: |
| Code | True | 103 |
| Design | True | 41 |
| Reasoning | True | 69 |

The code is worth preserving and reviewing in a branch/PR with its tests and
evidence. Keep BF16 as the default and FP8 as an explicit option. Before a
production promotion or a strict parity claim, resolve the remaining greedy
output gap and run broader quality coverage on the current profile.
The existing dirty tree contains multiple iterations and untracked kernel/test
dependencies; a push must include the intended implementation and dependencies
as a reviewed set. No commit, push, image rebuild or deployment occurred here.

## Reproduction and evidence

The complete new evidence is under
`$SHARAR_RESULTS/reviews/2026-09-09-tps/`. It includes frozen inputs, pinned
image commands, all AIPerf samples and raw streams, source hashes, resource
logs, CPU checks and restoration checks. `summarize.py` re-audits the responses
and verifies AIPerf’s throughput arithmetic before writing `results.json`.
Use fresh output labels with the recorded coordinator:

```bash
python3 coordinator_e.py sharar sharar-fp8-new --repeats 5
python3 coordinator_e.py sglang sglang-new --repeats 5
```

These commands use the recorded exclusive offline window wrapper, which
temporarily stops and restores the existing service and memory guard.
The original production container/image is restored and healthy.

Target revision: `319f741cce68d7914884900c138a1fbb70a42f30`.
DFlash2 revision: `dedf8df68adfb1afeaf7b7480c0a0243108177b4`.
SGLang image: `sha256:60166a99b6610e2b4b3febbeb6411e4cbc9338e996ab6c80d3339f86a7b5b9c3`.
Sharar base image: `sha256:0d849f3f7ebd1309d1ab0828f00d4eaffe410126dd7db8ea31d226095655130a`,
with the optimized repository source mounted read-only.

[Machine-readable results manifest](gb10-tps-results.json).
