# Sharar versus SGLang on the same machine, model and harness

This is the historical MTP comparison, before the lifecycle and draft-cache
repairs. Use [the current plain/DFlash2 benchmark](competitive.md) for the
2026-09-05 source. The old prompts, engine versions and memory settings differ;
do not combine rows from the two experiments into a speedup claim.

Measured on 2026-09-04 on one DGX Spark (GB10, driver 580.95.05) with
`RadixArk/Qwen3.8-27B-NVFP4` at the pinned revision, the frozen
`bench/benchmark_text_server.py` matrix (256 output tokens, 5 measured batches
per cell, medians) and `tools/eval_gsm8k.py` (first 100 GSM8K test problems,
greedy, thinking off, concurrency 4, same prompt on every engine).

Engines:

- SGLang plain: the frozen reference container (SGLang at commit
  `6d34a4d3ce`, FlashInfer attention, Triton GDN, CUDA graphs, no
  speculative decoding), port 30000.
- SGLang MTP: the same image relaunched with its built-in speculative
  decoding for this model (`--speculative-algorithm NEXTN`, 3 steps, 4 draft
  tokens, `tools/reference/run-sglang-spec.sh`), port 30002.
- Sharar plain: the runtime image with `SHARAR_SPEC=0`.
- Sharar MTP: the runtime image with `SHARAR_SPEC=3` (chain drafts of 3).

## Throughput and TTFT (aggregate output tok/s; median / p95 TTFT in seconds)

| Input | Conc | SGLang plain | Sharar plain | SGLang MTP | Sharar MTP |
|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 12.7 (0.13 / 0.13) | 11.9 (0.19 / 0.19) | 34.8 (0.16 / 0.16) | 32.7 (0.16 / 0.16) |
| 128 | 4 | 46.4 (0.31 / 0.31) | 43.7 (0.35 / 0.38) | 115.5 (0.35 / 0.35) | 105.1 (0.40 / 0.40) |
| 2,048 | 1 | 12.2 (0.81 / 0.81) | 11.5 (0.85 / 0.85) | 31.0 (0.86 / 0.86) | 22.8 (0.86 / 0.86) |
| 2,048 | 4 | 40.7 (2.34 / 3.07) | 38.4 (2.45 / 3.26) | 65.6 (2.84 / 3.28) | 61.8 (2.54 / 3.39) |
| 8,192 | 1 | 10.8 (3.33 / 3.33) | 10.2 (3.55 / 3.55) | 21.2 (3.48 / 3.48) | 19.5 (3.65 / 3.65) |
| 8,192 | 4 | 18.3 (8.46 / 31.7) | 26.9 (10.3 / 13.8) | 32.5 (8.99 / 20.6) | 38.5 (10.7 / 14.4) |

Runs: `controlled-27b-sparklang-cli-candidate-20260903T005850Z`,
`controlled-27b-sharar-baseline-20260904T125049Z`,
`controlled-27b-sglang-spec-20260904T154941Z`,
`controlled-27b-sharar-spec3g-20260904T153017Z`. Quality canaries 4/4 on
all four. SGLang's speculative run reports 3.5 to 4.0 accepted tokens per
round on these prompts; Sharar's server-side statistics over its matrix show
79.5% draft acceptance.

## Accuracy (GSM8K, first 100 test problems, greedy, thinking off)

| Engine | Correct | Aggregate tok/s at concurrency 4 |
|---|---:|---:|
| SGLang plain | 93 / 100 | 45.5 |
| SGLang MTP | 95 / 100 | 105.6 |
| Sharar MTP | 94 / 100 | 94.6 |

The three scores are close on this 100-problem subset (one problem is one
point); this small sample does not establish accuracy equivalence. The result files
are `${SHARAR_RESULTS}/parity/gsm8k_*.json`.

## Reading

- Plain versus plain: Sharar is 5 to 7% slower in five cells, which is the
  measured FP8 GEMV bandwidth gap, and 47% faster with less than half the p95
  TTFT at 8K x 4 because of its scheduler.
- Speculative versus speculative: SGLang is faster in five cells (6 to 9%,
  and 26% at 2,048 x 1) and Sharar is faster at 8K x 4 (+19%, p95 TTFT
  14.4 s against 20.6 s). Where both targets produce the same kind of
  continuation the acceptance is the same (see below); the remaining gap at
  batch 1 is about 8 ms per round of engine cost.
- The observed GSM8K scores are close; broader quality equivalence is unproven.

## Where the 2,048 x 1 gap comes from

The harness prompt is a unique header followed by one repeated pangram. Each
engine's `/generate` reports per-request acceptance (Sharar added
`spec_accept_length` and `spec_accept_rate` to `meta_info` with the same
meaning as SGLang's). Sent one at a time after a cache flush, the five
2,048-token repetition prompts give:

| Prompt | SGLang MTP tok/round (accept) | SGLang continuation | Sharar MTP tok/round (accept) | Sharar continuation |
|---|---:|---|---:|---|
| rep 0 | 3.88 (95%) | ` processing a deterministic text benchmark.` then the pangram repeated | 2.59 (53%) | ` lazy` then a reply |
| rep 1 | 3.94 (98%) | ` lazy` then the pangram repeated | 2.91 (64%) | a reply |
| rep 2 | 3.94 (98%) | ` lazy` then the pangram repeated | 3.61 (87%) | `<think>` then a reply |
| rep 3 | 3.94 (99%) | ` lazy dog` then the pangram repeated | 3.46 (83%) | `<think>` then a reply |
| rep 4 | 3.05 (69%) | ` lazy dog while a local inference` then a reply | 2.61 (53%) | ` lazy` then a reply |

Both drafts are the same checkpoint MTP head; the drafts accept at 98% when
the target keeps repeating the pangram and at 50 to 85% when it writes free
text. The engines differ in what their *target* does on these prompts, and
that difference is not a Sharar defect:

- On rep 0 the prompt ends with `jumps over the`. Sharar's plain forward puts
  ` lazy` at log-probability -0.02 (a 4-nat margin over `\n\n`). SGLang, both
  plain and MTP, answers ` processing a deterministic text benchmark.` at
  -0.22 in one call and `\n\n` at -0.22 in another; ` lazy` is not in its
  top 3. That is far outside the noise floor established in `docs/parity.md`.
- The same prompt with its first token dropped (2,047 tokens) or duplicated
  (2,049 tokens) makes SGLang predict ` lazy` at -0.00. Across all five keys,
  SGLang's next token at exactly 2,048 tokens sits at -0.01 to -0.32 and
  flips between calls, while at 2,047, 2,049, 3,000, 4,096 and 6,144 tokens
  it is at -0.00 to -0.04 and stable. 2,048 is the reference's
  `--chunked-prefill-size`, so its prefill of a prompt of exactly that
  length takes a different path whose output is degraded and non-deterministic.
- The repeated-pangram continuation that SGLang then produces is what makes
  its draft accept at 98% in that cell; a target that writes free text gives
  both engines the same 50 to 85%.

So the 26% at 2,048 x 1 is an artifact of the harness prompt length on
SGLang; the like-for-like cells are 128 x 1 and 128 x 4, where both engines
accept 98 to 100% and SGLang is 6 to 9% faster per round.

## Per-round cost at batch 1 (Sharar, dev container, `sharar bench-decode --spec 3`)

| Bucket | ms per round (morning) | ms per round (after the sync audit) | Notes |
|---|---:|---:|---|
| Draft (3 chain steps, CUDA graphs) | 25 | 21 (3 x 6.5 GPU) | the MTP layer plus NVFP4 lm_head read 1.42 GB per step, floor 6.3 ms at 225 GB/s |
| Verify forward (4 rows, CUDA graph) | 90 | 88 (86 GPU) | plain decode step is 80 ms in its graph; the extra is attention over 4 queries and the GDN verify kernel |
| Accept bookkeeping | 4.5 | 2.8 | one strided state copy per kind, one host download |
| Host stalls between graphs | ~7 | ~1 | eight synchronous uploads/downloads per round removed (`docs/phase7.md`) |
| Round wall time at batch 1 | 119 | 109 | SGLang MTP: 111 |

The remaining round is GPU time. SGLang MTP's 34.8 tok/s at 3.88 tokens per
round is 111 ms per round, so at equal acceptance the two engines are now
level at batch 1.
