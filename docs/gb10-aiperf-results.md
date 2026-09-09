# GB10: AIPerf 10,000 input / 1,000 output, greedy

2026-09-08. Exact token budgets pass; cross-engine output equality fails. These are normal generations from the current Sharar BF16-draft source and the pinned SGLang image. The existing production image was not rebuilt or deployed.

## Measured serving latency

AIPerf 0.12.0, concurrency one. Code has four measured requests per engine across two independent model loads; design and reasoning have two per engine on one load. Each workload/load also has an excluded full-generation warmup. These are small-sample means, not tail-latency estimates.

| Prompt | Sharar mean (s) | SGLang mean (s) | Sharar time relative to SGLang | Sharar / SGLang rounds |
|---|---:|---:|---:|---:|
| Code | 34.969 | 30.904 | +13.2% | 264 / 205 |
| Design | 34.273 | 39.203 | -12.6% | 258 / 269 |
| Reasoning | 23.891 | 25.304 | -5.6% | 170 / 161 |

Positive relative time means Sharar takes longer. Raw samples, standard deviations and ranges are retained in `gb10-aiperf-results.json`. All reported times come from AIPerf request metrics, excluding model loading, graph capture, client startup, flushes and warmups.

## What makes SGLang faster on code

| Code, mean | Sharar | SGLang |
|---|---:|---:|
| Time to first token | 3.733 s | 4.193 s |
| Decode duration | 31.236 s | 26.712 s |
| Output throughput, including prefill | 28.597 tokens/s | 32.358 tokens/s |
| Output throughput after first token | 31.982 tokens/s | 37.400 tokens/s |
| Committed decode tokens per round | 3.784 | 4.873 |
| Effective decode ms per reported round | 118.318 | 130.300 |

SGLang needs 205 reported verification rounds for code; Sharar needs 264. Each engine emits 999 tokens after its initial prefill token. The extra 59 target traversals outweigh Sharar's lower effective round cost. Effective round cost is `(AIPerf request latency − TTFT) / reported rounds`; it includes draft, verification, state/feedback, scheduling and streaming. It is not an isolated target-kernel measurement.

The content differs. SGLang begins “I'll start by exploring the repository…” and emits tool-call text; Sharar begins a written ownership/rollback design. Generated tool calls are benchmark output only and are not executed. The design and reasoning cases show why the earlier code-only result cannot establish a general SGLang advantage.

The new SGLang code output reproduces all 1,000 IDs in the earlier captured reference. The [same-history investigation](gb10-acceptance-parity.md) therefore remains directly relevant to this continuation: on 128 identical reference histories, Sharar scored 609 accepted-prefix-plus-correction tokens versus SGLang's 617, a 1.3% deficit. This one-sided, teacher-forced diagnostic is not a timed generation and does not prove universal parity. It does show that the large free-generation round gap is not itself evidence of an equally large draft-porting loss.

The pinned SGLang implementation also has these concrete optimizations. Their presence does not establish their individual speedup; no isolated ablation of them is claimed here.

| Enabled SGLang path | Current Sharar path | Source evidence |
|---|---|---|
| Fused draft QKV projection | Separate Q and combined KV projections | SGLang `srt/models/dflash.py`; Sharar `model/parallel_draft.py` |
| Fused context-KV materialization across five draft layers | Per-layer projection, K norm, RoPE and cache writes | SGLang `dflash_worker_v2.py::_init_fused_kv_helper`; startup log confirms enabled; Sharar `ParallelDraftModel.context_kv` |
| FP8 E4M3 draft KV | BF16 draft KV | Prior captured draft caches; target KV is FP8 in both |
| Compiled/fused draft operations and captured draft/target graphs; two continuous decode steps | Custom fused convolution and norm kernels, captured draft/target graphs, device lattice traversal | Pinned server command, SGLang `dflash.py` / `dflash_worker_v2.py`, Sharar `proposers/parallel_graphs.py` / `engine/spec_graphs.py` |

Both engines already use the checkpoint's NVFP4/FP8 target matrices on GB10. The [existing kernel profile](gb10-10k1k-audit.md) identifies matrix traffic as the main round cost. This benchmark supplies no evidence of a missing GB10 feature that alone explains the code result.

## Exact token and text checks

| Prompt | Identical input IDs | Output IDs per response | Common generated prefix | First differing output IDs (Sharar / SGLang) |
|---|---:|---:|---:|---:|
| Code | 10,000 / 10,000 | 1,000 / 1,000 | 0 tokens | 2 / 40 |
| Design | 10,000 / 10,000 | 1,000 / 1,000 | 32 tokens | 3447 / 28151 |
| Reasoning | 10,000 / 10,000 | 1,000 / 1,000 | 0 tokens | 13962 / 27775 |

All 24 main AIPerf requests (16 measured, eight warmups) passed the exact input/output count, finish and cache checks. **All three cross-engine token-equality and text-equality checks fail.** The first divergence is output token 1 for code and reasoning, and token 33 for design (positions above use a prefix length, not a one-based index). All repeats within each engine have one output-ID hash and one text hash per prompt in these trials, including the independent code reloads. This establishes repeatability for these tested conditions, not for every batch shape.

The code/reasoning mismatch exists at the first target output, before any draft proposal can affect the continuation. Confirmed numerical differences include target residual-sum normalization (Sharar rounds the sum to BF16 before normalization; SGLang normalizes the FP32 sum) and FP8 weight requantization (Sharar uses an FP32 intermediate; SGLang rounds through FP16). Other differences remain in normalization, activation rounding and recurrent/attention kernels. Their individual contributions to these continuations have not all been isolated.

Untimed first-token probes reproduce the first output ID of every measured generation. On code, Sharar gives `logit(#) − logit(I) = +3.375`; SGLang gives the same difference as `−3.1875` (computed from its returned logprobs, whose common normalizer cancels). The relative preference shifts by **6.5625 logit units**. SGLang leads its second-best token by 2.125 units; Sharar leads its second-best by 0.375. This is a substantial target-output difference, not evidence of a sampling-seed problem or merely an exact argmax tie. It does not identify a single responsible kernel; the scores follow all 64 target layers.

Temperature zero selects the largest computed logit; it does not make two implementations compute the same logits. SGLang itself documents [reduction-order and batching effects under greedy decoding](https://docs.sglang.io/docs/advanced_features/deterministic_inference). These numerical differences explain why exact agreement requires a separate test; they do not turn a failed equality requirement into a pass. Sharar's older tolerance-based parity checks are explicitly distinguished from exact output equality. The prior FP32-norm serving candidate remains disabled after its quality regressions.

The throughput comparison is valid for these full engine configurations and generated continuations. It is **not an identical-output speed comparison**, and exact cross-engine parity has not been fixed.

## Protocol and reproduction

- Same target revision `319f741cce68d7914884900c138a1fbb70a42f30` and DFlash2 revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`.
- SGLang image `sha256:60166a99b6610e2b4b3febbeb6411e4cbc9338e996ab6c80d3339f86a7b5b9c3`, source `5f55db35e`. Sharar runs the working source mounted read-only into its pinned image; source hashes are retained.
- Identical raw `/v1/completions` payloads, including the 10,000-element prompt list. No extra chat template or client re-tokenization. Temperature 0, seed 42, top-p 1, top-k −1, neutral penalties, no stop strings, ignored EOS, max tokens 1,000, one completion. Server seed also 42.
- Target FP8 KV, FP32 recurrent state, 60,000 KV entries, 24 state slots, maximum batch four, prefill chunks 2,048, verification width 16; actual benchmark concurrency one.
- Full cache flush before every request; server cache reporting verifies no reused prompt tokens. Frozen prompts and their hashes are retained.
- Stock AIPerf CLI uses `inputs_json`, `--streaming`, `--use-server-token-count`, and `--export-level raw`. It sends exact replay payloads as described in the [official AIPerf replay documentation](https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/inputs-json-replay). Each invocation sends one request so the harness can flush between repeats.
- Every response is independently checked against AIPerf's actual raw payload, returned prompt IDs, raw output IDs, authoritative usage, length finish reason, and final stream terminator. Complete text and IDs are saved; text is never re-tokenized to manufacture output counts.
- Sharar's completions API now supports optional raw token-ID reporting and a final streaming usage chunk with cached-token counts. This only changes reporting, not target or draft execution. The API tests cover chunk boundaries and optional fields; the audit tests reject inconsistent/missing evidence.
- Final CPU suite: 369 passed, 8 skipped (6.52 s). Real AIPerf requests exercise the updated HTTP path on GB10.

The first qualification load, `sharar-A`, is retained but excluded because CPU tests overlapped early trials. The measured sequence starts after CPU tests finish: `sglang-B`, `sharar-D`, `sharar-E`, `sglang-C`. The first two loads cover all three prompts; the last two repeat code. One untimed native SGLang logprob probe after `sglang-B` failed because it used the OpenAI `seed` spelling; the native spelling is `sampling_seed`. Its error and correction are retained separately from the successful AIPerf requests. First-token diagnostics run after the second SGLang load.

The original production container/image and active memory guard were restored. Its health, pinned metadata and exact `SPARKLANG_OK` canary pass; final MemAvailable is 57.25 GiB. Evidence root: `${SHARAR_RESULTS}/reviews/2026-09-08-aiperf/`. It contains the isolated client dependency lock, protocol, commands, source snapshots, all raw AIPerf exports, full output text/IDs, per-load first-token probes, memory/clock telemetry, and the service restoration evidence. `summary.json` is the compact machine-readable result.

Reproduce with [the client](../bench/gb10_aiperf.py) and the pinned server/window scripts in the evidence root; see [bench/README.md](../bench/README.md#exact-id-aiperf-comparison).
