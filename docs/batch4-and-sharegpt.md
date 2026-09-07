# Batch-4 verify costs, small-M GEMM policy, and the ShareGPT comparison

Date: 2026-09-07 (night). Source baseline `c39092a`.

## 1. Where a batch-4 round goes

Kernel attribution of one DFlash2 k=15 round (16 verify rows per request),
`verify_b4_profile.py`, GPU milliseconds per round:

| Kernel family | B=1 (16 rows) | B=4 (64 rows) |
|---|---:|---:|
| NVFP4 MLP GEMMs (cuDNN/CUTLASS at B=1; b12x gate/up + CUTLASS down at B=4) | 50.2 | 55.2 |
| FP8 projections (`_scaled_mm`, nvjet + xmma kernels) | 35.5 | ~36 |
| Draft (bf16, 5 layers) | ~5 | ~11 |
| b/a gate projections (bf16 `x @ W^T`, 48 launches) | 0.2 | 4.3 |
| GDN verify kernel + reconstruction | 3.3 | 11.7 |
| Attention (8K context) | 0.3 | 6.5 |
| Round wall time | 118 | 140 (144 at 8K) |

Microbenchmarks at the real projection shapes (`smallm_gemm.py`) put the FP8
GEMMs at M=64 within 10 to 30% of their weight floor with either
`_scaled_mm` or FlashInfer, so they are not the lever. Two things were:

- **NVFP4 gate/up backend.** cuDNN beats b12x up to 64 rows (M=64: 0.51 vs
  0.62 ms per call, 64 calls per round); b12x wins at prefill M. The policy
  threshold moved from 16 to 64 rows.
- **b/a gate projections: a wrong attribution, reverted.** The 4.3 ms
  `cutlass_80 bf16` kernel in the batch-4 profile was assumed to be the 48
  per-layer [M, 5120] x [96, 5120]^T gate projections. A Triton replacement
  was written, tested and measured: the cuBLAS kernel stayed (it belongs to
  a different bf16 product in the round) and the Triton kernel added 3.3 ms
  of its own (two programs serializing an 80-step K loop). Removed again.

Result of the policy change alone (`verify_b4_profile.py`, kernel time per
round): NVFP4 GEMMs 55.2 to 46.7 ms at batch 4, round 140.8 to 137.4 ms
(about 134 ms without the reverted kernel); batch 1 unchanged at 118 ms.

## 2. What was not done

- The FP8 projections and the NVFP4 GEMMs are at their floors for these row
  counts; batch-4 rounds are weight-bound like batch-1 rounds.
- The GEMM-epilogue NVFP4 quantization (about 12% of prefill) still needs a
  CUTLASS SM120 kernel. CUTLASS 4.5 headers and nvcc 13 are in the dev
  image; the tractable design is two GEMMs (gate, up) where the up GEMM's
  epilogue loads the bf16 gate as an auxiliary input, applies SiLU and the
  product, and writes NVFP4 with block scale factors (`LinCombBlockScaleFactor`).
  That halves the traffic saving (about 6% of prefill) but uses only standard
  EVT nodes. Not attempted in this round; the benchmark below is
  decode-dominated and would not move.
- The conv kernel's q/k/v layout: the GDN kernels assume contiguous
  `[T, H, D]` operands, so the conv kernel would need three output pointers
  and the wrappers a stride-aware path. About 2% of prefill.

## 3. ShareGPT comparison: Sharar, SGLang DFlash2, Atlas DFlash2

`bench/sharegpt_bench.py`, same frozen prompts for every engine
(`sharegpt_inputs_v1.json`: 32 ShareGPT first turns per bucket, seed 7, the
recorded reply length as the output budget, natural stopping, temperature 0,
thinking off). Buckets by prompt length in this model's tokens: short (<= 256;
median 22 in, 253 out budget), medium (256 to 1,024; 430 in, 385 out), long
(1,024 to 4,096; 1,830 in, 287 out). Closed loop at concurrency 1, 4 and 8;
every engine allows 4 running requests, so 8 queues. All 288 requests
succeeded on every engine, no thinking tags, mean generated tokens 283 / 283
/ 273 (Sharar / SGLang / Atlas), first answers token-identical across engines.

Engines, each alone on the GPU, launched from the same host:

- Sharar: production image of this commit (DFlash2 k=15, reconstruction,
  cold tier), `run-sharar.sh` defaults.
- SGLang: `lmsysorg/sglang:dev-qwen38-27b-dflash2` with native DFLASH (8
  draft tokens, unquantized draft, FlashInfer, torch.compile, FP8 KV, FP32
  SSM, 35,000 KV tokens, 4 running), the configuration of the earlier
  competitive study.
- Atlas: the pinned source build (`136520b`, with the lm_head compatibility
  patch the earlier study needed for this checkpoint's packed NVFP4 head),
  DFlash gamma 8, window 2,048, FP8 KV, FP32 SSM, prefill 2,048, 4 seqs,
  utilization 0.47. Its DFlash2 accepts 0 to 1 of 7 drafts on this text
  (its own log; the earlier source audit found the draft-capture defects), so
  it runs close to plain decode.

Aggregate output tok/s over the bucket's makespan; in parentheses TTFT p50
in seconds and mean time per output token in ms:

| bucket | conc | Sharar | SGLang DFlash2 | Atlas DFlash2 |
|---|---:|---:|---:|---:|
| short | 1 | **33.0** (0.11, 30.5) | 29.7 (0.28, 32.6) | 16.5 (0.31, 60.2) |
| short | 4 | 56.5 (0.31, 67.5) | **85.2** (0.41, 42.5) | 48.0 (0.34, 82.0) |
| short | 8 | 67.5 (9.7, 54.6) | **86.2** (9.9, 42.4) | 49.4 (15.8, 80.6) |
| medium | 1 | **36.1** (0.20, 28.1) | 33.2 (0.36, 30.4) | 16.9 (0.89, 57.4) |
| medium | 4 | **109.4** (0.33, 34.8) | 97.1 (0.50, 38.6) | 48.5 (0.87, 78.5) |
| medium | 8 | **109.8** (13.1, 34.6) | 95.3 (13.9, 39.2) | 48.7 (24.3, 78.4) |
| long | 1 | **31.0** (0.66, 31.8) | 27.8 (0.90, 34.7) | 14.4 (2.90, 62.7) |
| long | 4 | **83.7** (1.02, 45.2) | 74.2 (1.28, 50.7) | 31.9 (3.12, 115.5) |
| long | 8 | **85.6** (13.5, 45.5) | 74.9 (14.8, 49.2) | 32.2 (38.2, 122.7) |

Reading: Sharar leads SGLang in seven of nine cells by 9 to 15% and has the
lowest TTFT everywhere; Atlas is 2 to 3x behind both. The two cells SGLang
wins are short prompts at concurrency 4 and 8, by 34 and 28%. There the
acceptance on chat text is 2 to 3 tokens per round and Sharar verifies 16
rows per request (64 rows per round, 137 ms) where SGLang verifies 8 (32
rows), so the wide block costs more than it returns in that regime.
Evidence: `${SHARAR_RESULTS}/reviews/2026-09-07h-sharegpt/` (per-request
JSONL with every output text and timing).
