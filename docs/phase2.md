# Phase 2 notes: decode loop, CUDA graphs, kernel fusions

Measured on 2026-09-03 in the dev container. Reference numbers are the frozen
SGLang container's (`docs/parity.md`, roadmap section 3).

## What exists

- `sharar/memory/kv_pool.py`: paged FP8 KV for the 16 attention layers,
  page size 1, page 0 reserved for padded batch rows.
- `sharar/memory/state_pool.py`: per-layer conv/SSM slots for the 48 GDN
  layers with committed slots now and scratch slots reserved for Phase 7.
- `sharar/engine/executor.py`: admit / prefill / decode_step / generate for
  batches up to 4; `sharar/engine/graphs.py`: one CUDA graph per batch size
  (1, 2, 4) with padded dummy rows and FlashInfer's graph-mode decode wrapper.
- `sharar/engine/sampler.py`: greedy, temperature, top-k, top-p, seeds.
- Fused kernels in `sharar/kernels/`: `fused_norm.py` (Gemma RMSNorm with
  residual add and FP8 output), `fp8_quant.py` (static FP8 quant, gated
  variant for the attention output), `qk_norm_rope.py` (per-head norm + RoPE
  from the qkv projection), `nvfp4_quant.py` (NVFP4 activation quantizer bit
  compatible with FlashInfer's, plus the fused SiLU-mul variant),
  `csrc/fp8_gemv.cu` (SM120 warp-per-row FP8 GEMV for M <= 4, ported from
  SGLang with attribution and extended to batched rows).

## Results

| Metric | Sharar | Reference |
|---|---:|---:|
| Decode, batch 1, CUDA graphs | 12.06 tok/s (82.9 ms/step) | 12.74 tok/s (78.5 ms/step) |
| Decode, batch 2 | 23.1 aggregate tok/s | n/a |
| Decode, batch 4 | 44.6 aggregate tok/s | 46.4 aggregate tok/s |
| Prefill 2,048 tokens (GPU) | 813 to 819 ms across runs | 810 ms TTFT over HTTP |
| Prefill 8,192 tokens | 3.9 s | 3.3 s TTFT |
| Canaries (greedy, decoded text) | 4/4 pass | 4/4 pass |
| Determinism (rerun, eager vs graph, rows in a batch) | identical | n/a |

Canary continuations match the reference token for token on exact (5/5),
arithmetic (3/3) and needle (12/12); the JSON canary produces valid JSON in a
different formatting, inside the noise floor of `docs/parity.md`.

## Where decode time goes (batch 1, eager, per step)

| Kernel group | ms | Bytes streamed | Effective GB/s |
|---|---:|---:|---:|
| NVFP4 GEMMs (MLP + lm_head, CUTLASS SM120) | 46.8 | 10.3 GB | 220 |
| FP8 GEMVs (attention and GDN projections) | 31.0 | 6.0 GB | 194 |
| Everything else (GDN recurrence, conv, norms, attention, sampling) | 3.6 | | |

Every GEMM kernel available here (cuBLAS, CUTLASS, cuDNN, the b12x backend,
Triton variants, and the CUDA port) tops out near 225 GB/s on GB10 with cold
weights, about 82% of the 273 GB/s figure. The FP8 GEMVs run at 194 GB/s in
the model versus 225 in isolation; closing that gap would put the step at the
reference's 78.5 ms. That work belongs to Phase 9.

## Prefill findings

- FlashInfer's plain `fp4_quantize` ("cuda" backend) ran at about 43 GB/s and
  its fused SiLU quantizer at about 120 GB/s. Both are replaced by Triton
  kernels that are bit compatible with the FlashInfer layout. The Triton
  versions are correct but not yet faster: the 129 quantizer launches of a
  2,048-token prefill take 54 ms either way, so the e2m1 rounding and nibble
  packing (currently `tl.where` chains plus a reshape-sum) are compute bound
  and are a Phase 9 item worth about 40 ms per 2K prefill.
- `mm_fp4` backend by shape (cold L2): `b12x` is fastest for the gate/up GEMM
  at M=2048 (336 vs 305 TFLOPS for CUTLASS), `cudnn` for the down GEMM (309 vs
  285). At M <= 4 all backends are within noise; `cudnn` is used.
- `torch._scaled_mm` and FlashInfer `bmm_fp8` are equivalent for FP8 at
  M=2048 (about 190 TFLOPS).
- Norm, quantization and rope now run as single fused kernels; the GDN
  projection is split into two contiguous GEMMs using the reference's shared
  requantized weights, which removed the last large copy.

## Decisions

- Batch-size-dependent rounding is real and expected: GEMM tiling differs by
  M, so batch 1 and batch 4 give different (both valid) tokens. The
  determinism test compares eager with graph at equal batch size.
- The KV page size stays 1 (the reference's choice); a page-size measurement
  is deferred to Phase 3 where the radix cache decides the page granularity.
- First use of a new (shape, kernel) pair compiles Triton/cuDNN code; a warmup
  over the supported shapes belongs to `sharar serve` (Phase 5).
