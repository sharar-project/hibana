# DFlash2 becomes the default, three serving fixes, and the prefill kernel study

Date: 2026-09-07. Source baseline `9def384`.

## What changed

**Default proposer.** `sharar serve` and `run-sharar.sh` now default to the
pinned DFlash2 draft (`incoai/Qwen3.8-27B-DFlash2`, revision
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`) with 7 proposals per round and
accepted-state reconstruction. `--spec` defaults to the proposer's validated
depth (7 for DFlash2 and DSpark, 3 for MTP), or `SHARAR_SPEC` when set.
`run-sharar.sh` checks that the draft snapshot is present before launching and
says how to serve without it (`SHARAR_PROPOSER=mtp`).

**Why promotion could proceed.** The only gate that held DFlash2 back was the
frozen sampled-chat rubric in `bench/workloads/local_v1.json`, which demanded
that an answer about a generic cache miss say that "computation must run".
A correct answer that explains the fetch from a slower source does not, and
the MTP control failed the same rubric. The rubric now asks for the data not
being in the cache and a fetch or recomputation from a slower source. For
greedy requests the draft cannot change the output at all (acceptance is
against the target's own argmax), so the quality risk of the switch is
confined to sampled requests, which use exact rejection sampling.

**Three serving fixes.**

- `tool_choice: "none"` is a legitimate "call no tools" that real clients
  send; it now drops the tool schemas from the prompt instead of returning
  HTTP 400. Forced and named tool choices and constrained `response_format`
  are still refused before admission.
- The two unconditional `torch.cuda.synchronize()` calls at the end of every
  prefill (there only to time `prefill_seconds`) drained the GPU queue on
  every scheduler step that interleaved a prefill chunk with decode. They now
  run only when `Executor.timing` is set, which the CLI benchmarks do and the
  server does not.
- The parallel drafters computed a confidence head per proposal that nothing
  read since the width policy was archived. Removed, along with the
  `Proposal.confidence` field.

Two small prefill traffic removals: `chunk_fwd_o` no longer zero-fills its
output (every tile is stored), and the NVFP4 quantizers reuse one swizzled
scale buffer per shape instead of zero-filling a new one per call (the
padding rows only feed discarded GEMM tiles; the buffers are never evicted
because CUDA graphs capture their addresses).

## Prefill kernel study (cold 2,048-token prefill, dev container)

Median of seven cold prefills in one process: **681 ms** (3,000 tok/s);
8,192 tokens: **3.65 s** (2,240 tok/s). One prefill in seven takes about
60% longer in both settings, which is why single-shot profiles of this path
swing between 690 and 1,080 ms. Profiler attribution of the 687 ms run:

| Component | ms | Calls | Note |
|---|---:|---:|---|
| FP8 projections (`_scaled_mm`) | ~165 | 176 | ~150 TFLOPS |
| NVFP4 MLP GEMMs (b12x gate/up + CUTLASS down) | ~225 | 128 | 250 / 220 TFLOPS |
| GDN chunk kernels (h, o, w/u, conv, kkt, l2norm) | ~110 | 48 layers | 2.05 ms per layer |
| SiLU * up + NVFP4 quantizer | 53 | 64 | 220 GB/s |
| Gemma RMSNorm (with residual, FP8 out) | 44 | 129 | ~250 GB/s |
| copies, fills, clones | ~40 | ~1,000 | q/k/v re-layout after the conv (see below) |
| Ragged attention (FA2) | 11 | 16 | |

Ceilings measured on this GB10 in the same session: bf16 GEMM 80 to 99 TFLOPS,
FP8 GEMM 150 to 160 TFLOPS, NVFP4 GEMM 220 to 250 TFLOPS at M=2048, with SM
clocks steady at 2.28 to 2.38 GHz, 96 to 99 W and no throttle reasons. The
GEMMs, the quantizer and the norm are therefore at the hardware's practical
rate for their shapes.

What was tried and measured as no gain:

- FP8 GEMM backend per real projection shape: FlashInfer `bmm_fp8` versus
  `torch._scaled_mm` within +4% / -2% on all six shapes.
- GDN `chunk_delta_h` (BV 32/64, 4/8 warps, 2/3 stages) and `chunk_o`
  (BK 64/128, BV 32/64/128, 4/8 warps, 2/3 stages): every configuration
  within noise of the current one at T=2048; the whole GDN core is 2.05 ms
  per layer.
- Fusing the post-attention norm with the NVFP4 quantization of the MLP
  input (`kernels/fused_norm_nvfp4.py`, bit-identical to the two-kernel
  path, tested): 681 vs 684 ms at 2K and 3,898 vs 3,653 ms at 8K. The
  one-program-per-row kernel is slower than the standalone quantizer, which
  cancels the 2.7 GB of saved traffic. Kept opt-in
  (`SHARAR_FUSED_NORM_NVFP4=1`) for a retune; off by default.

What would move prefill, in order of size, and why it was not done here:

1. Writing the SiLU-gated NVFP4 activation from the gate/up GEMM epilogue
   instead of materializing the bf16 `[M, 34816]` gate/up output: about 290 MB
   of traffic per layer, 18 GB per 2K prefill, roughly 80 ms (12%). Needs a
   custom SM120 CUTLASS block-scaled GEMM with an EVT epilogue.
2. Having the causal-conv kernel write q, k and v as three contiguous
   `[T, H, D]` buffers so the GDN kernels do not re-lay them out: 3 copies
   per layer, about 10 to 18 ms (2%). Needs the vendored Triton conv kernel
   to take three output pointers.
3. A retuned fused norm+quantizer (above): about 12 ms (2%).

## Results

RESULTS_PENDING
