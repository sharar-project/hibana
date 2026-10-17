# QSA kernels

All production QSA kernels live in this package:

- `attention.py`: sparse attention, with SGLang's launch table and the optimized-prefill launch policy.
- `selection.py`: the `original` path's graph-safe scorer, Torch top-k and page assembly (called by
  `FlashNextIndexer.select` in `sharar/model/flashnext/indexer.py`).
- `prefill.py`: optimized eager-prefill adapter and per-forward planning.
- `query_prep.py`, `compression.py`, `maintenance_plan.py`: fused index maintenance.
- `tiled_scaled.py`: optimized tiled scorer.
- `topk_pages.py`: fused radix top-k and page assembly.
- `csrc/`: native CUDA sources and the SGLang license for their adaptations.
- `LICENSE.vllm`: upstream license for the adapted Triton scorer in `tiled_scaled.py`.

The package loads the GPU attention code only when `sparse_attention` is imported
(`from sharar.kernels.qsa import sparse_attention`), so CPU-only tooling can import
the prefill adapter without Triton.

## Kernel design

The optimized eager-prefill path uses the same index-projection GEMM and physical
cache layouts as the `original` path. Its pipeline is:

1. **Index maintenance (CUDA):** fuse index-query Gemma RMSNorm/RoPE with raw-key
   scatter. A second kernel averages each completed four-token key group in
   FP32 and applies normalization/RoPE to its compressed key. Raw and compressed
   BF16 keys live in the per-page auxiliary cache.
2. **Scoring (Triton):** tile queries and compressed keys for tensor-core dot
   products with FP32 accumulation. Sum `relu(q · k)` across the four index heads
   and scale by `1/sqrt(128)`. Causal masking and initialization are fused into
   the score write, so every output entry is defined in one launch.
3. **Top-k + assembly (CUDA):** one CTA per score row stages sortable FP32 keys
   in shared memory, finds the top-512 threshold with four byte-wise radix
   passes, then sorts only the winners. Expand selected groups into physical
   pages and append the pending tail, emitting up to 2,051 pages and a count.
   Boundary ties prefer ascending logical group index. Scoring and selection
   are separate kernels; the FP32 score buffer is chunked to bound scratch use.
4. **Sparse attention (Triton):** one program per query/KV-head gathers selected
   FP8 K/V, converts to BF16, and performs tiled QK, online softmax and PV with
   FP32 output accumulation. The 12 query heads sharing a KV head reuse its
   gathered tile. No full attention-probability matrix is materialized; the
   optimized large-prefill policy changes launch parameters, not the kernel body.

Compression/page-table plans are built from host-known forward metadata and
reused across attention layers within that forward. Projections, the main K/V
cache write, output gating and output projection remain outside these fusions.

## Eager-prefill backend selection

`SHARAR_QSA_PREFILL_BACKEND` selects one of:

- `original` (the engine default, and `exact`'s): `FlashNextIndexer.select` over `selection.py`, 256 rows per chunk.
- `auto` (Flash-Next's `fast` and `throughput` profiles): the optimized kernels when eligible, otherwise `original`.
- `optimized`: require a compatible SM121 QSA model and non-stable numerics;
  reject unsupported sparse eager-prefill inputs, including contexts over 32K.

Both `auto` and `optimized` select fused index query/raw-key maintenance,
key compression, tiled FP32 score fill/scale and radix top-512/page assembly
on eligible SM121 TP1/TP2 eager sparse-prefill calls with <=32K context. TP2
uses 12 query heads / 1 KV head per rank and the replicated 4-head indexer.
Dense prefill, decode/verify/capture and replicated MTP attention are outside
this setting and use the `original` path. With `auto`, stable numerics and
unsupported shapes use the `original` kernels. With `optimized`, incompatible
settings fail at construction and unsupported sparse eager-prefill inputs
fail before index-cache writes. Attention runs the same kernel body; eligible calls
of more than 512 rows launch with BLOCK_N 16, one warp and one stage.

The score reduction order and stable ascending-logical-index tie selection
can differ from the `original` Torch top-k. Compilation/launch errors in either
optimized policy surface to the caller without silently retrying another backend.

Plans are shared across layers through the forward's side inputs, not persisted
in a model or cache. Both backends use the same physical FP8 K/V and interleaved
BF16 index-cache layouts.
CUDA sources and their license ship as package assets; TVM-FFI's normal JIT
cache is used. Native modules are loaded before graph capture.
