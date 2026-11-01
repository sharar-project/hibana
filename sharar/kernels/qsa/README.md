# QSA kernels

All production QSA kernels live in this package:

- `attention.py`: sparse attention, with SGLang's launch table and the optimized-prefill launch policy.
- `selection.py`: the `original` path's graph-safe scorer, Torch top-k and page assembly (called by
  `FlashNextIndexer.select` in `sharar/model/flashnext/indexer.py`).
- `score_exact.py`: the `original` selection's eager-prefill kernels on SM121 (`csrc/score_exact.cu`): the same
  scores, bit for bit, and the same pages.
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

- `original` (the engine default, and `exact`'s): `FlashNextIndexer.select` over `selection.py`, 256 rows per chunk
  (in eager prefill on SM121 through `score_exact.py`, with the same result).
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

## The `original` selection in eager prefill

On SM121, eager prefill rows that take the `original` selection (every row under `original`, so `exact`'s; rows of
chunks past 32K under `auto`) run `score_exact.py` instead of the Triton scorer and Torch top-k, with the same result:

1. **Scoring (CUDA):** the Triton `_score_kernel`'s FP32 order, made explicit. Triton 3.7 lays its [64 groups, 128]
   key tile out 8 dimensions a lane over 16 lanes: each lane sums its 8 products in order (BF16 products are exact
   in FP32, so a product followed by FMAs rounds as products and adds would), the lane sums meet in a butterfly
   (lanes 8 apart first, then 4, 2, 1), and the heads' relu'd sums are added from +0 in head order, then scaled.
   One thread evaluates that tree for each of its 4 x 4 (row, group) outputs, from FP32 tiles of 64 rows (one
   sequence) and 64 compressed keys in shared memory, so each key load serves 64 rows instead of one. A block
   scores 8 key tiles; the next head's queries, or the next tile's keys, load into registers while a head computes.
2. **Top-k + assembly (CUDA, any row length):** the selection of `topk_pages.cu` (the 512 largest finite scores,
   equal scores by ascending group, emitted in that order), which is the order Torch's top-k returns for these rows
   (a stable descending sort; ties are common, since relu makes zeros). Two passes over the row: an 11-bit
   histogram finds the bin of the 512th score, and the second takes the scores above it and stages that bin in
   shared memory, where three more radix passes find the exact threshold. A bin of more than 4,096 scores is
   resolved over the row instead, and a tie at the threshold is cut in group order by one more ordered pass.

`tests/gpu/test_qsa_exact.py` checks the scores' bits against the Triton kernel and the pages against
`select_pages` (ties, zeros, crowded bins, partly visible rows, several sequences), and the attention output
through the indexer. A Triton upgrade that changed `_score_kernel`'s layout would fail the first check.

Plans are shared across layers through the forward's side inputs, not persisted
in a model or cache. Both backends use the same physical FP8 K/V and interleaved
BF16 index-cache layouts.
CUDA sources and their license ship as package assets; TVM-FFI's normal JIT
cache is used. Native modules are loaded before graph capture.
