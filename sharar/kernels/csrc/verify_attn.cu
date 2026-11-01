// Verify attention for the 27B on SM121 (GB10): up to 16 query rows a sequence, 6 query heads a KV head, head_dim 256,
// against a paged NVFP4 KV cache (page size 1: two E2M1 codes a byte, an E4M3 scale a block of 16), split over the
// context.
//
// One block takes a sequence's rows of one KV head's query group (96 queries: rows x 6 heads) and one chunk of its
// pages. Both products run on the FP16 tensor cores (mma m16n8k16, FP32 accumulation), FlashInfer's precision: Q from
// BF16 to FP16 in registers, K and V converted once a tile into FP16 in shared memory (exact: an E2M1 code times its
// E4M3 scale fits FP16). The products are transposed, S^T = K Q^T and O^T = V^T P^T, so the pages and the head
// dimension are the MMA's rows and a warp's 8 queries its columns: 12 compute warps, three on every SM partition, each
// holding 64 accumulators. Four producer warps gather tiles of 32 pages with cp.async (three stages, two more tiles
// pulled into L2 ahead) and convert them while the compute warps work on the tile before. An online softmax in base
// 2 a query; the block writes its chunk's normalized output and log-sum-exp, and `merge` combines the chunks. Each
// sequence's chunk is its length over the splits, read on the device: one launch (and one CUDA graph) serves any
// context. GB10, 16 rows a request over 64K: one request 0.55 -> 0.44 ms a layer, four 2.35 -> 1.64 (with
// `verify_splits`), against FlashInfer's 0.91 and 3.60.
//
// `prefix_attn` runs the same tiles for a prefill chunk against its cached prefix (no causal mask), untransposed: a
// block takes 128 consecutive (token, head) rows of one KV head's query group of a sequence, eight warps of 16 queries,
// so every SM partition holds two of them and a converted tile serves a third more rows. Its merge folds in the chunk's
// own causal attention (FlashInfer's ragged output and log-sum-exp), so the prefix needs no separate merge. GB10, a
// 2,048-token chunk over an 8K prefix: 7.1 ms against 9.8 for FlashInfer's fa2 (42 -> 58 TFLOPS).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

namespace {

constexpr int D = 256;              // head_dim
constexpr int G = 6;                // query heads a KV head
constexpr int VERIFY_WARPS = 12;    // 8 queries each: 96 = 16 rows x 6 heads
constexpr int PREFIX_WARPS = 8;     // prefill against a prefix: 128 (token, head) rows a block
// verify: four more warps load and convert while the twelve compute (a layer over 64K contexts, 16 rows a request, the
// same splits: 2.36 -> 1.85 ms at four requests; 12 + 4 warps hold at most 128 registers a thread, so no more)
constexpr int VERIFY_PRODUCERS = 4;
// verify: tiles pulled into L2 past the newest load (64K: one request 0.46 -> 0.44 ms, four 1.68 -> 1.66)
constexpr int PREFETCH = 2;
constexpr int BN = 32;              // pages a tile
constexpr int STAGES = 3;           // raw tiles: two in flight while one converts (the pipeline needs three)
constexpr int F4ROW = D / 2 + D / 16;  // an NVFP4 row: 128 bytes of codes, then 16 E4M3 scales
constexpr int HROW = 2 * D + 16;    // FP16 K and V row stride: 528 bytes, conflict-free fragment reads
constexpr int SMEM = STAGES * BN * F4ROW * 2 + 2 * 2 * BN * HROW;  // raw K and V stages, two FP16 K and V tiles
static_assert(SMEM <= 101376, "GB10 gives a block at most 99 KB of shared memory");

__device__ __forceinline__ void mma_f16(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int n = valid ? 16 : 0;  // zero-fill an invalid row
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t r[4], const void* smem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t r[4], const void* smem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) {
  __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// Four E2M1 codes (two bytes of a word's half `w`, low nibble first) times `scale` (f16x2) -> two f16x2 words, by the
// hardware conversion (exact: a code's two significant bits times an E4M3 scale's four fit FP16)
__device__ __forceinline__ void e2m1x4_f16(uint32_t w16, uint32_t scale2, uint32_t& lo, uint32_t& hi) {
  asm("{\n .reg .b8 b0, b1, b2, b3;\n mov.b32 {b0, b1, b2, b3}, %2;\n"
      " cvt.rn.f16x2.e2m1x2 %0, b0;\n cvt.rn.f16x2.e2m1x2 %1, b1;\n}\n"
      : "=r"(lo), "=r"(hi) : "r"(w16));
  asm("mul.rn.f16x2 %0, %0, %2;\n mul.rn.f16x2 %1, %1, %2;\n" : "+r"(lo), "+r"(hi) : "r"(scale2));
}

struct Params {
  const __nv_bfloat16* q;   // [tokens, QH, D]
  const uint8_t* k;         // [pages, 1, KVH, D / 2] E2M1 codes
  const uint8_t* v;
  const int32_t* kv_indptr;
  const int32_t* kv_indices;
  const int32_t* qo_indptr;
  float* o;                 // [tokens, QH, splits, D]
  float* lse;               // [tokens, QH, splits]
  int64_t q_stride;         // elements a token
  int64_t page_stride;      // bytes a page of codes
  const uint8_t* ksf;       // scales [pages, 1, KVH, D / 16] E4M3
  const uint8_t* vsf;
  int64_t sf_page_stride;
  int kvh_count;
  int splits;
  float scale_log2;
  const int32_t* blocks;    // prefix_attn: [blocks, 2] each block's sequence and first flattened (token, head) row
};

// A block's KV tiles: a tile's raw NVFP4 rows gathered into a stage, and converted to FP16 into a tile buffer. The
// movers are the MOVERS threads from `first`.
struct Tiles {
  uint8_t* ks;  // raw K and V stages [STAGES][BN][F4ROW], then the FP16 K and V tiles [2][BN][HROW]
  uint8_t* vs;
  uint8_t* kh;
  uint8_t* vh;
  const uint8_t* kbase;
  const uint8_t* vbase;
  const uint8_t* ksf;
  const uint8_t* vsf;
  const int32_t* pages;  // the sequence's pages
  int64_t page_stride, sf_page_stride;
  int hi;

  __device__ __forceinline__ Tiles(uint8_t* smem, const Params& p, int kvh, int k0, int hi_)
      : ks(smem), vs(smem + STAGES * BN * F4ROW), kh(smem + 2 * STAGES * BN * F4ROW), vh(kh + 2 * BN * HROW),
        kbase(p.k + static_cast<int64_t>(kvh) * (D / 2)), vbase(p.v + static_cast<int64_t>(kvh) * (D / 2)),
        ksf(p.ksf + static_cast<int64_t>(kvh) * (D / 16)), vsf(p.vsf + static_cast<int64_t>(kvh) * (D / 16)),
        pages(p.kv_indices + k0), page_stride(p.page_stride), sf_page_stride(p.sf_page_stride), hi(hi_) {}

  // a mover's items of a tile (BN rows of 8 chunks of codes and one of scales, for K and V)
  template <int MOVERS>
  static constexpr int ITEMS = (BN * 9 + MOVERS - 1) / MOVERS;

  // the pages of a mover's items of the tile from `start` (read ahead of its load, so the copies wait on no index)
  template <int MOVERS>
  __device__ __forceinline__ void fetch(int first, int start, int (&pg)[ITEMS<MOVERS>]) const {
#pragma unroll
    for (int n = 0; n < ITEMS<MOVERS>; ++n) {
      const int i = threadIdx.x - first + n * MOVERS, tok = start + i / 9;
      pg[n] = i < BN * 9 && tok < hi ? pages[tok] : 0;
    }
  }

  template <int MOVERS>
  __device__ __forceinline__ void load(int first, int stage, int start, const int (&pg)[ITEMS<MOVERS>]) const {
#pragma unroll
    for (int n = 0; n < ITEMS<MOVERS>; ++n) {
      const int i = threadIdx.x - first + n * MOVERS;
      if (i >= BN * 9) break;
      const int r = i / 9, c = i % 9;
      const bool ok = start + r < hi;
      const int64_t src = c < 8 ? pg[n] * page_stride + c * 16 : pg[n] * sf_page_stride;
      cp_async16(ks + (stage * BN + r) * F4ROW + c * 16, (c < 8 ? kbase : ksf) + src, ok);
      cp_async16(vs + (stage * BN + r) * F4ROW + c * 16, (c < 8 ? vbase : vsf) + src, ok);
    }
  }

  template <int MOVERS>
  __device__ __forceinline__ void load(int first, int stage, int start) const {
    int pg[ITEMS<MOVERS>];
    fetch<MOVERS>(first, start, pg);
    load<MOVERS>(first, stage, start, pg);
  }

  // producers: a tile PREFETCH tiles past the newest load is pulled into L2 (no shared memory: more DRAM reads in
  // flight than the stages hold); a mover's lines of it, (page, part) with parts K codes, K scales, V codes, V scales
  template <int MOVERS>
  static constexpr int LINES = (BN * 4 + MOVERS - 1) / MOVERS;

  template <int MOVERS>
  __device__ __forceinline__ void fetch_lines(int first, int start, int (&pg)[LINES<MOVERS>]) const {
#pragma unroll
    for (int n = 0; n < LINES<MOVERS>; ++n) {
      const int i = threadIdx.x - first + n * MOVERS, tok = start + i / 4;
      pg[n] = i < BN * 4 && tok < hi ? pages[tok] : -1;
    }
  }

  template <int MOVERS>
  __device__ __forceinline__ void prefetch(int first, const int (&pg)[LINES<MOVERS>]) const {
#pragma unroll
    for (int n = 0; n < LINES<MOVERS>; ++n) {
      const int part = (threadIdx.x - first + n * MOVERS) & 3;
      if (pg[n] < 0) continue;
      const uint8_t* a = part == 0 ? kbase + pg[n] * page_stride : part == 1 ? ksf + pg[n] * sf_page_stride
                       : part == 2 ? vbase + pg[n] * page_stride : vsf + pg[n] * sf_page_stride;
      asm volatile("prefetch.global.L2 [%0];" :: "l"(a));
    }
  }

  // a tile's K and V to FP16 (exact) into tile buffer cb, once for every warp: a block of 16 values (one scale) a step
  template <int MOVERS>
  __device__ __forceinline__ void convert(int first, int stage, int cb) const {
    for (int i = threadIdx.x - first; i < BN * (D / 16) * 2; i += MOVERS) {
      const bool isv = i >= BN * (D / 16);
      const int j = isv ? i - BN * (D / 16) : i;
      const int r = j / (D / 16), blk = j % (D / 16);
      const uint8_t* raw = (isv ? vs : ks) + (stage * BN + r) * F4ROW;
      const uint2 codes = *reinterpret_cast<const uint2*>(raw + blk * 8);
      const __half_raw sh = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(raw[D / 2 + blk]), __NV_E4M3);
      const uint32_t s2 = static_cast<uint32_t>(sh.x) | (static_cast<uint32_t>(sh.x) << 16);
      uint32_t h[8];  // 16 values as f16x2, in order
      e2m1x4_f16(codes.x & 0xffffu, s2, h[0], h[1]);
      e2m1x4_f16(codes.x >> 16, s2, h[2], h[3]);
      e2m1x4_f16(codes.y & 0xffffu, s2, h[4], h[5]);
      e2m1x4_f16(codes.y >> 16, s2, h[6], h[7]);
      uint4* dst = reinterpret_cast<uint4*>((isv ? vh : kh) + (cb * BN + r) * HROW + blk * 32);
      dst[0] = make_uint4(h[0], h[1], h[2], h[3]);
      dst[1] = make_uint4(h[4], h[5], h[6], h[7]);
    }
  }
};

// The pipeline over a chunk's `tiles` from page `lo`: STAGES - 1 raw tiles in flight; each iteration converts tile
// it + 1 while it computes tile it, one barrier an iteration (it publishes tile it's conversion and frees tile it - 1's
// raw stage and buffer). With PW producer warps (the MOVERS threads from `first`) the movers wait for their own copies
// (one named barrier among them) and the compute warps never do; without, every warp loads, converts and computes.
template <int PW, int MOVERS, class Compute>
__device__ __forceinline__ void pipeline(const Tiles& tl, int first, int tiles, int lo, Compute compute) {
  const bool mover = static_cast<int>(threadIdx.x) >= first;
  int pg[Tiles::ITEMS<MOVERS>];  // producers: the pages of the next tile to load
  int lp[Tiles::LINES<MOVERS>];  // and of the next tile to prefetch into L2
  if (mover) {
    if (PW && PREFETCH) {
      for (int i = STAGES - 1; i < STAGES - 1 + PREFETCH; ++i) {
        tl.fetch_lines<MOVERS>(first, lo + i * BN, lp);
        tl.prefetch<MOVERS>(first, lp);
      }
      tl.fetch_lines<MOVERS>(first, lo + (STAGES - 1 + PREFETCH) * BN, lp);
    }
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
      if (i < tiles) tl.load<MOVERS>(first, i, lo + i * BN);
      cp_commit();
    }
    if (PW) tl.fetch<MOVERS>(first, lo + (STAGES - 1) * BN, pg);
    cp_wait<STAGES - 2>();
    if (PW) asm volatile("bar.sync 1, %0;" :: "n"(MOVERS));
  }
  if (!PW) __syncthreads();
  if (mover && tiles > 0) tl.convert<MOVERS>(first, 0, 0);
  for (int it = 0; it < tiles; ++it) {
    if (PW) {
      __syncthreads();  // tile it's conversion is published; tile it - 1's buffer and raw stage are free
      if (mover) {
        if (it + STAGES - 1 < tiles) {
          tl.load<MOVERS>(first, (it + STAGES - 1) % STAGES, lo + (it + STAGES - 1) * BN, pg);
          tl.fetch<MOVERS>(first, lo + (it + STAGES) * BN, pg);
        }
        if (PREFETCH && it + STAGES - 1 + PREFETCH < tiles) {
          tl.prefetch<MOVERS>(first, lp);
          tl.fetch_lines<MOVERS>(first, lo + (it + STAGES + PREFETCH) * BN, lp);
        }
        cp_commit();
        cp_wait<STAGES - 2>();  // tile it + 1's raw rows
        asm volatile("bar.sync 1, %0;" :: "n"(MOVERS));
        if (it + 1 < tiles) tl.convert<MOVERS>(first, (it + 1) % STAGES, (it + 1) & 1);
      } else {
        compute(it & 1, lo + it * BN);
      }
    } else {
      cp_wait<STAGES - 3 < 0 ? 0 : STAGES - 3>();  // tile it + 1's raw rows (and tile it's)
      __syncthreads();
      if (it + STAGES - 1 < tiles) tl.load<MOVERS>(first, (it + STAGES - 1) % STAGES, lo + (it + STAGES - 1) * BN);
      cp_commit();
      if (it + 1 < tiles) tl.convert<MOVERS>(first, (it + 1) % STAGES, (it + 1) & 1);
      compute(it & 1, lo + it * BN);
    }
  }
}

// prefix_attn: W warps (16 queries each) load, convert and compute
template <int W>
__global__ void __launch_bounds__(W * 32, 1) prefix_kernel(const Params p) {
  extern __shared__ __align__(16) uint8_t smem[];
  const int seq = p.blocks[2 * blockIdx.x], kvh = blockIdx.y, split = blockIdx.z;
  const int row0 = p.blocks[2 * blockIdx.x + 1];  // the block's first flattened row
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int QH = p.kvh_count * G;
  const int q0 = p.qo_indptr[seq], n_rows = p.qo_indptr[seq + 1] - q0;
  const int k0 = p.kv_indptr[seq], kv_len = p.kv_indptr[seq + 1] - k0;
  const int per = (kv_len + p.splits - 1) / p.splits;
  const int chunk = (per + BN - 1) / BN * BN;
  const int lo = split * chunk, hi = min(lo + chunk, kv_len);
  const Tiles tl(smem, p, kvh, k0, hi);

  // the warp's queries m = 16 warp + g and + 8 (past the block's first row): row m / 6, head m % 6
  int row[2], head[2];
  bool live[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int m = row0 + warp * 16 + g + 8 * h;
    row[h] = m / G;
    head[h] = kvh * G + m % G;
    live[h] = row[h] < n_rows;
  }
  // a warp whose queries are all past the rows only loads
  const bool warp_live = (row0 + warp * 16) / G < n_rows;

  // Q in FP16 A fragments (m16n8k16): k-step c holds dims 16 c + 2 t, + 1 (regs 0, 1) and + 8, + 9 (regs 2, 3)
  uint32_t qa[D / 16][4];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const __nv_bfloat16* src = p.q + static_cast<int64_t>(q0 + row[h]) * p.q_stride + static_cast<int64_t>(head[h]) * D;
#pragma unroll
    for (int c = 0; c < D / 16; ++c) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        uint32_t w = 0;
        if (live[h]) {
          const __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162*>(src + c * 16 + half * 8 + 2 * t);
          w = pack_f16(__low2float(x), __high2float(x));
        }
        qa[c][h + 2 * half] = w;
      }
    }
  }

  float acc[D / 8][4];
#pragma unroll
  for (int j = 0; j < D / 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};

  const int tiles = hi > lo ? (hi - lo + BN - 1) / BN : 0;
  auto compute = [&](int cb, int start) {
    if (warp_live) {
      // S = Q K^T: BN / 8 tiles of 8 pages; B fragments K[page g][16 c + 2 t, + 1] and [+ 8, + 9]
      const uint8_t* kt = tl.kh + cb * BN * HROW;
      float s[BN / 8][4];
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
        // two k-steps a ldmatrix x4: matrices (pages 8 j .. + 7) x dims 16 c + 0..7, + 8..15, 16 (c + 1) + 0..7, + 8..15
        const uint8_t* kr = kt + (j * 8 + (lane & 7)) * HROW + (lane >> 3) * 16;
#pragma unroll
        for (int c = 0; c < D / 16; c += 2) {
          uint32_t b[4];
          ldsm_x4(b, kr + c * 32);
          mma_f16(s[j], qa[c], b[0], b[1]);
          mma_f16(s[j], qa[c + 1], b[2], b[3]);
        }
      }
      // scale, mask, online softmax (rows g: s[.][0..1]; g + 8: s[.][2..3])
      float mnew[2] = {mrow[0], mrow[1]};
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int h = e >> 1;
          const int tok = start + j * 8 + 2 * t + (e & 1);
          const float x = s[j][e] * p.scale_log2;
          s[j][e] = (tok < hi && live[h]) ? x : -INFINITY;
          mnew[h] = fmaxf(mnew[h], s[j][e]);
        }
      }
      float alpha[2], rsum[2] = {0.f, 0.f};
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        mnew[h] = fmaxf(mnew[h], __shfl_xor_sync(0xffffffffu, mnew[h], 1));
        mnew[h] = fmaxf(mnew[h], __shfl_xor_sync(0xffffffffu, mnew[h], 2));
        const float base = mnew[h] == -INFINITY ? 0.f : mnew[h];
        alpha[h] = exp2f(mrow[h] - base);
        mrow[h] = mnew[h];
        mnew[h] = base;
      }
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int h = e >> 1;
          s[j][e] = exp2f(s[j][e] - mnew[h]);
          rsum[h] += s[j][e];
        }
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        rsum[h] += __shfl_xor_sync(0xffffffffu, rsum[h], 1);
        rsum[h] += __shfl_xor_sync(0xffffffffu, rsum[h], 2);
        lrow[h] = lrow[h] * alpha[h] + rsum[h];
      }
#pragma unroll
      for (int n = 0; n < D / 8; ++n) {
        acc[n][0] *= alpha[0]; acc[n][1] *= alpha[0];
        acc[n][2] *= alpha[1]; acc[n][3] *= alpha[1];
      }
      // O += P V: P's accumulators are the A fragments of k16 steps (pages 16 kk .. 16 kk + 15)
#pragma unroll
      for (int kk = 0; kk < BN / 16; ++kk) {
        uint32_t pa[4];
        pa[0] = pack_f16(s[2 * kk][0], s[2 * kk][1]);
        pa[1] = pack_f16(s[2 * kk][2], s[2 * kk][3]);
        pa[2] = pack_f16(s[2 * kk + 1][0], s[2 * kk + 1][1]);
        pa[3] = pack_f16(s[2 * kk + 1][2], s[2 * kk + 1][3]);
        // two 8-dim n-tiles a ldmatrix x4: matrices (pages 0-7, dims n), (pages 8-15, dims n), then n + 8
        const int mtx = lane >> 3, r8 = lane & 7;
        const int prow = kk * 16 + (mtx & 1) * 8 + r8;
#pragma unroll
        for (int n2 = 0; n2 < D / 16; ++n2) {
          const int col = n2 * 16 + (mtx >> 1) * 8;
          uint32_t b[4];
          ldsm_x4_trans(b, tl.vh + (cb * BN + prow) * HROW + col * 2);
          mma_f16(acc[2 * n2], pa, b[0], b[1]);
          mma_f16(acc[2 * n2 + 1], pa, b[2], b[3]);
        }
      }
    }
  };
  pipeline<0, W * 32>(tl, 0, tiles, lo, compute);
  if (!warp_live) return;
  // the chunk's normalized output and log-sum-exp (base 2) a query
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!live[h]) continue;
    const int64_t at = (static_cast<int64_t>(q0 + row[h]) * QH + head[h]) * p.splits + split;
    const float inv = lrow[h] > 0.f ? 1.f / lrow[h] : 0.f;
    float* dst = p.o + at * D;
#pragma unroll
    for (int n = 0; n < D / 8; ++n) {
      const float2 v = make_float2(acc[n][2 * h] * inv, acc[n][2 * h + 1] * inv);
      *reinterpret_cast<float2*>(dst + n * 8 + 2 * t) = v;
    }
    if (t == 0) p.lse[at] = lrow[h] > 0.f ? mrow[h] + log2f(lrow[h]) : -INFINITY;
  }
}

// 8x8 b16 transpose within the warp: thread (g, t) gets elements (2 t, g) and (2 t + 1, g)
__device__ __forceinline__ uint32_t movt(uint32_t x) {
  uint32_t y;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(y) : "r"(x));
  return y;
}

// verify: the transposed products, 8 queries a warp, so 96 queries (16 rows x 6 heads) take 12 compute warps, three
// on every SM partition (16 queries a warp took six, two partitions half idle). S^T = K Q^T with K's pages as the
// MMA's rows (A by ldmatrix) and Q in B fragments; P^T's accumulators become B fragments by movmatrix; O^T = V^T P^T
// with V^T by ldmatrix .trans. A warp holds 64 accumulators (256 dims x 8 queries) and 32 registers of Q. PW more
// warps load and convert the tiles.
template <int W, int PW>
__global__ void __launch_bounds__((W + PW) * 32, 1) verify_kernel(const Params p) {
  constexpr int MOVERS = (PW ? PW : W) * 32, FIRST = PW ? W * 32 : 0;  // the threads that load and convert
  extern __shared__ __align__(16) uint8_t smem[];
  const int seq = blockIdx.x, kvh = blockIdx.y, split = blockIdx.z;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int QH = p.kvh_count * G;
  const int q0 = p.qo_indptr[seq], n_rows = p.qo_indptr[seq + 1] - q0;
  const int k0 = p.kv_indptr[seq], kv_len = p.kv_indptr[seq + 1] - k0;
  const int per = (kv_len + p.splits - 1) / p.splits;
  const int chunk = (per + BN - 1) / BN * BN;
  const int lo = split * chunk, hi = min(lo + chunk, kv_len);
  const Tiles tl(smem, p, kvh, k0, hi);

  // queries m = 8 warp + n (row m / 6, head m % 6): Q's B fragments take n = g, the accumulators' columns n = 2 t, + 1
  const int mq = warp * 8 + g;
  const bool live_q = mq / G < n_rows;
  int row[2], head[2], sees[2];
  bool live[2];
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    const int m = warp * 8 + 2 * t + e;
    row[e] = m / G;
    head[e] = kvh * G + m % G;
    live[e] = row[e] < n_rows;
    sees[e] = kv_len - n_rows + row[e];  // causal: a row sees the context through itself
  }
  const bool warp_live = warp < W && (warp * 8) / G < n_rows;

  // Q^T in FP16 B fragments (m16n8k16): k-step c holds dims 16 c + 2 t, + 1 (reg 0) and + 8, + 9 (reg 1) of query g
  uint32_t qb[D / 16][2];
  {
    const __nv_bfloat16* src =
        p.q + static_cast<int64_t>(q0 + mq / G) * p.q_stride + static_cast<int64_t>(kvh * G + mq % G) * D;
#pragma unroll
    for (int c = 0; c < D / 16; ++c) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        uint32_t w = 0;
        if (live_q && warp < W) {
          const __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162*>(src + c * 16 + half * 8 + 2 * t);
          w = pack_f16(__low2float(x), __high2float(x));
        }
        qb[c][half] = w;
      }
    }
  }

  // O^T: m-tile i holds dims 16 i + g (regs 0, 1) and + 8 (regs 2, 3) of queries 2 t, + 1
  float acc[D / 16][4];
#pragma unroll
  for (int i = 0; i < D / 16; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};

  const int tiles = hi > lo ? (hi - lo + BN - 1) / BN : 0;
  auto compute = [&](int cb, int start) {
    if (!warp_live) return;
    const uint8_t* kt = tl.kh + cb * BN * HROW;
    const uint8_t* vt = tl.vh + cb * BN * HROW;
    // S^T = K Q^T: BN / 16 m-tiles of 16 pages; A fragments by ldmatrix x4: matrices (pages 0-7, dims 0-7),
    // (pages 8-15, dims 0-7), (pages 0-7, dims 8-15), (pages 8-15, dims 8-15) of the k-step
    float s[BN / 16][4];
#pragma unroll
    for (int j = 0; j < BN / 16; ++j) {
      s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
      const uint8_t* kr = kt + (j * 16 + (lane & 15)) * HROW + (lane >> 4) * 16;
#pragma unroll
      for (int c = 0; c < D / 16; ++c) {
        uint32_t a[4];
        ldsm_x4(a, kr + c * 32);
        mma_f16(s[j], a, qb[c][0], qb[c][1]);
      }
    }
    // scale, mask, online softmax (pages 16 j + g: s[j][0..1], + 8: s[j][2..3]; queries 2 t + (e & 1))
    float mnew[2] = {mrow[0], mrow[1]};
#pragma unroll
    for (int j = 0; j < BN / 16; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int h = e & 1;
        const int tok = start + j * 16 + g + 8 * (e >> 1);
        const float x = s[j][e] * p.scale_log2;
        s[j][e] = (tok < hi && tok <= sees[h] && live[h]) ? x : -INFINITY;
        mnew[h] = fmaxf(mnew[h], s[j][e]);
      }
    }
    float alpha[2], rsum[2] = {0.f, 0.f};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      mnew[h] = fmaxf(mnew[h], __shfl_xor_sync(0xffffffffu, mnew[h], 4));
      mnew[h] = fmaxf(mnew[h], __shfl_xor_sync(0xffffffffu, mnew[h], 8));
      mnew[h] = fmaxf(mnew[h], __shfl_xor_sync(0xffffffffu, mnew[h], 16));
      const float base = mnew[h] == -INFINITY ? 0.f : mnew[h];
      alpha[h] = exp2f(mrow[h] - base);
      mrow[h] = mnew[h];
      mnew[h] = base;
    }
#pragma unroll
    for (int j = 0; j < BN / 16; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int h = e & 1;
        s[j][e] = exp2f(s[j][e] - mnew[h]);
        rsum[h] += s[j][e];
      }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      rsum[h] += __shfl_xor_sync(0xffffffffu, rsum[h], 4);
      rsum[h] += __shfl_xor_sync(0xffffffffu, rsum[h], 8);
      rsum[h] += __shfl_xor_sync(0xffffffffu, rsum[h], 16);
      lrow[h] = lrow[h] * alpha[h] + rsum[h];
    }
#pragma unroll
    for (int i = 0; i < D / 16; ++i) {
      acc[i][0] *= alpha[0]; acc[i][1] *= alpha[1];
      acc[i][2] *= alpha[0]; acc[i][3] *= alpha[1];
    }
    // O^T += V^T P^T, a k16 step of pages a m-tile of S^T: its accumulators (pages g, + 8 of queries 2 t, + 1)
    // transposed are the B fragments (pages 2 t, + 1 and + 8, + 9 of query g)
#pragma unroll
    for (int j = 0; j < BN / 16; ++j) {
      const uint32_t b0 = movt(pack_f16(s[j][0], s[j][1])), b1 = movt(pack_f16(s[j][2], s[j][3]));
      // A = V^T by ldmatrix x4 .trans: matrices (dims 0-7, pages 0-7), (dims 8-15, pages 0-7), (dims 0-7, pages 8-15),
      // (dims 8-15, pages 8-15) of the m-tile
      const uint8_t* vr = vt + (j * 16 + ((lane >> 4) << 3) + (lane & 7)) * HROW + ((lane >> 3) & 1) * 16;
#pragma unroll
      for (int i = 0; i < D / 16; ++i) {
        uint32_t a[4];
        ldsm_x4_trans(a, vr + i * 32);
        mma_f16(acc[i], a, b0, b1);
      }
    }
  };
  pipeline<PW, MOVERS>(tl, FIRST, tiles, lo, compute);
  if (!warp_live) return;
  // the chunk's normalized output and log-sum-exp (base 2) a query
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    if (!live[e]) continue;
    const int64_t at = (static_cast<int64_t>(q0 + row[e]) * QH + head[e]) * p.splits + split;
    const float inv = lrow[e] > 0.f ? 1.f / lrow[e] : 0.f;
    float* dst = p.o + at * D;
#pragma unroll
    for (int i = 0; i < D / 16; ++i) {
      dst[16 * i + g] = acc[i][e] * inv;
      dst[16 * i + g + 8] = acc[i][2 + e] * inv;
    }
    if (g == 0) p.lse[at] = lrow[e] > 0.f ? mrow[e] + log2f(lrow[e]) : -INFINITY;
  }
}

// Combines a query's splits (and, `extra_o` given, one more normalized output with its log-sum-exp: a prefill's own
// causal attention) into its BF16 output.
__global__ void merge_kernel(const float* __restrict__ o, const float* __restrict__ lse, __nv_bfloat16* __restrict__ out,
                             int splits, int64_t out_stride, int QH, const __nv_bfloat16* __restrict__ extra_o,
                             const float* __restrict__ extra_lse) {
  const int token = blockIdx.x, head = blockIdx.y;
  const int64_t base = static_cast<int64_t>(token) * QH + head;
  float top = extra_o ? extra_lse[base] : -INFINITY;
  for (int s = 0; s < splits; ++s) top = fmaxf(top, lse[base * splits + s]);
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float num = 0.f, den = 0.f;
    for (int s = 0; s < splits; ++s) {
      const float l = lse[base * splits + s];
      if (l == -INFINITY) continue;
      const float w = exp2f(l - top);
      num += w * o[(base * splits + s) * D + d];
      den += w;
    }
    if (extra_o && extra_lse[base] != -INFINITY) {
      const float w = exp2f(extra_lse[base] - top);
      num += w * __bfloat162float(extra_o[base * D + d]);
      den += w;
    }
    out[static_cast<int64_t>(token) * out_stride + head * D + d] = __float2bfloat16(den > 0.f ? num / den : 0.f);
  }
}

}  // namespace

namespace {

Params params(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor ksf, torch::Tensor vsf, torch::Tensor kv_indptr,
              torch::Tensor kv_indices, torch::Tensor qo_indptr, torch::Tensor o, torch::Tensor lse, double sm_scale, int64_t splits,
              const int32_t* blocks) {
  TORCH_CHECK(q.size(2) == D && k.size(3) == D / 2 && q.size(1) == k.size(2) * G,
              "verify_attn: head_dim 256, 6 query heads a KV head");
  TORCH_CHECK(k.dtype() == torch::kUInt8 && v.dtype() == k.dtype() && k.stride(0) == v.stride(0) && ksf.size(3) == D / 16 &&
              ksf.stride(0) == vsf.stride(0), "verify_attn: an NVFP4 cache and its scales");
  return Params{reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(k.data_ptr()),
                reinterpret_cast<const uint8_t*>(v.data_ptr()), kv_indptr.data_ptr<int32_t>(), kv_indices.data_ptr<int32_t>(),
                qo_indptr.data_ptr<int32_t>(), o.data_ptr<float>(), lse.data_ptr<float>(), q.stride(0), k.stride(0),
                reinterpret_cast<const uint8_t*>(ksf.data_ptr()), reinterpret_cast<const uint8_t*>(vsf.data_ptr()), ksf.stride(0),
                static_cast<int>(k.size(2)), static_cast<int>(splits), static_cast<float>(sm_scale * 1.4426950408889634), blocks};
}

template <void (*kernel)(Params)>
void launch(int threads, const Params& p, unsigned blocks, unsigned kvh, cudaStream_t stream) {
  static const bool sized = [] {  // once a kernel, before any graph capture
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    return true;
  }();
  (void)sized;
  kernel<<<dim3(blocks, kvh, static_cast<unsigned>(p.splits)), threads, SMEM, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void verify_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor ksf, torch::Tensor vsf,
                 torch::Tensor kv_indptr, torch::Tensor kv_indices, torch::Tensor qo_indptr, torch::Tensor o, torch::Tensor lse,
                 torch::Tensor out, double sm_scale, int64_t splits) {
  const Params p = params(q, k, v, ksf, vsf, kv_indptr, kv_indices, qo_indptr, o, lse, sm_scale, splits, nullptr);
  auto stream = at::cuda::getCurrentCUDAStream();
  launch<verify_kernel<VERIFY_WARPS, VERIFY_PRODUCERS>>((VERIFY_WARPS + VERIFY_PRODUCERS) * 32, p, static_cast<unsigned>(qo_indptr.numel() - 1), static_cast<unsigned>(k.size(2)), stream);
  merge_kernel<<<dim3(static_cast<unsigned>(q.size(0)), static_cast<unsigned>(q.size(1))), 128, 0, stream>>>(
      o.data_ptr<float>(), lse.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(splits),
      out.stride(0), static_cast<int>(q.size(1)), nullptr, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prefix_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor ksf, torch::Tensor vsf,
                 torch::Tensor kv_indptr, torch::Tensor kv_indices, torch::Tensor qo_indptr, torch::Tensor blocks, torch::Tensor o,
                 torch::Tensor lse, torch::Tensor out, double sm_scale, int64_t splits, c10::optional<torch::Tensor> extra_o,
                 c10::optional<torch::Tensor> extra_lse) {
  TORCH_CHECK(blocks.dtype() == torch::kInt && blocks.is_contiguous() && blocks.dim() == 2 && blocks.size(1) == 2, "prefix_attn: blocks");
  TORCH_CHECK(!extra_o || (extra_o->is_contiguous() && extra_o->dtype() == torch::kBFloat16 && extra_lse->is_contiguous() &&
                           extra_lse->dtype() == torch::kFloat && extra_o->numel() == q.numel()), "prefix_attn: the new rows' output");
  const Params p = params(q, k, v, ksf, vsf, kv_indptr, kv_indices, qo_indptr, o, lse, sm_scale, splits, blocks.data_ptr<int32_t>());
  auto stream = at::cuda::getCurrentCUDAStream();
  launch<prefix_kernel<PREFIX_WARPS>>(PREFIX_WARPS * 32, p, static_cast<unsigned>(blocks.size(0)), static_cast<unsigned>(k.size(2)), stream);
  merge_kernel<<<dim3(static_cast<unsigned>(q.size(0)), static_cast<unsigned>(q.size(1))), 128, 0, stream>>>(
      o.data_ptr<float>(), lse.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(splits),
      out.stride(0), static_cast<int>(q.size(1)), extra_o ? reinterpret_cast<const __nv_bfloat16*>(extra_o->data_ptr()) : nullptr,
      extra_lse ? extra_lse->data_ptr<float>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("verify_attn", &verify_attn);
  m.def("prefix_attn", &prefix_attn);
}
