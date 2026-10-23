// Verify attention for the 27B on SM121 (GB10): up to 16 query rows a sequence, 6 query heads a KV head, head_dim 256,
// against a paged NVFP4 KV cache (page size 1: two E2M1 codes a byte, an E4M3 scale a block of 16), split over the
// context.
//
// One block takes a sequence's rows of one KV head's query group (96 queries: rows x 6 heads, 16 a warp, 6 warps) and
// one chunk of its pages. Both products run on the FP16 tensor cores (mma m16n8k16, FP32 accumulation), FlashInfer's
// precision: Q from BF16 to FP16 in registers, K and V converted once a tile into FP16 in shared memory (exact: an E2M1
// code times its E4M3 scale fits FP16), K read by ldmatrix and V transposed (ldmatrix .trans). Tiles of 32 pages are
// gathered with cp.async, three stages in flight, and a tile converts while the one before it computes. An
// online softmax in base 2 a query; the block writes its chunk's normalized output and log-sum-exp, and `merge`
// combines the chunks. Each sequence's chunk is its length over the splits, read on the device: one launch (and one
// CUDA graph) serves any context.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>

namespace {

constexpr int D = 256;              // head_dim
constexpr int G = 6;                // query heads a KV head
constexpr int WARPS = 6;            // 16 queries each: 96 = 16 rows x 6 heads
constexpr int THREADS = WARPS * 32;
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
};

__global__ void __launch_bounds__(THREADS, 1) split_kernel(const Params p) {
  extern __shared__ __align__(16) uint8_t smem[];
  // raw K and V stages [STAGES][BN][F4ROW], then the FP16 K and V tiles [2][BN][HROW]
  uint8_t* ks = smem;
  uint8_t* vs = ks + STAGES * BN * F4ROW;
  uint8_t* kh = vs + STAGES * BN * F4ROW;
  uint8_t* vh = kh + 2 * BN * HROW;

  const int seq = blockIdx.x, kvh = blockIdx.y, split = blockIdx.z;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int QH = p.kvh_count * G;
  const int q0 = p.qo_indptr[seq], n_rows = p.qo_indptr[seq + 1] - q0;
  const int k0 = p.kv_indptr[seq], kv_len = p.kv_indptr[seq + 1] - k0;
  const int per = (kv_len + p.splits - 1) / p.splits;
  const int chunk = (per + BN - 1) / BN * BN;
  const int lo = split * chunk, hi = min(lo + chunk, kv_len);

  // the warp's queries m = 16 warp + g and + 8: row m / 6, head m % 6
  int row[2], head[2];
  bool live[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int m = warp * 16 + g + 8 * h;
    row[h] = m / G;
    head[h] = kvh * G + m % G;
    live[h] = row[h] < n_rows;
  }
  const bool warp_live = warp * 16 / G < n_rows;  // a warp whose queries are all past the rows only loads

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
  const int sees[2] = {kv_len - n_rows + row[0], kv_len - n_rows + row[1]};

  float acc[D / 8][4];
#pragma unroll
  for (int j = 0; j < D / 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};

  const uint8_t* kbase = p.k + static_cast<int64_t>(kvh) * (D / 2);
  const uint8_t* vbase = p.v + static_cast<int64_t>(kvh) * (D / 2);
  auto load = [&](int stage, int start) {
    // BN rows of 8 chunks of codes and one of scales, for K and V
    for (int i = threadIdx.x; i < BN * 9; i += THREADS) {
      const int r = i / 9, c = i % 9;
      const int tok = start + r;
      const bool ok = tok < hi;
      const int64_t page = ok ? p.kv_indices[k0 + tok] : 0;
      const int64_t src = c < 8 ? page * p.page_stride + c * 16 : page * p.sf_page_stride;
      const uint8_t* kb = c < 8 ? kbase : p.ksf + static_cast<int64_t>(kvh) * (D / 16);
      const uint8_t* vb = c < 8 ? vbase : p.vsf + static_cast<int64_t>(kvh) * (D / 16);
      cp_async16(ks + (stage * BN + r) * F4ROW + c * 16, kb + src, ok);
      cp_async16(vs + (stage * BN + r) * F4ROW + c * 16, vb + src, ok);
    }
  };

  const int tiles = hi > lo ? (hi - lo + BN - 1) / BN : 0;
  // A tile's K and V to FP16 (exact) into tile buffer cb, once for every warp
  auto convert = [&](int stage, int cb) {
    // a block of 16 values (one scale) a step
    for (int i = threadIdx.x; i < BN * (D / 16) * 2; i += THREADS) {
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
  };
  auto compute = [&](int cb, int start) {
    if (warp_live) {
      // S = Q K^T: BN / 8 tiles of 8 pages; B fragments K[page g][16 c + 2 t, + 1] and [+ 8, + 9]
      const uint8_t* kt = kh + cb * BN * HROW;
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
          s[j][e] = (tok < hi && tok <= sees[h] && live[h]) ? x : -INFINITY;
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
          ldsm_x4_trans(b, vh + (cb * BN + prow) * HROW + col * 2);
          mma_f16(acc[2 * n2], pa, b[0], b[1]);
          mma_f16(acc[2 * n2 + 1], pa, b[2], b[3]);
        }
      }
    }
  };
  // The pipeline: STAGES - 1 raw tiles in flight; each iteration converts tile it + 1 while it computes tile it,
  // one barrier an iteration (it publishes tile it's conversion and frees tile it - 1's raw stage and buffer)
#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) {
    if (i < tiles) load(i, lo + i * BN);
    cp_commit();
  }
  cp_wait<STAGES - 2>();
  __syncthreads();
  if (tiles > 0) convert(0, 0);
  for (int it = 0; it < tiles; ++it) {
    cp_wait<STAGES - 3 < 0 ? 0 : STAGES - 3>();  // tile it + 1's raw rows (and tile it's)
    __syncthreads();
    if (it + STAGES - 1 < tiles) load((it + STAGES - 1) % STAGES, lo + (it + STAGES - 1) * BN);
    cp_commit();
    if (it + 1 < tiles) convert((it + 1) % STAGES, (it + 1) & 1);
    compute(it & 1, lo + it * BN);
  }
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

__global__ void merge_kernel(const float* __restrict__ o, const float* __restrict__ lse, __nv_bfloat16* __restrict__ out,
                             int splits, int64_t out_stride, int QH) {
  const int token = blockIdx.x, head = blockIdx.y;
  const int64_t base = static_cast<int64_t>(token) * QH + head;
  float top = -INFINITY;
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
    out[static_cast<int64_t>(token) * out_stride + head * D + d] = __float2bfloat16(den > 0.f ? num / den : 0.f);
  }
}

}  // namespace

void verify_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor ksf, torch::Tensor vsf,
                 torch::Tensor kv_indptr, torch::Tensor kv_indices, torch::Tensor qo_indptr, torch::Tensor o, torch::Tensor lse,
                 torch::Tensor out, double sm_scale, int64_t splits) {
  TORCH_CHECK(q.size(2) == D && k.size(3) == D / 2 && q.size(1) == k.size(2) * G,
              "verify_attn: head_dim 256, 6 query heads a KV head");
  TORCH_CHECK(k.dtype() == torch::kUInt8 && v.dtype() == k.dtype() && k.stride(0) == v.stride(0) && ksf.size(3) == D / 16 &&
              ksf.stride(0) == vsf.stride(0), "verify_attn: an NVFP4 cache and its scales");
  const int n = static_cast<int>(qo_indptr.numel() - 1), kvh = static_cast<int>(k.size(2));
  const Params p{reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(k.data_ptr()),
                 reinterpret_cast<const uint8_t*>(v.data_ptr()), kv_indptr.data_ptr<int32_t>(), kv_indices.data_ptr<int32_t>(),
                 qo_indptr.data_ptr<int32_t>(), o.data_ptr<float>(), lse.data_ptr<float>(), q.stride(0), k.stride(0),
                 reinterpret_cast<const uint8_t*>(ksf.data_ptr()), reinterpret_cast<const uint8_t*>(vsf.data_ptr()), ksf.stride(0),
                 kvh, static_cast<int>(splits), static_cast<float>(sm_scale * 1.4426950408889634)};
  auto stream = at::cuda::getCurrentCUDAStream();
  static bool configured = false;
  if (!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    configured = true;
  }
  split_kernel<<<dim3(n, kvh, static_cast<unsigned>(splits)), THREADS, SMEM, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  merge_kernel<<<dim3(static_cast<unsigned>(q.size(0)), static_cast<unsigned>(q.size(1))), 128, 0, stream>>>(
      o.data_ptr<float>(), lse.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(splits),
      out.stride(0), static_cast<int>(q.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("verify_attn", &verify_attn); }
