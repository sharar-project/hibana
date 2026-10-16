// Three-bit decode GEMM for SM120/SM121: y[m, n] = x_scale[m] * row_scale[n] * sum_k xq[m, k] * s[n, k/32] * C[code[n, k]]
//
// xq is E4M3 (the engine's static-scale projection inputs, or `quant_rows`
// per-token quantization of a BF16 MLP input), C is an eight-entry E4M3
// codebook, s the per-32 E4M3 group scale. Weights are stored as bit planes
// laid out per mma fragment (see sharar/kernels/w3.py `pack`): a lane reads
// three 16-byte words per 32-row block and 128-k chunk and turns four codes
// into the four E4M3 bytes of an m16n8k32 B fragment with three rotates,
// three masked ORs and one PRMT byte lookup. The tensor core does the dot
// against the E4M3 activations; each 32-k group's product is scaled once in
// FP32. Weights stream once per launch with evict-first loads, two chunks
// in flight per warp, so the kernel is bound by the weight stream at 16
// through 64 activation rows. Small N splits K across blocks; the last block
// of a column reduces the FP32 partials. Namespace `moe` runs the same streaming loop over Flash-Next's routed
// experts held in this format (grouped by expert, rows gathered; sharar/kernels/w3_moe.py).

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kWarps = 4;                    // warps per block, 32 output rows each
constexpr int kBlockRows = kWarps * 32;      // 128
constexpr int kChunk = 128;                  // k per chunk
constexpr int kPlaneWords = 3 * 32;          // uint4 per (32-row block, chunk): 3 loads x 32 lanes
constexpr int kScaleWords = 8;               // uint4 per (32-row block, chunk): two per accumulator column pair t

__device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t r) { return __funnelshift_r(x, x, r); }

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

// Rotation for sub-step s (0..7) of a chunk: bit (4j + b + rot(s)) mod 32 of plane word b holds bit b of weight j.
__device__ __forceinline__ constexpr uint32_t rot(int s) { return 16u * (s >> 2) + (s & 3); }

// Four E4M3 weight bytes (one B-fragment register) for sub-step s from the three plane words.
template <int S>
__device__ __forceinline__ uint32_t expand(uint32_t p0, uint32_t p1, uint32_t p2, uint32_t lut_lo, uint32_t lut_hi) {
  const uint32_t q = (rotr(p0, rot(S)) & 0x1111u) | (rotr(p1, rot(S)) & 0x2222u) | (rotr(p2, rot(S)) & 0x4444u);
  return prmt(lut_lo, lut_hi, q);
}

__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)));
}

__device__ __forceinline__ void mma_e4m3(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
}

template <int MT>
struct Tile {
  float acc[MT][4][4];
};

template <int MT, int C, bool XG>
__device__ __forceinline__ void group(float (&acc)[MT][4][4], const uint32_t* __restrict__ wd, const uint8_t* __restrict__ sb,
                                      const uint8_t* __restrict__ x, const float* __restrict__ xg, int M, int K, int k0, int g, int t,
                                      uint32_t lut_lo, uint32_t lut_hi) {
  uint32_t a[MT][4];
  float xa[MT], xb[MT];  // per-group activation scales of rows g and g + 8 (XG), else 1
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int r0 = mt * 16 + g;
    const uint8_t* xr = x + static_cast<size_t>(r0) * K + k0 + C * 32 + t * 4;
    const bool ok0 = r0 < M, ok1 = r0 + 8 < M;
    a[mt][0] = ok0 ? __ldg(reinterpret_cast<const uint32_t*>(xr)) : 0u;
    a[mt][1] = ok1 ? __ldg(reinterpret_cast<const uint32_t*>(xr + static_cast<size_t>(8) * K)) : 0u;
    a[mt][2] = ok0 ? __ldg(reinterpret_cast<const uint32_t*>(xr + 16)) : 0u;
    a[mt][3] = ok1 ? __ldg(reinterpret_cast<const uint32_t*>(xr + static_cast<size_t>(8) * K + 16)) : 0u;
    if (XG) {
      const int G = K / 32, grp = (k0 >> 5) + C;
      xa[mt] = ok0 ? __ldg(xg + static_cast<size_t>(r0) * G + grp) : 0.f;
      xb[mt] = ok1 ? __ldg(xg + static_cast<size_t>(r0 + 8) * G + grp) : 0.f;
    } else {
      xa[mt] = xb[mt] = 1.f;
    }
  }
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    const uint32_t p0 = wd[3 * nt], p1 = wd[3 * nt + 1], p2 = wd[3 * nt + 2];
    const uint32_t b0 = expand<2 * C>(p0, p1, p2, lut_lo, lut_hi);
    const uint32_t b1 = expand<2 * C + 1>(p0, p1, p2, lut_lo, lut_hi);
    // the accumulator's columns are rows 8 nt + 2 t and + 1 of W, each with its own group scale
    const float s0 = e4m3_to_float(sb[(nt * 4 + C) * 2]), s1 = e4m3_to_float(sb[(nt * 4 + C) * 2 + 1]);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      float d[4];
      mma_e4m3(d, a[mt], b0, b1);
      acc[mt][nt][0] = fmaf(d[0], s0 * xa[mt], acc[mt][nt][0]);
      acc[mt][nt][1] = fmaf(d[1], s1 * xa[mt], acc[mt][nt][1]);
      acc[mt][nt][2] = fmaf(d[2], s0 * xb[mt], acc[mt][nt][2]);
      acc[mt][nt][3] = fmaf(d[3], s1 * xb[mt], acc[mt][nt][3]);
    }
  }
}

template <int MT, bool XG, bool ROW_TILES = false>
__global__ void __launch_bounds__(kWarps * 32) w3_gemm_kernel(
    __nv_bfloat16* __restrict__ y,            // [M, N]
    float* __restrict__ ws,                   // [splits, M, N] partials when gridDim.y > 1
    int* __restrict__ counters,               // [N / 128], zero between launches
    const uint8_t* __restrict__ x,            // [M, K] e4m3
    const float* __restrict__ x_scale, const int x_scale_stride,  // XG: [M, K/32] per-group scales; else a scalar or one per row
    const uint4* __restrict__ planes,         // [N/32][K/128][3][32]
    const uint4* __restrict__ scales,         // [N/32][K/128][4 t][2]
    const float* __restrict__ row_scale,      // [N]
    const uint32_t lut_lo, const uint32_t lut_hi,
    const int full_M, const int N, const int K, const int chunks_per_split) {
  // Only the unsplit, statically scaled 64-row QKV projection uses this
  // specialization. Adjacent CTAs share weights through L2 while holding
  // half the row accumulators; each row keeps its original K reduction.
  static_assert(!ROW_TILES || (MT == 2 && !XG));
  const int row_begin = ROW_TILES ? (blockIdx.x % 2) * 32 : 0;
  const int column_block = ROW_TILES ? blockIdx.x / 2 : blockIdx.x;
  const int M = ROW_TILES ? 32 : full_M;
  if (ROW_TILES) {
    y += static_cast<size_t>(row_begin) * N;
    x += static_cast<size_t>(row_begin) * K;
    x_scale += row_begin * x_scale_stride;
  }
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int nb = column_block * kWarps + warp;
  const int KC = K / kChunk;
  const int c_begin = blockIdx.y * chunks_per_split, c_end = c_begin + chunks_per_split;
  const uint4* wp = planes + static_cast<size_t>(nb) * KC * kPlaneWords + lane;
  const uint4* sp = scales + static_cast<size_t>(nb) * KC * kScaleWords + 2 * t;

  float acc[MT][4][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;

  uint4 w[2][3];
  uint4 s[2][2];
  auto load = [&](int c, uint4 (&wd)[3], uint4 (&sd)[2]) {
    const uint4* p = wp + static_cast<size_t>(c) * kPlaneWords;
    wd[0] = __ldcs(p);
    wd[1] = __ldcs(p + 32);
    wd[2] = __ldcs(p + 64);
    sd[0] = __ldg(sp + static_cast<size_t>(c) * kScaleWords);
    sd[1] = __ldg(sp + static_cast<size_t>(c) * kScaleWords + 1);
  };
  load(c_begin, w[0], s[0]);
  if (c_begin + 1 < c_end) load(c_begin + 1, w[1], s[1]);

  for (int c = c_begin; c < c_end; c += 2) {
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      const int cc = c + slot;
      if (cc < c_end) {
        uint4 wl[3] = {w[slot][0], w[slot][1], w[slot][2]};
        const uint4 sl[2] = {s[slot][0], s[slot][1]};
        if (cc + 2 < c_end) load(cc + 2, w[slot], s[slot]);
        const uint32_t* wd = reinterpret_cast<const uint32_t*>(wl);
        const uint8_t* sb = reinterpret_cast<const uint8_t*>(sl);
        const int k0 = cc * kChunk;
        group<MT, 0, XG>(acc, wd, sb, x, x_scale, M, K, k0, g, t, lut_lo, lut_hi);
        group<MT, 1, XG>(acc, wd, sb, x, x_scale, M, K, k0, g, t, lut_lo, lut_hi);
        group<MT, 2, XG>(acc, wd, sb, x, x_scale, M, K, k0, g, t, lut_lo, lut_hi);
        group<MT, 3, XG>(acc, wd, sb, x, x_scale, M, K, k0, g, t, lut_lo, lut_hi);
      }
    }
  }

  // epilogue: scale by the row and activation scales, store bf16 or fp32 partials
  const int n_base = nb * 32;
  float rs[4][2];
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    rs[nt][0] = row_scale[n_base + nt * 8 + 2 * t];
    rs[nt][1] = row_scale[n_base + nt * 8 + 2 * t + 1];
  }
  const bool split = gridDim.y > 1;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int m = mt * 16 + g + half * 8;
      if (m < M) {
        const float xs = XG ? 1.f : x_scale[m * x_scale_stride];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int n = n_base + nt * 8 + 2 * t;
          const float v0 = acc[mt][nt][2 * half] * rs[nt][0] * xs;
          const float v1 = acc[mt][nt][2 * half + 1] * rs[nt][1] * xs;
          if (split) {
            __stcg(reinterpret_cast<float2*>(ws + (static_cast<size_t>(blockIdx.y) * M + m) * N + n), make_float2(v0, v1));
          } else {
            *reinterpret_cast<__nv_bfloat162*>(y + static_cast<size_t>(m) * N + n) = __floats2bfloat162_rn(v0, v1);
          }
        }
      }
    }
  }
  if (!split) return;

  __shared__ int last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(counters + blockIdx.x, 1) == static_cast<int>(gridDim.y) - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  const int col0 = blockIdx.x * kBlockRows;
  for (int idx = threadIdx.x; idx < M * (kBlockRows / 2); idx += kWarps * 32) {
    const int m = idx / (kBlockRows / 2), n = col0 + (idx % (kBlockRows / 2)) * 2;
    float2 sum = make_float2(0.f, 0.f);
    for (int sp2 = 0; sp2 < static_cast<int>(gridDim.y); ++sp2) {
      const float2 v = __ldcg(reinterpret_cast<const float2*>(ws + (static_cast<size_t>(sp2) * M + m) * N + n));
      sum.x += v.x;
      sum.y += v.y;
    }
    *reinterpret_cast<__nv_bfloat162*>(y + static_cast<size_t>(m) * N + n) = __floats2bfloat162_rn(sum.x, sum.y);
  }
  if (threadIdx.x == 0) counters[blockIdx.x] = 0;
}

// Per-row E4M3 quantization of a BF16 [M, K] matrix: one block per row.
__global__ void __launch_bounds__(256) quant_rows_kernel(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ out,
                                                         float* __restrict__ scale, const int K) {
  __shared__ float red[8];
  const int row = blockIdx.x;
  const __nv_bfloat16* xr = x + static_cast<size_t>(row) * K;
  float amax = 0.f;
  for (int k = threadIdx.x * 8; k < K; k += 256 * 8) {
    const uint4 v = *reinterpret_cast<const uint4*>(xr + k);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(h[i]);
      amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < 8 ? red[threadIdx.x] : 0.f;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  const float s = fmaxf(red[0] * (1.f / 448.f), 1e-12f);
  if (threadIdx.x == 0) scale[row] = s;
  const float inv = 1.f / s;
  uint8_t* orow = out + static_cast<size_t>(row) * K;
  for (int k = threadIdx.x * 8; k < K; k += 256 * 8) {
    const uint4 v = *reinterpret_cast<const uint4*>(xr + k);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
    uint2 packed;
    uint8_t* pb = reinterpret_cast<uint8_t*>(&packed);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(h[i]);
      const __nv_fp8x2_storage_t two = __nv_cvt_float2_to_fp8x2(make_float2(f.x * inv, f.y * inv), __NV_SATFINITE, __NV_E4M3);
      pb[2 * i] = two & 0xFF;
      pb[2 * i + 1] = two >> 8;
    }
    *reinterpret_cast<uint2*>(orow + k) = packed;
  }
}

// Per-(row, 32-group) E4M3 quantization of one row of a BF16 [M, K] matrix: one thread per group (256 threads).
__device__ __forceinline__ void quant_groups_row(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ out,
                                                 float* __restrict__ scale, const int K, const int row) {
  const int G = K / 32;
  for (int grp = threadIdx.x; grp < G; grp += 256) {
    const __nv_bfloat16* xg = x + static_cast<size_t>(row) * K + grp * 32;
    uint4 v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = *reinterpret_cast<const uint4*>(xg + i * 8);
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v[i]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(h[j]);
        amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
    }
    const float s = fmaxf(amax * (1.f / 448.f), 1e-12f), inv = 1.f / s;
    scale[static_cast<size_t>(row) * G + grp] = s;
    uint8_t* og = out + static_cast<size_t>(row) * K + grp * 32;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v[i]);
      uint2 packed;
      uint8_t* pb = reinterpret_cast<uint8_t*>(&packed);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(h[j]);
        const __nv_fp8x2_storage_t two = __nv_cvt_float2_to_fp8x2(make_float2(f.x * inv, f.y * inv), __NV_SATFINITE, __NV_E4M3);
        pb[2 * j] = two & 0xFF;
        pb[2 * j + 1] = two >> 8;
      }
      *reinterpret_cast<uint2*>(og + i * 8) = packed;
    }
  }
}

// Per-(row, 32-group) E4M3 quantization of a BF16 [M, K] matrix: one block per row.
__global__ void __launch_bounds__(256) quant_groups_kernel(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ out,
                                                           float* __restrict__ scale, const int K) {
  quant_groups_row(x, out, scale, K, blockIdx.x);
}

template <int MT, bool XG, bool ROW_TILES = false>
void launch(torch::Tensor y, torch::Tensor ws, torch::Tensor counters, torch::Tensor x, torch::Tensor x_scale, int x_scale_stride,
            torch::Tensor planes, torch::Tensor scales, torch::Tensor row_scale, uint32_t lut_lo, uint32_t lut_hi,
            int M, int N, int K, int splits) {
  const dim3 grid((N / kBlockRows) * (ROW_TILES ? 2 : 1), splits);
  w3_gemm_kernel<MT, XG, ROW_TILES><<<grid, kWarps * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), static_cast<float*>(ws.data_ptr()), static_cast<int*>(counters.data_ptr()),
      static_cast<const uint8_t*>(x.data_ptr()), static_cast<const float*>(x_scale.data_ptr()), x_scale_stride,
      static_cast<const uint4*>(planes.data_ptr()), static_cast<const uint4*>(scales.data_ptr()),
      static_cast<const float*>(row_scale.data_ptr()), lut_lo, lut_hi, M, N, K, (K / kChunk) / splits);
}

// ------------------------------------------------------------------ three-bit routed experts (Flash-Next, #73)
//
// The experts in this format (sharar/kernels/w3_moe.py): per expert one W3 matrix for fc1 (1280 x 2560, rows
// reordered so each 32-row block is sixteen up rows then the sixteen matching gate rows) and one for fc2
// (2560 x 640). Any row count, four launches, shapes fixed by M (graph-capturable, no host sync, no atomics in
// any sum):
//   prep     block 0: the plan. The distinct experts of the M x top-k selection in ascending id, and the pairs
//            (row * top-k + k) sorted by (expert, pair), in CSR form: slot d is expert exp[d] with pairs
//            order[start[d] .. start[d + 1]). Up to kOrderInKernel pairs it sorts them itself; beyond that the
//            caller passes the stable sort. Blocks 1..M: x -> E4M3 per (row, 32-group), as `quant_groups`.
//   fc1      grid (1280 / 128, min(pairs, 512)): one block per 128 rows of one selected expert (blocks past the
//            distinct count exit). The W3 GEMM's streaming loop with the expert's rows gathered as the A operand,
//            in passes of at most 64 rows (the weights stream once per pass; wide calls: one block per pass of at
//            most kWide rows). Epilogue: h = bf16(silu(bf16 gate)
//            * bf16 up) for the block's 64 intermediate columns, then E4M3 per (pair, 32-group) into hq / hs.
//   fc2      grid (2560 / 128, min(pairs, 512)): the same loop over each pair's quantized h; FP32 y per pair.
//   combine  out[r] = bf16(sum_k w[r, k] * y[r, k]) in selection order.
namespace moe {

constexpr int kE = 512, kMaxTopk = 10, kH = 2560, kI = 640, kPass = 64, kOrderInKernel = 640;
// Wide calls (`gemm_pass_kernel`) run passes of up to kWide pairs: one expansion of an expert's codes then feeds up to
// seven row tiles, and an expert with 65-112 pairs (about 80 in a 4,096-row call) takes one pass instead of two. The
// same bits as kPass passes. 112 against 96 and 128 on real experts (#144): 128 spills too many registers.
constexpr int kWide = 112;
// plan words: D, the expert of each slot, the slot starts (D + 1 of them)
constexpr int kPlanExp = 1, kPlanStart = kPlanExp + kE, kPlanWords = kPlanStart + kE + 1;
// Wide calls (`Params::work`): plan word kPlanWords holds W, and work[2 w], work[2 w + 1] the distinct slot and the
// first pair of the w-th pass (kWide pairs at most), the passes of a slot adjacent and the slots in ascending order.
constexpr int kPlanWork = kPlanWords;

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

struct Params {
  const __nv_bfloat16* x;
  const int* ids;
  const float* weights;
  const uint4 *planes13, *scales13, *planes2, *scales2;
  const float *row13, *row2;
  int* plan;
  int* work;    // wide calls: one (slot, first pair) per pass, else null
  int* order;   // [pairs] sorted by (expert, pair)
  uint8_t* xq;  // [M, kH] e4m3
  float* xs;    // [M, kH / 32]
  uint8_t* hq;  // [pairs, kI] e4m3
  float* hs;    // [pairs, kI / 32]
  float* y;     // [pairs, kH]
  __nv_bfloat16* out;
  uint32_t lut_lo, lut_hi;
  int M;
  int topk;       // experts per row: ids and weights are [M, topk], 1 <= topk <= kMaxTopk
  uint32_t row_magic;  // pair q's row is q / topk = __umulhi(q, row_magic) (row_of): no runtime division in the passes
  int e_lo, e_n;  // this rank's experts: ids e_lo .. e_lo + e_n - 1, stored from index 0 (expert parallel; else 0, 512)
  bool sort;      // block 0 sorts the pairs (else `order` is given)
};

// Pair q's row. row_magic = 2^32 / topk rounded up (0 for topk 1): exact for q * topk < 2^32, far past any call's pairs.
__device__ __forceinline__ int row_of(const Params& p, int q) {
  return p.row_magic ? static_cast<int>(__umulhi(static_cast<uint32_t>(q), p.row_magic)) : q;
}

// A pair's expert as an index into this rank's experts, or -1 when another rank holds it or the id names none (512).
__device__ __forceinline__ int local_expert(const Params& p, int id) {
  const int e = id - p.e_lo;
  return e >= 0 && e < p.e_n ? e : -1;
}

__global__ void __launch_bounds__(512) prep_kernel(const Params p) {
  if (blockIdx.x > 0) {
    if (threadIdx.x < 256) quant_groups_row(p.x, p.xq, p.xs, kH, blockIdx.x - 1);
    return;
  }
  __shared__ int count[kE], slot[kE], start[kE + 1];
  __shared__ int ids[kOrderInKernel];
  const int tid = threadIdx.x, lane = tid & 31, P = p.M * p.topk;
  count[tid] = 0;  // 512 threads, one expert each
  __syncthreads();
  for (int i = tid; i < P; i += 512) {
    const int e = local_expert(p, p.ids[i]);
    if (p.sort) ids[i] = e;
    if (e >= 0) atomicAdd(count + e, 1);  // integer counts: the result does not depend on the order
  }
  __syncthreads();
  if (tid < 32) {  // distinct experts in ascending id, and their exclusive starts
    int base = 0, total = 0;
    for (int c = 0; c < kE / 32; ++c) {
      const int e = c * 32 + lane, n = count[e];
      const unsigned ballot = __ballot_sync(0xffffffffu, n > 0);
      const int d = base + __popc(ballot & ((1u << lane) - 1u));
      int inclusive = n;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, inclusive, o);
        if (lane >= o) inclusive += v;
      }
      slot[e] = n > 0 ? d : -1;
      if (n > 0) {
        start[d] = total + inclusive - n;
        p.plan[kPlanExp + d] = e;
        p.plan[kPlanStart + d] = total + inclusive - n;
      }
      base += __popc(ballot);
      total += __shfl_sync(0xffffffffu, inclusive, 31);
    }
    if (lane == 0) {
      p.plan[0] = base;
      p.plan[kPlanStart + base] = total;
      start[base] = total;
    }
    if (p.work) {  // one entry per pass of kWide pairs, slot by slot
      __syncwarp();
      int written = 0;
      for (int c = 0; c * 32 < base; ++c) {
        const int d = c * 32 + lane;
        const int passes = d < base ? (start[d + 1] - start[d] + kWide - 1) / kWide : 0;
        int inclusive = passes;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
          const int v = __shfl_up_sync(0xffffffffu, inclusive, o);
          if (lane >= o) inclusive += v;
        }
        for (int q = 0; q < passes; ++q) {
          const int w = written + inclusive - passes + q;
          p.work[2 * w] = d;
          p.work[2 * w + 1] = q * kWide;
        }
        written += __shfl_sync(0xffffffffu, inclusive, 31);
      }
      if (lane == 0) p.plan[kPlanWork] = written;
    }
  }
  __syncthreads();
  if (p.sort && count[tid] > 0) {  // the expert's pairs in pair order
    int pos = start[slot[tid]];
    for (int i = 0; i < P; ++i)
      if (ids[i] == tid) p.order[pos++] = i;
  }
}

// One 32-k group of a chunk against MT tiles of gathered rows: rows g and g + 8 of tile mt are xr[mt][0 / 1]
// (null past the pass's rows) with their per-32 activation scales xg[mt][0 / 1].
template <int MT, int C>
__device__ __forceinline__ void group_rows(float (&acc)[MT][4][4], const uint32_t* __restrict__ wd, const uint8_t* __restrict__ sb,
                                           const uint8_t* const (&xr)[MT][2], const float* const (&xg)[MT][2], int k0, int t,
                                           uint32_t lut_lo, uint32_t lut_hi) {
  const int off = k0 + C * 32 + t * 4;
  uint32_t a[MT][4];
  float xa[MT], xb[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const uint8_t *r0 = xr[mt][0], *r1 = xr[mt][1];
    a[mt][0] = r0 ? __ldg(reinterpret_cast<const uint32_t*>(r0 + off)) : 0u;
    a[mt][1] = r1 ? __ldg(reinterpret_cast<const uint32_t*>(r1 + off)) : 0u;
    a[mt][2] = r0 ? __ldg(reinterpret_cast<const uint32_t*>(r0 + off + 16)) : 0u;
    a[mt][3] = r1 ? __ldg(reinterpret_cast<const uint32_t*>(r1 + off + 16)) : 0u;
    xa[mt] = r0 ? __ldg(xg[mt][0] + (k0 >> 5) + C) : 0.f;
    xb[mt] = r1 ? __ldg(xg[mt][1] + (k0 >> 5) + C) : 0.f;
  }
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    const uint32_t p0 = wd[3 * nt], p1 = wd[3 * nt + 1], p2 = wd[3 * nt + 2];
    const uint32_t b0 = expand<2 * C>(p0, p1, p2, lut_lo, lut_hi);
    const uint32_t b1 = expand<2 * C + 1>(p0, p1, p2, lut_lo, lut_hi);
    const float s0 = e4m3_to_float(sb[(nt * 4 + C) * 2]), s1 = e4m3_to_float(sb[(nt * 4 + C) * 2 + 1]);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      float d[4];
      mma_e4m3(d, a[mt], b0, b1);
      acc[mt][nt][0] = fmaf(d[0], s0 * xa[mt], acc[mt][nt][0]);
      acc[mt][nt][1] = fmaf(d[1], s1 * xa[mt], acc[mt][nt][1]);
      acc[mt][nt][2] = fmaf(d[2], s0 * xb[mt], acc[mt][nt][2]);
      acc[mt][nt][3] = fmaf(d[3], s1 * xb[mt], acc[mt][nt][3]);
    }
  }
}

// Wide calls stage a pass's activations per 128-k chunk in shared memory (the four warps read the same rows),
// double-buffered by cp.async, and read the A fragments with ldmatrix: the same bytes and scales as the
// per-lane loads of `group_rows`, in the same order, so the result is bit-identical.
constexpr int kAStride = kChunk + 16;  // bytes per staged row, padded: ldmatrix's eight rows hit distinct banks
template <int R>  // rows a pass stages
struct alignas(16) Staged {
  uint8_t a[2][R][kAStride];
  float s[2][R][4];
};

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(addr))));
}

template <int MT, int C>
__device__ __forceinline__ void group_staged(float (&acc)[MT][4][4], const uint32_t* __restrict__ wd, const uint8_t* __restrict__ sb,
                                             const uint8_t (*a)[kAStride], const float (*sc)[4], int lane, uint32_t lut_lo, uint32_t lut_hi) {
  const int g = lane >> 2;
  if constexpr (MT > 4) {  // the weight fragments first, then one row tile at a time: fewer live registers, the same bits
    uint32_t b[4][2];
    float s0[4], s1[4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const uint32_t p0 = wd[3 * nt], p1 = wd[3 * nt + 1], p2 = wd[3 * nt + 2];
      b[nt][0] = expand<2 * C>(p0, p1, p2, lut_lo, lut_hi);
      b[nt][1] = expand<2 * C + 1>(p0, p1, p2, lut_lo, lut_hi);
      s0[nt] = e4m3_to_float(sb[(nt * 4 + C) * 2]);
      s1[nt] = e4m3_to_float(sb[(nt * 4 + C) * 2 + 1]);
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      uint32_t af[4];
      ldmatrix_x4(af, &a[mt * 16 + (lane & 7) + 8 * ((lane >> 3) & 1)][C * 32 + 16 * (lane >> 4)]);
      const float xa = sc[mt * 16 + g][C], xb = sc[mt * 16 + g + 8][C];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        float d[4];
        mma_e4m3(d, af, b[nt][0], b[nt][1]);
        acc[mt][nt][0] = fmaf(d[0], s0[nt] * xa, acc[mt][nt][0]);
        acc[mt][nt][1] = fmaf(d[1], s1[nt] * xa, acc[mt][nt][1]);
        acc[mt][nt][2] = fmaf(d[2], s0[nt] * xb, acc[mt][nt][2]);
        acc[mt][nt][3] = fmaf(d[3], s1[nt] * xb, acc[mt][nt][3]);
      }
    }
    return;
  }
  uint32_t af[MT][4];
  float xa[MT], xb[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    // lane i addresses row (i % 8) + 8 ((i / 8) % 2) at byte 16 (i / 16): {row g, row g + 8} x {k bytes 4t, 16 + 4t}
    ldmatrix_x4(af[mt], &a[mt * 16 + (lane & 7) + 8 * ((lane >> 3) & 1)][C * 32 + 16 * (lane >> 4)]);
    xa[mt] = sc[mt * 16 + g][C];
    xb[mt] = sc[mt * 16 + g + 8][C];
  }
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    const uint32_t p0 = wd[3 * nt], p1 = wd[3 * nt + 1], p2 = wd[3 * nt + 2];
    const uint32_t b0 = expand<2 * C>(p0, p1, p2, lut_lo, lut_hi);
    const uint32_t b1 = expand<2 * C + 1>(p0, p1, p2, lut_lo, lut_hi);
    const float s0 = e4m3_to_float(sb[(nt * 4 + C) * 2]), s1 = e4m3_to_float(sb[(nt * 4 + C) * 2 + 1]);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      float d[4];
      mma_e4m3(d, af[mt], b0, b1);
      acc[mt][nt][0] = fmaf(d[0], s0 * xa[mt], acc[mt][nt][0]);
      acc[mt][nt][1] = fmaf(d[1], s1 * xa[mt], acc[mt][nt][1]);
      acc[mt][nt][2] = fmaf(d[2], s0 * xb[mt], acc[mt][nt][2]);
      acc[mt][nt][3] = fmaf(d[3], s1 * xb[mt], acc[mt][nt][3]);
    }
  }
}

// One pass: 128 weight rows (this block's) of expert e against `rows` <= 16 MT of its pairs, list[0 .. rows).
// STAGED: the activations come through shared memory (`Staged`, wide calls), else per-lane loads (decode).
// R: the rows `st` holds (kPass, or kWide in wide calls); hsm may share its memory, which the epilogue writes only
// after every warp is done with the staged activations.
template <bool FC1, int MT, bool STAGED, int R = kPass>
__device__ __forceinline__ void pass(const Params& p, float (*hsm)[kPass + 1], Staged<R>* st, int e, const int* __restrict__ list, int rows) {
  constexpr int N = FC1 ? 2 * kI : kH, K = FC1 ? kH : kI, KC = K / kChunk, XT = STAGED ? 1 : MT;
  static_assert(MT * 16 <= R && R <= kWarps * 32, "a pass stages at most one row per thread");
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, tid = threadIdx.x;
  const uint8_t* xbase = FC1 ? p.xq : p.hq;
  const float* sbase = FC1 ? p.xs : p.hs;
  const uint8_t* xr[XT][2];
  const float* xg[XT][2];
  constexpr uint32_t kNone = 0xFFFFFFFFu;
  uint32_t ar[MT];            // staged: byte offsets in xbase of this thread's pieces, rows tid / 8 + 16 u (kNone past the pass)
  const float* as = nullptr;  // staged: this thread's row of scales (tid < rows)
  if constexpr (STAGED) {
#pragma unroll
    for (int u = 0; u < MT; ++u) {
      ar[u] = kNone;
      const int m = tid / 8 + 16 * u;
      if (m < rows) {
        const int q = __ldg(list + m), r = FC1 ? row_of(p, q) : q;
        ar[u] = static_cast<uint32_t>(r) * K + (tid % 8) * 16;  // < 2^32: rows x 2560 or pairs x 640
      }
    }
    if (tid < rows) {
      const int q = __ldg(list + tid), r = FC1 ? row_of(p, q) : q;
      as = sbase + static_cast<size_t>(r) * (K / 32);
    }
    const int nz = MT * 16 - rows;  // rows of the last tile past the pass: zero bytes and zero scales
    for (int i = tid; i < 2 * nz * 8; i += kWarps * 32) {
      const int b = i / (nz * 8), m = rows + (i / 8) % nz, part = i % 8;
      *reinterpret_cast<uint4*>(&st->a[b][m][part * 16]) = make_uint4(0u, 0u, 0u, 0u);
      if (part == 0) *reinterpret_cast<float4*>(&st->s[b][m][0]) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  } else {
#pragma unroll
    for (int mt = 0; mt < XT; ++mt)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int m = mt * 16 + g + 8 * half;
        xr[mt][half] = nullptr;
        xg[mt][half] = nullptr;
        if (m < rows) {
          const int q = __ldg(list + m), r = FC1 ? row_of(p, q) : q;
          xr[mt][half] = xbase + static_cast<size_t>(r) * K;
          xg[mt][half] = sbase + static_cast<size_t>(r) * (K / 32);
        }
      }
  }
  auto stage = [&](int c, int b) {  // chunk c's activations into buffer b: one commit group, possibly empty
    if (c < KC) {
#pragma unroll
      for (int u = 0; u < MT; ++u)
        if (ar[u] != kNone) cp_async16(&st->a[b][tid / 8 + 16 * u][(tid % 8) * 16], xbase + ar[u] + c * kChunk);
      if (as) cp_async16(&st->s[b][tid][0], as + c * 4);
    }
    cp_async_commit();
  };
  const int nb = blockIdx.x * kWarps + warp;  // 32-row block within the expert
  const size_t block = static_cast<size_t>(e) * (N / 32) + nb;
  const uint4* wp = (FC1 ? p.planes13 : p.planes2) + block * KC * kPlaneWords + lane;
  const uint4* sp = (FC1 ? p.scales13 : p.scales2) + block * KC * kScaleWords + 2 * t;

  float acc[MT][4][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;
  uint4 w[2][3];
  uint4 s[2][2];
  auto load = [&](int c, uint4 (&wd)[3], uint4 (&sd)[2]) {
    const uint4* q = wp + static_cast<size_t>(c) * kPlaneWords;
    wd[0] = __ldcs(q);
    wd[1] = __ldcs(q + 32);
    wd[2] = __ldcs(q + 64);
    sd[0] = __ldcs(sp + static_cast<size_t>(c) * kScaleWords);
    sd[1] = __ldcs(sp + static_cast<size_t>(c) * kScaleWords + 1);
  };
  if constexpr (STAGED && MT > 4) {
    // Wide passes: a chunk's weights one chunk ahead in registers (a chunk takes twice as long as at four tiles, so
    // one ahead covers about the latency two did, in fewer registers); the activations as below.
    load(0, w[0], s[0]);
    stage(0, 0);
    stage(1, 1);
    for (int cc = 0; cc < KC; ++cc) {
      const int slot = cc & 1;
      uint4 wl[3] = {w[0][0], w[0][1], w[0][2]};
      const uint4 sl[2] = {s[0][0], s[0][1]};
      if (cc + 1 < KC) load(cc + 1, w[0], s[0]);
      const uint32_t* wd = reinterpret_cast<const uint32_t*>(wl);
      const uint8_t* sb = reinterpret_cast<const uint8_t*>(sl);
      cp_async_wait1();
      __syncthreads();
      group_staged<MT, 0>(acc, wd, sb, st->a[slot], st->s[slot], lane, p.lut_lo, p.lut_hi);
      group_staged<MT, 1>(acc, wd, sb, st->a[slot], st->s[slot], lane, p.lut_lo, p.lut_hi);
      group_staged<MT, 2>(acc, wd, sb, st->a[slot], st->s[slot], lane, p.lut_lo, p.lut_hi);
      group_staged<MT, 3>(acc, wd, sb, st->a[slot], st->s[slot], lane, p.lut_lo, p.lut_hi);
      __syncthreads();
      stage(cc + 2, slot);
    }
  } else {
  load(0, w[0], s[0]);
  load(1, w[1], s[1]);
  if constexpr (STAGED) {
    stage(0, 0);
    stage(1, 1);
  }
  for (int c = 0; c < KC; c += 2) {
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
      const int cc = c + slot;
      if (cc < KC) {
        uint4 wl[3] = {w[slot][0], w[slot][1], w[slot][2]};
        const uint4 sl[2] = {s[slot][0], s[slot][1]};
        if (cc + 2 < KC) load(cc + 2, w[slot], s[slot]);
        const uint32_t* wd = reinterpret_cast<const uint32_t*>(wl);
        const uint8_t* sb = reinterpret_cast<const uint8_t*>(sl);
        if constexpr (STAGED) {
          cp_async_wait1();  // chunk cc's group is complete (cc + 1's may still be in flight)
          __syncthreads();
          group_staged<MT, 0>(acc, wd, sb, st->a[slot], st->s[slot], lane, p.lut_lo, p.lut_hi);
          group_staged<MT, 1>(acc, wd, sb, st->a[slot], st->s[slot], lane, p.lut_lo, p.lut_hi);
          group_staged<MT, 2>(acc, wd, sb, st->a[slot], st->s[slot], lane, p.lut_lo, p.lut_hi);
          group_staged<MT, 3>(acc, wd, sb, st->a[slot], st->s[slot], lane, p.lut_lo, p.lut_hi);
          __syncthreads();  // every warp is done with buffer `slot` before chunk cc + 2 lands in it
          stage(cc + 2, slot);
        } else {
          const int k0 = cc * kChunk;
          group_rows<MT, 0>(acc, wd, sb, xr, xg, k0, t, p.lut_lo, p.lut_hi);
          group_rows<MT, 1>(acc, wd, sb, xr, xg, k0, t, p.lut_lo, p.lut_hi);
          group_rows<MT, 2>(acc, wd, sb, xr, xg, k0, t, p.lut_lo, p.lut_hi);
          group_rows<MT, 3>(acc, wd, sb, xr, xg, k0, t, p.lut_lo, p.lut_hi);
        }
      }
    }
  }
  }

  const float* rs = (FC1 ? p.row13 : p.row2) + static_cast<size_t>(e) * N + nb * 32;
  if (!FC1) {  // y[pair, n] in FP32
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int m = mt * 16 + g + 8 * half;
        if (m < rows) {
          float* yr = p.y + static_cast<size_t>(__ldg(list + m)) * kH + nb * 32;
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            const int n = nt * 8 + 2 * t;
            *reinterpret_cast<float2*>(yr + n) = make_float2(acc[mt][nt][2 * half] * rs[n], acc[mt][nt][2 * half + 1] * rs[n + 1]);
          }
        }
      }
    return;
  }
  // fc1: columns 0..15 of a warp's block are up rows, 16..31 the matching gate rows; the block holds 64
  // intermediate columns (two 32-groups) of every row, quantized per (pair, group) once all warps wrote them.
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int m = mt * 16 + g + 8 * half;
      if (m < rows) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            const int j = nt * 8 + 2 * t + i;
            const float u = bf16r(acc[mt][nt][2 * half + i] * rs[j]), gv = bf16r(acc[mt][nt + 2][2 * half + i] * rs[16 + j]);
            hsm[m][warp * 16 + j] = bf16r(gv / (1.f + __expf(-gv)) * u);
          }
      }
    }
  __syncthreads();
  for (int i = tid; i < 2 * rows; i += kWarps * 32) {
    const int m = i >> 1, grp = i & 1, q = __ldg(list + m);
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(hsm[m][grp * 32 + i]));
    const float sc = fmaxf(amax * (1.f / 448.f), 1e-12f), inv = 1.f / sc;
    const int col = blockIdx.x * 64 + grp * 32;
    p.hs[static_cast<size_t>(q) * (kI / 32) + col / 32] = sc;
    uint32_t packed[8];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const __nv_fp8x2_storage_t two = __nv_cvt_float2_to_fp8x2(
          make_float2(hsm[m][grp * 32 + 2 * i] * inv, hsm[m][grp * 32 + 2 * i + 1] * inv), __NV_SATFINITE, __NV_E4M3);
      reinterpret_cast<uint16_t*>(packed)[i] = two;
    }
    uint4* dst = reinterpret_cast<uint4*>(p.hq + static_cast<size_t>(q) * kI + col);
    dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
    dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
  }
  __syncthreads();  // hsm is reused by the next pass
}

// fc1 (FC1) or fc2 for 128 weight rows of one selected expert: blockIdx.y the distinct slot, blockIdx.x the row block.
// MTMAX 1 serves calls of at most 16 rows (one pass, and the register budget of one row tile: a wider variant in
// the same kernel costs decode about 15% in occupancy), 4 everything else.
template <bool FC1, int MTMAX>
__global__ void __launch_bounds__(kWarps * 32) gemm_kernel(const Params p) {
  const int d = blockIdx.y;
  if (d >= __ldg(p.plan)) return;
  const int e = __ldg(p.plan + kPlanExp + d), s0 = __ldg(p.plan + kPlanStart + d), n = __ldg(p.plan + kPlanStart + d + 1) - s0;
  __shared__ float hsm[FC1 ? MTMAX * 16 : 1][kPass + 1];
  if constexpr (MTMAX == 1) {
    pass<FC1, 1, false, kPass>(p, hsm, nullptr, e, p.order + s0, n);
  } else {
    __shared__ Staged<kPass> st;
    for (int base = 0; base < n; base += kPass) {
      const int rows = min(kPass, n - base);
      const int* list = p.order + s0 + base;
      if (rows <= 16) pass<FC1, 1, true>(p, hsm, &st, e, list, rows);
      else if (rows <= 32) pass<FC1, 2, true>(p, hsm, &st, e, list, rows);
      else if (rows <= 48) pass<FC1, 3, true>(p, hsm, &st, e, list, rows);
      else pass<FC1, 4, true>(p, hsm, &st, e, list, rows);
    }
  }
}

// Wide calls: one pass of up to kWide pairs per block, blockIdx.y the work entry, blockIdx.x the 128-row weight block.
// An expert's passes are adjacent in launch order and run at the same time, so its weights mostly come from L2 after
// the first pass, where `gemm_kernel` loops over them in one block and streams the weights from DRAM once per pass
// (#144). Per pair the arithmetic is `gemm_kernel`'s, so the bits are the same. The epilogue's tile shares memory with
// the staged activations (two blocks an SM, as with kPass passes). (Eight-warp blocks that run two passes on one set
// of weight loads give the same bits too, but one block per SM loses more to latency than it saves.)
template <bool FC1, int MT>
__device__ __forceinline__ void wide_pass(const Params& p, float (*hsm)[kPass + 1], Staged<kWide>* st, int e, const int* list, int rows) {
  if constexpr (MT * 16 < kWide) {
    if (rows > MT * 16) return wide_pass<FC1, MT + 1>(p, hsm, st, e, list, rows);
  }
  pass<FC1, MT, true, kWide>(p, hsm, st, e, list, rows);
}

template <bool FC1>
__global__ void __launch_bounds__(kWarps * 32) gemm_pass_kernel(const Params p) {
  const int w = blockIdx.y;
  if (w >= __ldg(p.plan + kPlanWork)) return;
  const int d = __ldg(p.work + 2 * w), base = __ldg(p.work + 2 * w + 1);
  const int e = __ldg(p.plan + kPlanExp + d), s0 = __ldg(p.plan + kPlanStart + d), n = __ldg(p.plan + kPlanStart + d + 1) - s0;
  const int rows = min(kWide, n - base);
  const int* list = p.order + s0 + base;
  __shared__ union Smem {
    Staged<kWide> st;
    float hsm[FC1 ? kWide : 1][kPass + 1];
  } sm;
  Staged<kWide>* st = &sm.st;
  wide_pass<FC1, 1>(p, sm.hsm, st, e, list, rows);
}

__global__ void __launch_bounds__(256) combine_kernel(const Params p) {
  const int idx = blockIdx.x * 256 + threadIdx.x;
  if (idx >= p.M * kH) return;
  const int r = idx / kH, col = idx - r * kH;
  float v = 0.f;
  int e[kMaxTopk];  // every id loaded first, so the weight and y loads below overlap
#pragma unroll
  for (int k = 0; k < kMaxTopk; ++k) e[k] = k < p.topk ? local_expert(p, __ldg(p.ids + r * p.topk + k)) : -1;
#pragma unroll
  for (int k = 0; k < kMaxTopk; ++k)
    if (e[k] >= 0)  // another rank's pairs (its partial sum has them) and empty slots add nothing
      v += __ldg(p.weights + r * p.topk + k) * p.y[static_cast<size_t>(r * p.topk + k) * kH + col];
  p.out[idx] = __float2bfloat16_rn(v);
}

}  // namespace moe

}  // namespace

torch::Tensor w3_gemm(torch::Tensor x, torch::Tensor x_scale, torch::Tensor planes, torch::Tensor scales, torch::Tensor row_scale,
                      int64_t lut, int64_t splits, torch::Tensor ws, torch::Tensor counters) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == at::kFloat8_e4m3fn && x.is_contiguous(), "x must be a contiguous e4m3 CUDA matrix");
  TORCH_CHECK(x_scale.dtype() == at::kFloat && x_scale.is_contiguous() && row_scale.dtype() == at::kFloat, "scales must be contiguous fp32");
  TORCH_CHECK(planes.dtype() == at::kUInt32 && scales.dtype() == at::kByte && ws.dtype() == at::kFloat && counters.dtype() == at::kInt);
  const int M = x.size(0), K = x.size(1), N = row_scale.size(0);
  TORCH_CHECK(M >= 1 && M <= 64, "M must be 1..64");
  TORCH_CHECK(N % kBlockRows == 0 && K % kChunk == 0, "N must be a multiple of 128 and K of 128");
  TORCH_CHECK(planes.numel() == static_cast<int64_t>(N / 32) * (K / kChunk) * kPlaneWords * 4, "planes layout mismatch");
  TORCH_CHECK(scales.numel() == static_cast<int64_t>(N / 32) * (K / kChunk) * kScaleWords * 16, "scales layout mismatch");
  TORCH_CHECK(splits >= 1 && (K / kChunk) % splits == 0, "splits must divide the chunk count");
  TORCH_CHECK(splits == 1 || (ws.numel() >= splits * static_cast<int64_t>(M) * N && counters.numel() >= N / kBlockRows), "workspace too small");
  const bool grouped = x_scale.numel() == static_cast<int64_t>(M) * (K / 32) && K > 32;
  const int x_scale_stride = x_scale.numel() == 1 ? 0 : 1;
  TORCH_CHECK(grouped || x_scale_stride == 0 || x_scale.numel() == M, "x_scale must be a scalar, one per row or one per row and 32-group");
  auto y = torch::empty({M, N}, x.options().dtype(at::kBFloat16));
  const uint32_t lut_lo = static_cast<uint32_t>(lut & 0xFFFFFFFFu), lut_hi = static_cast<uint32_t>(static_cast<uint64_t>(lut) >> 32);
  if (M == 64 && N == 14336 && K == 5120 && !grouped && splits == 1) {
    launch<2, false, true>(y, ws, counters, x, x_scale, x_scale_stride, planes, scales, row_scale,
                          lut_lo, lut_hi, M, N, K, 1);
    return y;
  }
  const int MT = (M + 15) / 16;
#define W3_LAUNCH(T) if (MT == T) { if (grouped) launch<T, true>(y, ws, counters, x, x_scale, x_scale_stride, planes, scales, row_scale, lut_lo, lut_hi, M, N, K, static_cast<int>(splits)); \
                                    else launch<T, false>(y, ws, counters, x, x_scale, x_scale_stride, planes, scales, row_scale, lut_lo, lut_hi, M, N, K, static_cast<int>(splits)); }
  W3_LAUNCH(1) W3_LAUNCH(2) W3_LAUNCH(3) W3_LAUNCH(4)
#undef W3_LAUNCH
  return y;
}

std::vector<torch::Tensor> quant_rows(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2, "x must be a contiguous bf16 matrix");
  const int M = x.size(0), K = x.size(1);
  TORCH_CHECK(K % 8 == 0 && M >= 1);
  auto out = torch::empty({M, K}, x.options().dtype(at::kFloat8_e4m3fn));
  auto scale = torch::empty({M}, x.options().dtype(at::kFloat));
  quant_rows_kernel<<<M, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<uint8_t*>(out.data_ptr()), static_cast<float*>(scale.data_ptr()), K);
  return {out, scale};
}

std::vector<torch::Tensor> quant_groups(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2, "x must be a contiguous bf16 matrix");
  const int M = x.size(0), K = x.size(1);
  TORCH_CHECK(K % 32 == 0 && M >= 1);
  auto out = torch::empty({M, K}, x.options().dtype(at::kFloat8_e4m3fn));
  auto scale = torch::empty({M, K / 32}, x.options().dtype(at::kFloat));
  quant_groups_kernel<<<M, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<uint8_t*>(out.data_ptr()), static_cast<float*>(scale.data_ptr()), K);
  return {out, scale};
}

// Routed experts in the three-bit format for x [M, 2560] bf16, ids [M, k] int32, weights [M, k] fp32 (k <= 10) -> bf16
// [M, 2560] (the routed sum; no shared expert). The weights hold experts e_lo .. e_lo + E - 1 (all 512, or one
// rank's range under expert parallelism): pairs routed elsewhere are skipped, and the result is this rank's partial
// sum. `wide_rows`: from this many rows each pass of up to kWide pairs gets its own block (`gemm_pass_kernel`; the same bits).
// `order`: when there are more than 640 pairs, this rank's pairs stably sorted by expert first (int32 [M * k],
// the others after them), else empty (the prep kernel sorts them). An id outside e_lo .. e_lo + E - 1 (another rank's
// expert, or 512 and above: none) adds no pair, whatever its weight.
torch::Tensor w3_moe(torch::Tensor x, torch::Tensor ids, torch::Tensor weights, torch::Tensor order,
                     torch::Tensor planes13, torch::Tensor scales13, torch::Tensor row13,
                     torch::Tensor planes2, torch::Tensor scales2, torch::Tensor row2, int64_t lut, int64_t e_lo,
                     int64_t wide_rows) {
  using namespace moe;
  const int M = x.size(0), topk = ids.dim() == 2 ? ids.size(1) : 0, P = M * topk;
  TORCH_CHECK(M >= 1, "w3_moe: no rows");
  TORCH_CHECK(x.is_cuda() && x.dtype() == at::kBFloat16 && x.is_contiguous() && x.size(1) == kH, "x must be contiguous bf16 [M, 2560]");
  TORCH_CHECK(ids.dim() == 2 && ids.dtype() == at::kInt && ids.is_contiguous() && ids.size(0) == M && topk >= 1 && topk <= kMaxTopk,
              "ids must be int32 [M, k] with 1 <= k <= 10");
  TORCH_CHECK(weights.dtype() == at::kFloat && weights.is_contiguous() && weights.sizes() == ids.sizes(), "weights must be fp32 [M, k] like ids");
  const bool sort = P <= kOrderInKernel;
  TORCH_CHECK(sort ? order.numel() == 0 : (order.dtype() == at::kInt && order.is_contiguous() && order.numel() == P),
              "order: empty up to 640 pairs, else the int32 stable sort of the ids");
  TORCH_CHECK(planes13.dtype() == at::kUInt32 && planes2.dtype() == at::kUInt32 && scales13.dtype() == at::kByte && scales2.dtype() == at::kByte);
  const int64_t E = planes13.size(0);
  TORCH_CHECK(E >= 1 && e_lo >= 0 && e_lo + E <= kE, "at most 512 experts");
  TORCH_CHECK(planes13.numel() == E * 2 * kI * kH * 3 / 32 && planes2.numel() == E * kH * kI * 3 / 32, "expert planes");
  TORCH_CHECK(scales13.numel() == E * 2 * kI * kH / 32 && scales2.numel() == E * kH * kI / 32, "expert scales");
  TORCH_CHECK(row13.numel() == E * 2 * kI && row2.numel() == E * kH, "row scales");
  auto i32 = x.options().dtype(at::kInt), u8 = x.options().dtype(at::kByte), f32 = x.options().dtype(at::kFloat);
  auto plan = torch::empty({kPlanWords + 1}, i32);
  const int slots = std::min<int64_t>(P, E);
  const bool wide = M >= wide_rows;
  const int work_cap = slots + (P + kWide - 1) / kWide;  // sum over slots of ceil(n / kWide) <= slots + P / kWide
  auto work = wide ? torch::empty({2 * work_cap}, i32) : torch::Tensor();
  auto sorted = sort ? torch::empty({P}, i32) : order;
  auto xq = torch::empty({M, kH}, u8);
  auto xs = torch::empty({M, kH / 32}, f32);
  auto hq = torch::empty({P, kI}, u8);
  auto hs = torch::empty({P, kI / 32}, f32);
  auto y = torch::empty({P, kH}, f32);
  auto out = torch::empty({M, kH}, x.options());
  Params p{(const __nv_bfloat16*)x.data_ptr(), (const int*)ids.data_ptr(), (const float*)weights.data_ptr(),
           (const uint4*)planes13.data_ptr(), (const uint4*)scales13.data_ptr(), (const uint4*)planes2.data_ptr(), (const uint4*)scales2.data_ptr(),
           (const float*)row13.data_ptr(), (const float*)row2.data_ptr(), (int*)plan.data_ptr(),
           wide ? (int*)work.data_ptr() : nullptr, (int*)sorted.data_ptr(),
           (uint8_t*)xq.data_ptr(), (float*)xs.data_ptr(), (uint8_t*)hq.data_ptr(), (float*)hs.data_ptr(), (float*)y.data_ptr(),
           (__nv_bfloat16*)out.data_ptr(),
           static_cast<uint32_t>(lut & 0xFFFFFFFFu), static_cast<uint32_t>(static_cast<uint64_t>(lut) >> 32), M, topk,
           topk == 1 ? 0u : static_cast<uint32_t>(0xFFFFFFFFu / topk + 1), static_cast<int>(e_lo), static_cast<int>(E), sort};
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  prep_kernel<<<M + 1, 512, 0, stream>>>(p);
  if (M <= 16) {
    gemm_kernel<true, 1><<<dim3(2 * kI / kBlockRows, slots), kWarps * 32, 0, stream>>>(p);
    gemm_kernel<false, 1><<<dim3(kH / kBlockRows, slots), kWarps * 32, 0, stream>>>(p);
  } else if (wide) {
    gemm_pass_kernel<true><<<dim3(2 * kI / kBlockRows, work_cap), kWarps * 32, 0, stream>>>(p);
    gemm_pass_kernel<false><<<dim3(kH / kBlockRows, work_cap), kWarps * 32, 0, stream>>>(p);
  } else {
    gemm_kernel<true, 4><<<dim3(2 * kI / kBlockRows, slots), kWarps * 32, 0, stream>>>(p);
    gemm_kernel<false, 4><<<dim3(kH / kBlockRows, slots), kWarps * 32, 0, stream>>>(p);
  }
  combine_kernel<<<(M * kH + 255) / 256, 256, 0, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w3_moe", &w3_moe, "routed experts in the 3-bit format, any row count (SM120/121)");
  m.def("quant_groups", &quant_groups, "per-(row, 32-group) e4m3 quantization of a bf16 matrix");
  m.def("w3_gemm", &w3_gemm, "3-bit weight, e4m3 activation tensor-core GEMM for M <= 64 (SM120/121)");
  m.def("quant_rows", &quant_rows, "per-row e4m3 quantization of a bf16 matrix");
}
