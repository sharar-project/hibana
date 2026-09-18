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
// of a column reduces the FP32 partials.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

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

// Per-(row, 32-group) E4M3 quantization of a BF16 [M, K] matrix: one thread per group, one block per row.
__global__ void __launch_bounds__(256) quant_groups_kernel(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ out,
                                                           float* __restrict__ scale, const int K) {
  const int row = blockIdx.x, G = K / 32;
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant_groups", &quant_groups, "per-(row, 32-group) e4m3 quantization of a bf16 matrix");
  m.def("w3_gemm", &w3_gemm, "3-bit weight, e4m3 activation tensor-core GEMM for M <= 64 (SM120/121)");
  m.def("quant_rows", &quant_rows, "per-row e4m3 quantization of a bf16 matrix");
}
