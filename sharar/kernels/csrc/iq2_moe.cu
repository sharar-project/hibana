// SPDX-License-Identifier: Apache-2.0
// Native IQ2 grouped MoE. Stable GPU routing follows W3's grouped operator;
// Bolt uses a single up projection and ReLU squared instead of SwiGLU.
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <limits>
#include "ggml_codec.cuh"

namespace iq2_moe {
constexpr int kExperts = 512, kThreads = 128, kColumns = 128, kHStride = 136;

struct Params {
  const __nv_bfloat16* x;
  const void* ids;
  const float* weights;
  const uint8_t *uv, *um, *dv, *dm;
  const uint32_t* table;
  int *count, *work, *order;
  uint8_t *xq, *hq;
  float *xs, *hs, *y;
  __nv_bfloat16* out;
  int M, H, I, E, topk, base;
};

template <class ID>
__device__ __forceinline__ int local_expert(const Params& p, int q) {
  const int64_t id = static_cast<const ID*>(p.ids)[q];
  return id >= p.base && id < static_cast<int64_t>(p.base) + p.E ? static_cast<int>(id - p.base) : -1;
}

// Same group arithmetic/conversion as W3 quant_groups, distributed over 512
// threads here. No expanded weight or BF16 intermediate matrix is allocated.
__device__ __forceinline__ void quant_input(const Params& p, int row) {
  const int groups = p.H / 32;
  for (int group = threadIdx.x; group < groups; group += blockDim.x) {
    const __nv_bfloat16* src = p.x + static_cast<size_t>(row) * p.H + group * 32;
    uint4 words[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) words[j] = reinterpret_cast<const uint4*>(src)[j];
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const __nv_bfloat162* halves = reinterpret_cast<const __nv_bfloat162*>(&words[j]);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 v = __bfloat1622float2(halves[k]);
        amax = fmaxf(amax, fmaxf(fabsf(v.x), fabsf(v.y)));
      }
    }
    const float scale = fmaxf(amax * (1.f / 448.f), 1e-12f), inv = 1.f / scale;
    p.xs[static_cast<size_t>(row) * groups + group] = scale;
    uint8_t* dst = p.xq + static_cast<size_t>(row) * p.H + group * 32;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const __nv_bfloat162* halves = reinterpret_cast<const __nv_bfloat162*>(&words[j]);
      uint2 packed;
      uint16_t* bytes = reinterpret_cast<uint16_t*>(&packed);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 v = __bfloat1622float2(halves[k]);
        bytes[k] = __nv_cvt_float2_to_fp8x2(make_float2(v.x * inv, v.y * inv), __NV_SATFINITE, __NV_E4M3);
      }
      reinterpret_cast<uint2*>(dst)[j] = packed;
    }
  }
}

template <class ID, int ROWS>
__global__ void __launch_bounds__(512) prepare(Params p) {
  if (blockIdx.x != 0) { quant_input(p, blockIdx.x - 1); return; }
  __shared__ int counts[kExperts], starts[kExperts];
  const int tid = threadIdx.x, pairs = p.M * p.topk;
  counts[tid] = 0;
  __syncthreads();
  for (int q = tid; q < pairs; q += 512) {
    const int e = local_expert<ID>(p, q);
    if (e >= 0) atomicAdd(counts + e, 1);
  }
  __syncthreads();
  if (tid == 0) {
    int offset = 0, task = 0;
    for (int e = 0; e < p.E; ++e) {
      starts[e] = offset;
      for (int r = 0; r < counts[e]; r += ROWS) {
        p.work[3 * task] = e;
        p.work[3 * task + 1] = offset + r;
        p.work[3 * task + 2] = min(ROWS, counts[e] - r);
        ++task;
      }
      offset += counts[e];
    }
    *p.count = task;
  }
  __syncthreads();
  if (tid < p.E && counts[tid]) {
    int position = starts[tid];
    for (int q = 0; q < pairs; ++q)
      if (local_expert<ID>(p, q) == tid) p.order[position++] = q;
  }
}

__device__ __forceinline__ float relu2(float value) {
  value = __bfloat162float(__float2bfloat16_rn(value));
  value = fmaxf(value, 0.f);
  return __bfloat162float(__float2bfloat16_rn(value * value));
}

template <int Q, int MT, bool UP>
__global__ void __launch_bounds__(128) grouped(Params p) {
  const int task = blockIdx.y;
  if (task >= *p.count) return;
  const int expert = p.work[3 * task], first = p.work[3 * task + 1], rows = p.work[3 * task + 2];
  const int N = UP ? p.I : p.H, K = UP ? p.H : p.I, groups = K / 32;
  constexpr int meta = Q == 0 ? 10 : 2, words = Q == 0 ? 512 : 256;
  const uint8_t* values = (UP ? p.uv : p.dv) + static_cast<size_t>(expert) * N * (K / 256) * 64;
  const uint8_t* metadata = (UP ? p.um : p.dm) + static_cast<size_t>(expert) * N * (K / 256) * meta;
  const uint8_t* input = UP ? p.xq : p.hq;
  const float* scales = UP ? p.xs : p.hs;
  __shared__ uint32_t lut[words];
  __shared__ float intermediate[UP ? MT * 16 * kHStride : 1];
  for (int i = threadIdx.x; i < words; i += kThreads) lut[i] = p.table[i];
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x & 31, g = lane / 4, t = lane & 3;
  const int nbase = blockIdx.x * kColumns + warp * 32;
  int source_row[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = mt * 16 + g + half * 8;
      const int q = row < rows ? p.order[first + row] : -1;
      source_row[mt][half] = q < 0 ? -1 : UP ? q / p.topk : q;
    }
  float acc[MT][4][4] = {};
  for (int kb = 0; kb < K / 256; ++kb) {
    uint4 encoded[4];
    float2 bases[4];
    if constexpr (Q == 1) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int n = nbase + nt * 8;
        encoded[nt] = __ldcs(reinterpret_cast<const uint4*>(ggml::payload<Q>(values, n + g, kb * 8, K)) + t);
        bases[nt] = __half22float2(*reinterpret_cast<const __half2*>(metadata +
            (static_cast<size_t>(kb) * N + n + 2 * t) * 2));
      }
    }
#pragma unroll
    for (int step = 0; step < 8; ++step) {
      const int kg = kb * 8 + step;
      uint32_t a[MT][4];
      float2 xs[MT];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int r0 = source_row[mt][0], r1 = source_row[mt][1];
        const size_t off0 = static_cast<size_t>(r0 < 0 ? 0 : r0) * K + kg * 32 + t * 4;
        const size_t off1 = static_cast<size_t>(r1 < 0 ? 0 : r1) * K + kg * 32 + t * 4;
        a[mt][0] = r0 >= 0 ? __ldg(reinterpret_cast<const uint32_t*>(input + off0)) : 0u;
        a[mt][1] = r1 >= 0 ? __ldg(reinterpret_cast<const uint32_t*>(input + off1)) : 0u;
        a[mt][2] = r0 >= 0 ? __ldg(reinterpret_cast<const uint32_t*>(input + off0 + 16)) : 0u;
        a[mt][3] = r1 >= 0 ? __ldg(reinterpret_cast<const uint32_t*>(input + off1 + 16)) : 0u;
        xs[mt] = make_float2(r0 >= 0 ? scales[static_cast<size_t>(r0) * groups + kg] : 0.f,
                            r1 >= 0 ? scales[static_cast<size_t>(r1) * groups + kg] : 0.f);
      }
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int n = nbase + nt * 8;
        uint2 left, right;
        float s0 = 0, s1 = 0;
        if constexpr (Q == 1) {
          const uint4 record = encoded[nt];
          const uint32_t local_grid = step & 1 ? record.z : record.x;
          const uint32_t local_control = step & 1 ? record.w : record.y;
          const uint32_t grid = __shfl_sync(0xffffffffu, local_grid, 4 * g + step / 2);
          const uint32_t control = __shfl_sync(0xffffffffu, local_control, 4 * g + step / 2);
          left = ggml::iq2_pair((grid >> (8 * (t / 2))) & 255, (control >> (7 * (t / 2))) & 127, t, lut);
          right = ggml::iq2_pair((grid >> (8 * (2 + t / 2))) & 255, (control >> (7 * (2 + t / 2))) & 127, t, lut);
          const uint32_t sub0 = __shfl_sync(0xffffffffu, local_control, 4 * (2 * t) + step / 2) >> 28;
          const uint32_t sub1 = __shfl_sync(0xffffffffu, local_control, 4 * (2 * t + 1) + step / 2) >> 28;
          s0 = (bases[nt].x * (static_cast<float>(sub0) + 0.5f)) * 0.25f;
          s1 = (bases[nt].y * (static_cast<float>(sub1) + 0.5f)) * 0.25f;
        } else {
          const uint8_t* row = ggml::payload<Q>(values, n + g, kg, K);
          left = ggml::decode4<Q>(row, kg, 0, t, lut);
          right = ggml::decode4<Q>(row, kg, 1, t, lut);
        }
#pragma unroll
        for (int half = 0; half < (Q == 0 ? 2 : 1); ++half) {
          if constexpr (Q == 0) {
            s0 = ggml::factor<Q>(values, metadata, n + 2 * t, kg, half, N, K);
            s1 = ggml::factor<Q>(values, metadata, n + 2 * t + 1, kg, half, N, K);
          }
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            float d[4];
            ggml::mma_e4m3(d, a[mt], Q == 0 && half == 1 ? 0u : left.x, Q == 0 && half == 0 ? 0u : right.x);
            ggml::mma_residual(d, a[mt], Q == 0 && half == 1 ? 0u : left.y, Q == 0 && half == 0 ? 0u : right.y);
            acc[mt][nt][0] = fmaf(d[0], s0 * xs[mt].x, acc[mt][nt][0]);
            acc[mt][nt][1] = fmaf(d[1], s1 * xs[mt].x, acc[mt][nt][1]);
            acc[mt][nt][2] = fmaf(d[2], s0 * xs[mt].y, acc[mt][nt][2]);
            acc[mt][nt][3] = fmaf(d[3], s1 * xs[mt].y, acc[mt][nt][3]);
          }
        }
      }
    }
  }
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = mt * 16 + g + half * 8;
      if (row < rows) {
        const int q = p.order[first + row];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int col = warp * 32 + nt * 8 + 2 * t;
          const float2 result = make_float2(acc[mt][nt][2 * half], acc[mt][nt][2 * half + 1]);
          if constexpr (UP) *reinterpret_cast<float2*>(intermediate + row * kHStride + col) = make_float2(relu2(result.x), relu2(result.y));
          else *reinterpret_cast<float2*>(p.y + static_cast<size_t>(q) * N + blockIdx.x * kColumns + col) = result;
        }
      }
    }
  if constexpr (UP) {
    __syncthreads();
    for (int i = warp; i < rows * 4; i += 4) {
      const int row = i / 4, group = i % 4, q = p.order[first + row];
      const float value = intermediate[row * kHStride + group * 32 + lane];
      float amax = fabsf(value);
#pragma unroll
      for (int shift = 16; shift > 0; shift >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, shift));
      const float scale = fmaxf(amax * (1.f / 448.f), 1e-12f), inv = 1.f / scale;
      const int col = blockIdx.x * kColumns + group * 32;
      p.hq[static_cast<size_t>(q) * p.I + col + lane] = __nv_cvt_float_to_fp8(value * inv, __NV_SATFINITE, __NV_E4M3);
      if (lane == 0) p.hs[static_cast<size_t>(q) * (p.I / 32) + col / 32] = scale;
    }
  }
}

template <class ID, bool PAIRS>
__global__ void combine(Params p) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
  if constexpr (PAIRS) {
    if (idx < static_cast<size_t>(p.M) * p.topk * p.H && local_expert<ID>(p, idx / p.H) < 0) p.y[idx] = 0.f;
  } else if (idx < static_cast<size_t>(p.M) * p.H) {
    const int row = idx / p.H, col = idx % p.H;
    float value = 0.f;
    for (int k = 0; k < p.topk; ++k) {
      const int q = row * p.topk + k;
      if (local_expert<ID>(p, q) >= 0) value += p.weights[q] * p.y[static_cast<size_t>(q) * p.H + col];
    }
    p.out[idx] = __float2bfloat16_rn(value);
  }
}
}  // namespace iq2_moe

void iq2_moe_forward(torch::Tensor x, torch::Tensor ids, torch::Tensor weights,
                     std::vector<torch::Tensor> matrices, torch::Tensor lut,
                     std::vector<torch::Tensor> scratch, torch::Tensor out,
                     int64_t quant, int64_t base, int64_t rows, bool pairs) {
  using namespace iq2_moe;
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.is_contiguous(), "IQ2 MoE input must be contiguous CUDA BF16 [M,H]");
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(matrices.size() == 4 && scratch.size() == 8, "invalid IQ2 MoE buffers");
  TORCH_CHECK(quant == 0 || quant == 1, "IQ2 MoE codec must be XS or XXS");
  TORCH_CHECK(rows == 16 || rows == 32 || rows == 64, "IQ2 MoE row tile must be 16/32/64");
  TORCH_CHECK(matrices[0].dim() == 5, "invalid expert payload");
  const int64_t M = x.size(0), H = x.size(1), E = matrices[0].size(0), I = matrices[0].size(1) * 8;
  TORCH_CHECK(M >= 1 && H > 0 && I > 0 && H % 256 == 0 && I % 256 == 0 && E >= 1 && E <= kExperts, "invalid IQ2 MoE geometry");
  TORCH_CHECK(ids.dim() == 2 && ids.size(0) == M && ids.size(1) >= 1 && ids.size(1) <= 16 &&
      (ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong), "ids must be int32/int64 [M,topk], topk<=16");
  const int64_t P = M * ids.size(1), capacity = std::min(P, E) + (P + rows - 1) / rows;
  TORCH_CHECK(P <= std::numeric_limits<int>::max() && H <= std::numeric_limits<int>::max() && I <= std::numeric_limits<int>::max() &&
      base >= 0 && base + E <= std::numeric_limits<int>::max(), "IQ2 MoE geometry exceeds int32 indexing");
  TORCH_CHECK(weights.scalar_type() == at::kFloat && weights.sizes() == ids.sizes(), "routing weights must be FP32 like ids");
  auto check = [&](const torch::Tensor& t, at::ScalarType type, std::vector<int64_t> shape) {
    TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous() && t.scalar_type() == type && t.sizes().vec() == shape,
                "invalid IQ2 MoE tensor shape, dtype or device");
  };
  const int64_t meta = quant == 0 ? 10 : 2;
  check(matrices[0], at::kByte, {E, I / 8, H / 256, 8, 64});
  check(matrices[1], at::kByte, {E, H / 256, I, meta});
  check(matrices[2], at::kByte, {E, H / 8, I / 256, 8, 64});
  check(matrices[3], at::kByte, {E, I / 256, H, meta});
  check(lut, at::kUInt32, {quant == 0 ? 512 : 256});
  check(ids, ids.scalar_type(), {M, ids.size(1)});
  check(weights, at::kFloat, {M, ids.size(1)});
  check(scratch[0], at::kInt, {1}); check(scratch[1], at::kInt, {capacity, 3});
  check(scratch[2], at::kInt, {P}); check(scratch[3], at::kByte, {M, H});
  check(scratch[4], at::kFloat, {M, H / 32}); check(scratch[5], at::kByte, {P, I});
  check(scratch[6], at::kFloat, {P, I / 32}); check(scratch[7], at::kFloat, {P, H});
  if (pairs) { TORCH_CHECK(out.is_alias_of(scratch[7]), "pair output must be its workspace buffer"); }
  else { check(out, at::kBFloat16, {M, H}); }
  Params p{reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), ids.data_ptr(), weights.data_ptr<float>(),
      matrices[0].data_ptr<uint8_t>(), matrices[1].data_ptr<uint8_t>(), matrices[2].data_ptr<uint8_t>(), matrices[3].data_ptr<uint8_t>(), lut.data_ptr<uint32_t>(),
      scratch[0].data_ptr<int>(), scratch[1].data_ptr<int>(), scratch[2].data_ptr<int>(), scratch[3].data_ptr<uint8_t>(), scratch[5].data_ptr<uint8_t>(),
      scratch[4].data_ptr<float>(), scratch[6].data_ptr<float>(), scratch[7].data_ptr<float>(),
      pairs ? nullptr : reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(M), static_cast<int>(H), static_cast<int>(I),
      static_cast<int>(E), static_cast<int>(ids.size(1)), static_cast<int>(base)};
  auto stream = c10::cuda::getCurrentCUDAStream();
#define PREP(R) if (ids.scalar_type() == at::kLong) prepare<int64_t, R><<<M + 1, 512, 0, stream>>>(p); else prepare<int32_t, R><<<M + 1, 512, 0, stream>>>(p)
  if (rows == 16) { PREP(16); } else if (rows == 32) { PREP(32); } else { PREP(64); }
#undef PREP
#define GEMMS(Q, MT) \
  grouped<Q, MT, true><<<dim3(I / 128, capacity), 128, 0, stream>>>(p); \
  grouped<Q, MT, false><<<dim3(H / 128, capacity), 128, 0, stream>>>(p)
#define CODEC(Q) if (rows == 16) { GEMMS(Q, 1); } else if (rows == 32) { GEMMS(Q, 2); } else { GEMMS(Q, 4); }
  if (quant == 0) { CODEC(0); } else { CODEC(1); }
#undef CODEC
#undef GEMMS
  const size_t total = M * H * (pairs ? ids.size(1) : 1);
#define COMBINE(ID) if (pairs) combine<ID, true><<<(total + 255) / 256, 256, 0, stream>>>(p); else combine<ID, false><<<(total + 255) / 256, 256, 0, stream>>>(p)
  if (ids.scalar_type() == at::kLong) { COMBINE(int64_t); } else { COMBINE(int32_t); }
#undef COMBINE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("moe", &iq2_moe_forward, "Grouped IQ2 FP8 MoE with fused ReLU2/quantization"); }
