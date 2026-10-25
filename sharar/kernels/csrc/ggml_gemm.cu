// SPDX-License-Identifier: Apache-2.0
// Packed IQ2 FP8 Tensor Core kernels, using W3's warp-MMA arithmetic.
// Q8_0 retains an independent codec specialization for reference bring-up.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

namespace {
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ void mma_e4m3(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
}

}  // namespace

// Native GGML payloads. Layout specifications and IQ2 grids originate from
// ggml-org/llama.cpp d327bf1f0780f87b8c01b367b61853b0c291d7c4 (MIT,
// Copyright (c) 2023-2026 The ggml authors; THIRD_PARTY_NOTICES.md).
// Reuses the W3 FP8 MMA primitive and activation quantizer above. Two E4M3
// components exactly represent every decoded integer, including IQ2's 25/43
// and all signed Q8 values. No weight is rounded to a single FP8 value.
namespace ggml {

// Fragment loads span eight rows per warp. Padding distributes these rows
// over distinct shared-memory banks while keeping async-copy alignment.
constexpr int kActStride = 272;
constexpr int kScaleStride = 12;

// FP8 codebooks {8,24,44} and {0,1,-1}; the sum is exactly {8,25,43}.
__device__ __forceinline__ uint2 iq2_pair(uint32_t grid, uint32_t signs, int t, const uint32_t* lut) {
  signs |= (__popc(signs) & 1) << 7;
  const uint32_t mask = (signs >> (4 * (t & 1))) & 15;
  const uint32_t flip = ((mask & 1) << 7) | ((mask & 2) << 14) | ((mask & 4) << 21) | ((mask & 8) << 28);
  const uint32_t selector = (lut[grid] >> (16 * (t & 1))) & 65535;
  return make_uint2(prmt(0x00635c50u, 0u, selector) ^ flip, prmt(0x00b83800u, 0u, selector) ^ flip);
}

__device__ __forceinline__ void mma_residual(float d[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int Q>
__device__ __forceinline__ const uint8_t* payload(const uint8_t* values, int n, int kg, int K) {
  constexpr int bytes = Q == 2 ? 32 : 64;
  const int kb = Q == 2 ? kg : kg / 8, blocks = Q == 2 ? K / 32 : K / 256;
  return values + ((static_cast<size_t>(n / 8) * blocks + kb) * 8 + n % 8) * bytes;
}

template <int Q>
__device__ __forceinline__ uint2 decode4(const uint8_t* row, int kg, int half, int t, const uint32_t* lut) {
  if constexpr (Q == 2) {
    const uint32_t codes = *reinterpret_cast<const uint32_t*>(row + half * 16 + t * 4);
    uint32_t high = 0, low = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t pair = lut[(codes >> (8 * j)) & 255];
      high |= (pair & 255) << (8 * j);
      low |= ((pair >> 8) & 255) << (8 * j);
    }
    return make_uint2(high, low);
  } else {
    const int j = 2 * half + t / 2;
    uint32_t grid, signs;
    if constexpr (Q == 0) {
      const uint32_t descriptor = reinterpret_cast<const uint16_t*>(row)[(kg % 8) * 4 + j];
      grid = descriptor & 511;
      signs = descriptor >> 9;
    } else {
      row += (kg % 8) * 8;
      grid = row[j];
      signs = (*reinterpret_cast<const uint32_t*>(row + 4) >> (7 * j)) & 127;
    }
    return iq2_pair(grid, signs, t, lut);
  }
}

template <int Q>
__device__ __forceinline__ float factor(const uint8_t* values, const uint8_t* metadata,
                                       int n, int kg, int half, int N, int K) {
  constexpr int meta = Q == 0 ? 10 : 2;
  const int kb = Q == 2 ? kg : kg / 8;
  const uint8_t* record = metadata + (static_cast<size_t>(kb) * N + n) * meta;
  const float base = __half2float(*reinterpret_cast<const __half*>(record));
  if constexpr (Q == 2) return base;
  else {
    uint32_t scale;
    if constexpr (Q == 0) scale = (record[2 + kg % 8] >> (4 * half)) & 15;
    else scale = *reinterpret_cast<const uint32_t*>(payload<Q>(values, n, kg, K) + (kg % 8) * 8 + 4) >> 28;
    return (base * (static_cast<float>(scale) + 0.5f)) * 0.25f;
  }
}

template <int MT>
__device__ __forceinline__ void stage_activations(uint8_t* acts, float* scales, const uint8_t* x, const float* xs,
                                                 int M, int K, int row_start, int kb) {
#pragma unroll
  for (int vec = threadIdx.x; vec < MT * 16 * 16; vec += 128) {
    const int row = row_start + vec / 16;
    if (row < M) {
      const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(acts + (vec / 16) * kActStride + (vec % 16) * 16));
      const uint8_t* src = x + static_cast<size_t>(row) * K + kb * 256 + (vec % 16) * 16;
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
    }
  }
#pragma unroll
  for (int vec = threadIdx.x; vec < MT * 16 * 2; vec += 128) {
    const int row = row_start + vec / 2;
    if (row < M) {
      const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(scales + (vec / 2) * kScaleStride + (vec % 2) * 4));
      const float* src = xs + static_cast<size_t>(row) * (K / 32) + kb * 8 + (vec % 2) * 4;
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
    }
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <bool SHARED>
__device__ __forceinline__ uint32_t input_word(const uint8_t* pointer) {
  if constexpr (SHARED) return *reinterpret_cast<const uint32_t*>(pointer);
  else return __ldg(reinterpret_cast<const uint32_t*>(pointer));
}

template <int Q, int MT, int NT, bool STAGE_A = false>
__global__ void __launch_bounds__(128) kernel(
    const uint8_t* __restrict__ x, const float* __restrict__ xs, int scale_mode,
    const uint8_t* __restrict__ values, const uint8_t* __restrict__ metadata,
    const uint32_t* __restrict__ table, __nv_bfloat16* __restrict__ out,
    float* __restrict__ partials, int M, int N, int K, int splits) {
  constexpr int words = Q == 0 ? 512 : 256;
  __shared__ uint32_t lut[words];
  __shared__ __align__(16) uint8_t acts[STAGE_A ? 2 * MT * 16 * kActStride : 1];
  __shared__ __align__(16) float act_scales[STAGE_A ? 2 * MT * 16 * kScaleStride : 1];
  for (int i = threadIdx.x; i < words; i += 128) lut[i] = table[i];
  __syncthreads();
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int nbase = (blockIdx.x * 4 + threadIdx.x / 32) * NT * 8;
  const int mbase = blockIdx.y * MT * 16, groups = K / 32;
  float acc[MT][NT][4] = {};
  uint4 encoded[NT];
  float2 bases[NT];
  const int begin = static_cast<int>(blockIdx.z) * groups / splits;
  const int end = (static_cast<int>(blockIdx.z) + 1) * groups / splits;
  const int first = Q == 1 ? begin / 8 * 8 : begin;
  if constexpr (STAGE_A) stage_activations<MT>(acts, act_scales, x, xs, M, K, mbase, first / 8);
  for (int kg_block = first; kg_block < end; kg_block += Q == 1 ? 8 : 1) {
    const int slot = (kg_block / 8 - first / 8) & 1;
    if constexpr (STAGE_A) {
      if (kg_block + 8 < end) stage_activations<MT>(acts + (slot ^ 1) * MT * 16 * kActStride,
          act_scales + (slot ^ 1) * MT * 16 * kScaleStride, x, xs, M, K, mbase, kg_block / 8 + 1);
    }
    if constexpr (Q == 1) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int n = nbase + nt * 8 + g;
          encoded[nt] = n < N ? __ldcs(reinterpret_cast<const uint4*>(payload<Q>(values, n, kg_block, K)) + t)
                             : make_uint4(0, 0, 0, 0);
          const int column = nbase + nt * 8 + 2 * t;
          bases[nt] = column < N ? __half22float2(*reinterpret_cast<const __half2*>(
              metadata + (static_cast<size_t>(kg_block / 8) * N + column) * 2)) : make_float2(0.f, 0.f);
        }
    }
    if constexpr (STAGE_A) {
      if (kg_block + 8 < end) asm volatile("cp.async.wait_group 1;" ::: "memory");
      else asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
    }
#pragma unroll
    for (int step = 0; step < (Q == 1 ? 8 : 1); ++step) {
    const int kg = kg_block + step;
    if (kg < begin || kg >= end) continue;
    uint32_t a[MT][4];
    float xa[MT], xb[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int row = mbase + mt * 16 + g;
      const int stride = STAGE_A ? kActStride : K;
      const uint8_t* source = STAGE_A ? acts + slot * MT * 16 * kActStride : x;
      const size_t offset = static_cast<size_t>(STAGE_A ? mt * 16 + g : row) * stride
                            + (STAGE_A ? step : kg) * 32 + t * 4;
      a[mt][0] = row < M ? input_word<STAGE_A>(source + offset) : 0u;
      a[mt][1] = row + 8 < M ? input_word<STAGE_A>(source + offset + static_cast<size_t>(8) * stride) : 0u;
      a[mt][2] = row < M ? input_word<STAGE_A>(source + offset + 16) : 0u;
      a[mt][3] = row + 8 < M ? input_word<STAGE_A>(source + offset + static_cast<size_t>(8) * stride + 16) : 0u;
      if constexpr (STAGE_A) {
        xa[mt] = row < M ? act_scales[slot * MT * 16 * kScaleStride + (mt * 16 + g) * kScaleStride + step] : 0.f;
        xb[mt] = row + 8 < M ? act_scales[slot * MT * 16 * kScaleStride + (mt * 16 + g + 8) * kScaleStride + step] : 0.f;
      } else {
        xa[mt] = row < M ? xs[scale_mode == 2 ? static_cast<size_t>(row) * groups + kg : scale_mode == 1 ? row : 0] : 0.f;
        xb[mt] = row + 8 < M ? xs[scale_mode == 2 ? static_cast<size_t>(row + 8) * groups + kg : scale_mode == 1 ? row + 8 : 0] : 0.f;
      }
    }
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int source_n = nbase + nt * 8 + g;
      uint2 b0 = make_uint2(0, 0), b1 = make_uint2(0, 0);
      float s0, s1;
      if constexpr (Q == 1) {
        // Four adjacent lanes load a row's native K256 record once. Each
        // K32 step broadcasts its grid/sign words instead of reloading them.
        const int owner = (kg % 8) / 2;
        const uint32_t local_grid = kg & 1 ? encoded[nt].z : encoded[nt].x;
        const uint32_t local_control = kg & 1 ? encoded[nt].w : encoded[nt].y;
        const uint32_t grid = __shfl_sync(0xffffffffu, local_grid, 4 * g + owner);
        const uint32_t control = __shfl_sync(0xffffffffu, local_control, 4 * g + owner);
        b0 = iq2_pair((grid >> (8 * (t / 2))) & 255, (control >> (7 * (t / 2))) & 127, t, lut);
        b1 = iq2_pair((grid >> (8 * (2 + t / 2))) & 255, (control >> (7 * (2 + t / 2))) & 127, t, lut);
        const uint32_t scale0 = __shfl_sync(0xffffffffu, local_control, 4 * (2 * t) + owner) >> 28;
        const uint32_t scale1 = __shfl_sync(0xffffffffu, local_control, 4 * (2 * t + 1) + owner) >> 28;
        s0 = (bases[nt].x * (static_cast<float>(scale0) + 0.5f)) * 0.25f;
        s1 = (bases[nt].y * (static_cast<float>(scale1) + 0.5f)) * 0.25f;
      } else if (source_n < N) {
        const uint8_t* row = payload<Q>(values, source_n, kg, K);
        b0 = decode4<Q>(row, kg, 0, t, lut);
        b1 = decode4<Q>(row, kg, 1, t, lut);
      }
      // XS has independent K16 scales. Mask the other half of the K32 MMA;
      // XXS and Q8 share one scale for the whole K32 dot product.
#pragma unroll
      for (int half = 0; half < (Q == 0 ? 2 : 1); ++half) {
        const uint32_t h0 = Q == 0 && half == 1 ? 0u : b0.x;
        const uint32_t h1 = Q == 0 && half == 0 ? 0u : b1.x;
        const uint32_t l0 = Q == 0 && half == 1 ? 0u : b0.y;
        const uint32_t l1 = Q == 0 && half == 0 ? 0u : b1.y;
        const int column = nbase + nt * 8 + 2 * t;
        if constexpr (Q != 1) {
          s0 = column < N ? factor<Q>(values, metadata, column, kg, half, N, K) : 0.f;
          s1 = column + 1 < N ? factor<Q>(values, metadata, column + 1, kg, half, N, K) : 0.f;
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          float d[4];
          mma_e4m3(d, a[mt], h0, h1);
          mma_residual(d, a[mt], l0, l1);
          acc[mt][nt][0] = fmaf(d[0], s0 * xa[mt], acc[mt][nt][0]);
          acc[mt][nt][1] = fmaf(d[1], s1 * xa[mt], acc[mt][nt][1]);
          acc[mt][nt][2] = fmaf(d[2], s0 * xb[mt], acc[mt][nt][2]);
          acc[mt][nt][3] = fmaf(d[3], s1 * xb[mt], acc[mt][nt][3]);
        }
      }
    }
    }
    if constexpr (STAGE_A) __syncthreads();
  }
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = mbase + mt * 16 + g + half * 8;
      if (row < M)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int column = nbase + nt * 8 + 2 * t;
          if (column < N) {
            const size_t index = static_cast<size_t>(row) * N + column;
            const float v0 = acc[mt][nt][2 * half], v1 = acc[mt][nt][2 * half + 1];
            if (splits > 1)
              *reinterpret_cast<float2*>(partials + static_cast<size_t>(blockIdx.z) * M * N + index) = make_float2(v0, v1);
            else
              *reinterpret_cast<__nv_bfloat162*>(out + index) = __floats2bfloat162_rn(v0, v1);
          }
        }
    }
}

__global__ void reduce(const float* partials, __nv_bfloat16* out, size_t count, int splits) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    float value = 0;
    for (int s = 0; s < splits; ++s) value += partials[static_cast<size_t>(s) * count + i];
    out[i] = __float2bfloat16_rn(value);
  }
}

}  // namespace ggml

void ggml_gemm(torch::Tensor x, torch::Tensor xs, torch::Tensor values, torch::Tensor metadata,
               torch::Tensor lut, torch::Tensor out, torch::Tensor workspace, int64_t quant, int64_t splits,
               int64_t tile_n, int64_t tile_m, bool stage_a) {
  const c10::cuda::CUDAGuard device(x.device());
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kFloat8_e4m3fn && x.is_contiguous(), "GGML input must be contiguous CUDA E4M3");
  const int M = x.size(0), K = x.size(1), N = out.size(1);
  TORCH_CHECK(quant >= 0 && quant <= 2 && M > 0 && N > 0 && N % 8 == 0 && K > 0 && K % (quant == 2 ? 32 : 256) == 0, "invalid GGML geometry/codec");
  TORCH_CHECK((splits == 1 || splits == 2 || splits == 4 || splits == 8) && splits <= K / 32, "invalid GGML split count");
  TORCH_CHECK(tile_n == 32 || tile_n == 128, "GGML N tile must be 32 or 128");
  TORCH_CHECK(tile_m == 16 || tile_m == 32 || tile_m == 64, "GGML M tile must be 16, 32 or 64");
  for (const auto& tensor : {xs, values, metadata, lut, out, workspace})
    TORCH_CHECK(tensor.device() == x.device() && tensor.is_contiguous(), "GGML tensors must be contiguous on the input device");
  TORCH_CHECK(xs.scalar_type() == at::kFloat && (xs.numel() == 1 || xs.numel() == M || xs.numel() == static_cast<int64_t>(M) * K / 32), "invalid GGML activation scales");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.size(0) == M && workspace.scalar_type() == at::kFloat, "invalid GGML output/workspace");
  TORCH_CHECK(values.scalar_type() == at::kByte && metadata.scalar_type() == at::kByte && lut.scalar_type() == at::kUInt32, "invalid GGML storage types");
  TORCH_CHECK(values.numel() == static_cast<int64_t>(N) * K / (quant == 2 ? 1 : 4) &&
              metadata.numel() == static_cast<int64_t>(N) * (K / (quant == 2 ? 32 : 256)) * (quant == 0 ? 10 : 2) &&
              lut.numel() == (quant == 0 ? 512 : 256), "invalid GGML storage sizes");
  TORCH_CHECK(splits == 1 || workspace.numel() >= splits * static_cast<int64_t>(M) * N, "GGML split workspace too small");
  const int scale_mode = xs.numel() == static_cast<int64_t>(M) * K / 32 ? 2 : xs.numel() == M ? 1 : 0;
  TORCH_CHECK(!stage_a || (quant == 1 && scale_mode == 2), "activation staging requires IQ2_XXS and grouped activation scales");
  auto stream = c10::cuda::getCurrentCUDAStream();
#define GGML_LAUNCH(Q, MT, NT, SA) \
  ggml::kernel<Q, MT, NT, SA><<<dim3((N + 32 * NT - 1) / (32 * NT), (M + MT * 16 - 1) / (MT * 16), splits), 128, 0, stream>>>( \
      reinterpret_cast<const uint8_t*>(x.data_ptr()), xs.data_ptr<float>(), scale_mode, values.data_ptr<uint8_t>(), \
      metadata.data_ptr<uint8_t>(), lut.data_ptr<uint32_t>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), \
      workspace.data_ptr<float>(), M, N, K, splits)
#define GGML_TILE(Q, MT) \
  if constexpr (Q == 1) { if (stage_a) { \
    if (tile_n == 32) { GGML_LAUNCH(Q, MT, 1, true); } else { GGML_LAUNCH(Q, MT, 4, true); } \
  } else { \
    if (tile_n == 32) { GGML_LAUNCH(Q, MT, 1, false); } else { GGML_LAUNCH(Q, MT, 4, false); } \
  } } else { \
    if (tile_n == 32) { GGML_LAUNCH(Q, MT, 1, false); } else { GGML_LAUNCH(Q, MT, 4, false); } }
#define GGML_ROWS(Q) \
  if (tile_m == 16) { GGML_TILE(Q, 1); } else if (tile_m == 32) { GGML_TILE(Q, 2); } \
  else { GGML_TILE(Q, 4); }
  if (quant == 0) { GGML_ROWS(0); }
  else if (quant == 1) { GGML_ROWS(1); }
  else { GGML_ROWS(2); }
#undef GGML_ROWS
#undef GGML_TILE
#undef GGML_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (splits > 1) {
    const size_t count = static_cast<size_t>(M) * N;
    ggml::reduce<<<(count + 255) / 256, 256, 0, stream>>>(workspace.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), count, splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggml_gemm", &ggml_gemm, "packed IQ2_XS/IQ2_XXS/Q8_0 FP8-compute GEMM (SM120/121)");
}
