// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

// IQ2 layout/codebooks follow ggml-org/llama.cpp
// d327bf1f0780f87b8c01b367b61853b0c291d7c4 (MIT; THIRD_PARTY_NOTICES.md).
// Shared exact two-component FP8 decoding for dense and grouped GEMMs.
namespace ggml {
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

// FP8 codebooks {8,24,44} and {0,1,-1}; the sum is exactly {8,25,43}.
template <bool GLOBAL_LUT = false>
__device__ __forceinline__ uint32_t iq2_selector(uint32_t grid, uint32_t signs, int t, const uint32_t* lut) {
  signs |= (__popc(signs) & 1) << 7;
  const uint32_t mask = (signs >> (4 * (t & 1))) & 15;
  const uint32_t flip = ((mask & 1) << 2) | ((mask & 2) << 5) | ((mask & 4) << 8) | ((mask & 8) << 11);
  const uint32_t entry = GLOBAL_LUT ? __ldg(lut + grid) : lut[grid];
  const uint32_t selector = (entry >> (16 * (t & 1))) & 65535;
  return selector | flip;
}

__device__ __forceinline__ uint2 iq2_components(uint32_t selector) {
  // Indices 0/1/2 select positive magnitudes; 4/5/6 select negative ones.
  return make_uint2(prmt(0x00635c50u, 0x00e3dcd0u, selector), prmt(0x00b83800u, 0x0038b880u, selector));
}

template <bool GLOBAL_LUT = false>
__device__ __forceinline__ uint2 iq2_pair(uint32_t grid, uint32_t signs, int t, const uint32_t* lut) {
  return iq2_components(iq2_selector<GLOBAL_LUT>(grid, signs, t, lut));
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


}  // namespace ggml
