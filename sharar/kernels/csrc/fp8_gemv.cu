// Small-M per-tensor-scale FP8 GEMV for SM120/SM121 decode.
//
// Derived from SGLang's csrc/gemm/sm120_fp8_gemv.cuh (Apache-2.0, Copyright
// 2025-2026 SGLang Team), extended from M=1 to M<=4 activation rows so batched
// decode streams the weights once for the whole batch.
//
//   y[m, n] = (sum_k x[m, k] * w[n, k]) * alpha
//
// One warp computes kRows consecutive weight rows. The M activation rows are
// staged in shared memory; weights stream once with evict-first 16-byte loads;
// the reduction is a register + warp-shuffle tree.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>

namespace {

constexpr uint32_t kVec = 16;  // fp8 values per 16-byte load
constexpr uint32_t kMaxM = 4;
constexpr uint32_t kMaxK = 8192;  // kMaxM * kMaxK bytes of static shared memory

__device__ __forceinline__ float dot16(const float4 wv, const float4 xv) {
  const __nv_fp8x2_e4m3* w2 = reinterpret_cast<const __nv_fp8x2_e4m3*>(&wv);
  const __nv_fp8x2_e4m3* x2 = reinterpret_cast<const __nv_fp8x2_e4m3*>(&xv);
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float2 w01 = static_cast<float2>(w2[i]);
    const float2 x01 = static_cast<float2>(x2[i]);
    acc = fmaf(w01.x, x01.x, acc);
    acc = fmaf(w01.y, x01.y, acc);
  }
  return acc;
}

template <uint32_t kRows, uint32_t kNumWarps, uint32_t M>
__global__ void __launch_bounds__(kNumWarps * 32) fp8_gemv_kernel(
    __nv_bfloat16* __restrict__ out,
    const uint8_t* __restrict__ x,
    const uint8_t* __restrict__ w,
    const float* __restrict__ alpha,
    const uint32_t N,
    const uint32_t K) {
  __shared__ __align__(16) uint8_t sx[kMaxM * kMaxK];

  const uint32_t tid = threadIdx.x;
  for (uint32_t i = tid * kVec; i < M * K; i += kNumWarps * 32 * kVec) {
    *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(x + i);
  }
  __syncthreads();

  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;
  const uint32_t r0 = (blockIdx.x * kNumWarps + warp) * kRows;
  if (r0 >= N) return;

  float acc[kRows][M];
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
    for (uint32_t m = 0; m < M; ++m) acc[r][m] = 0.0f;
  }

  for (uint32_t k = lane * kVec; k < K; k += 32 * kVec) {
    float4 xv[M];
#pragma unroll
    for (uint32_t m = 0; m < M; ++m) {
      xv[m] = *reinterpret_cast<const float4*>(sx + m * K + k);
    }
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (r0 + r < N) {
        const float4 wv = __ldcs(reinterpret_cast<const float4*>(w + static_cast<size_t>(r0 + r) * K + k));
#pragma unroll
        for (uint32_t m = 0; m < M; ++m) acc[r][m] += dot16(wv, xv[m]);
      }
    }
  }

#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
    for (uint32_t m = 0; m < M; ++m) {
#pragma unroll
      for (uint32_t off = 16; off > 0; off >>= 1) {
        acc[r][m] += __shfl_down_sync(0xffffffff, acc[r][m], off);
      }
    }
  }
  if (lane == 0) {
    const float a = *alpha;
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (r0 + r < N) {
#pragma unroll
        for (uint32_t m = 0; m < M; ++m) {
          out[static_cast<size_t>(m) * N + r0 + r] = __float2bfloat16(acc[r][m] * a);
        }
      }
    }
  }
}

template <uint32_t kRows, uint32_t kNumWarps, uint32_t M>
void launch(torch::Tensor out, torch::Tensor x, torch::Tensor w, torch::Tensor alpha, uint32_t N, uint32_t K) {
  constexpr uint32_t rows_per_block = kRows * kNumWarps;
  const uint32_t blocks = (N + rows_per_block - 1) / rows_per_block;
  fp8_gemv_kernel<kRows, kNumWarps, M><<<blocks, kNumWarps * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<const uint8_t*>(x.data_ptr()),
      static_cast<const uint8_t*>(w.data_ptr()),
      static_cast<const float*>(alpha.data_ptr()),
      N, K);
}

}  // namespace

torch::Tensor fp8_gemv_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor alpha) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && alpha.is_cuda());
  TORCH_CHECK(x.dtype() == at::kFloat8_e4m3fn && w.dtype() == at::kFloat8_e4m3fn);
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && alpha.dtype() == at::kFloat);
  const uint32_t M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M >= 1 && M <= kMaxM, "M must be 1..4");
  TORCH_CHECK(K % (32 * kVec) == 0 && K <= kMaxK, "K must be a multiple of 512 and <= 8192");
  TORCH_CHECK(w.size(1) == K);
  auto out = torch::empty({M, N}, x.options().dtype(at::kBFloat16));
  const bool wide = N >= 8192;
#define DISPATCH_M(MM)                                                     \
  if (M == MM) {                                                           \
    if (wide) launch<2, 8, MM>(out, x, w, alpha, N, K);                    \
    else launch<1, 8, MM>(out, x, w, alpha, N, K);                         \
  }
  DISPATCH_M(1) DISPATCH_M(2) DISPATCH_M(3) DISPATCH_M(4)
#undef DISPATCH_M
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_gemv", &fp8_gemv_cuda, "small-M FP8 GEMV (SM120/121)");
}
