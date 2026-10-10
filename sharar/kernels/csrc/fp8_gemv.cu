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

// Per-tensor dynamic E4M3 quantization of BF16 activations with the arithmetic of
// kernels/flashnext_fused.py:quant_fp8_small (and draft_fp8.quant_dynamic): scale = max(amax * (1/448), 1e-12),
// q = clamp(x / scale, +-448) divided with round-to-nearest. Every block quantizes the whole (small) activation
// into its shared memory, so the GEMV needs no quantization launch of its own; the quantized values and the
// accumulation are the unfused pair's, bit for bit.
template <uint32_t kNumWarps>
__device__ __forceinline__ float block_amax(const __nv_bfloat16* __restrict__ x, uint32_t n) {
  __shared__ float red[kNumWarps];
  float m = 0.0f;
  for (uint32_t i = threadIdx.x * 8; i < n; i += kNumWarps * 32 * 8) {
    const uint4 v = *reinterpret_cast<const uint4*>(x + i);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(h[j]);
      m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
  }
#pragma unroll
  for (uint32_t off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = m;
  __syncthreads();
  m = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < kNumWarps; ++i) m = fmaxf(m, red[i]);
  return m;
}

__device__ __forceinline__ uint8_t quant_e4m3(float x, float scale) {
  const float q = fminf(fmaxf(__fdiv_rn(x, scale), -448.0f), 448.0f);
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3));
}

template <uint32_t kRows, uint32_t kNumWarps, uint32_t M, bool kQuant>
__global__ void __launch_bounds__(kNumWarps * 32) fp8_gemv_kernel(
    __nv_bfloat16* __restrict__ out,
    const uint8_t* __restrict__ x,
    const uint8_t* __restrict__ w,
    const float* __restrict__ alpha,
    const uint32_t N,
    const uint32_t K,
    float* __restrict__ scale_out) {
  __shared__ __align__(16) uint8_t sx[kMaxM * kMaxK];
  __shared__ float s_alpha;

  const uint32_t tid = threadIdx.x;
  if constexpr (kQuant) {
    // x is BF16 [M, K] and alpha the weight's scale.
    const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(x);
    const float scale = fmaxf(block_amax<kNumWarps>(xb, M * K) * (1.0f / 448.0f), 1e-12f);
    for (uint32_t i = tid * 8; i < M * K; i += kNumWarps * 32 * 8) {
      const uint4 v = *reinterpret_cast<const uint4*>(xb + i);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
      uint2 packed;
      uint8_t* pb = reinterpret_cast<uint8_t*>(&packed);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(h[j]);
        pb[2 * j] = quant_e4m3(f.x, scale);
        pb[2 * j + 1] = quant_e4m3(f.y, scale);
      }
      *reinterpret_cast<uint2*>(sx + i) = packed;
    }
    if (tid == 0) {
      s_alpha = scale * *alpha;
      if (blockIdx.x == 0 && scale_out != nullptr) scale_out[0] = scale;
    }
  } else {
    for (uint32_t i = tid * kVec; i < M * K; i += kNumWarps * 32 * kVec) {
      *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(x + i);
    }
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
    const float a = kQuant ? s_alpha : *alpha;
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

template <uint32_t kRows, uint32_t kNumWarps, uint32_t M, bool kQuant>
void launch(torch::Tensor out, torch::Tensor x, torch::Tensor w, torch::Tensor alpha, uint32_t N, uint32_t K, float* scale_out) {
  constexpr uint32_t rows_per_block = kRows * kNumWarps;
  const uint32_t blocks = (N + rows_per_block - 1) / rows_per_block;
  fp8_gemv_kernel<kRows, kNumWarps, M, kQuant><<<blocks, kNumWarps * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      static_cast<const uint8_t*>(x.data_ptr()),
      static_cast<const uint8_t*>(w.data_ptr()),
      static_cast<const float*>(alpha.data_ptr()),
      N, K, scale_out);
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
    if (wide) launch<2, 8, MM, false>(out, x, w, alpha, N, K, nullptr);    \
    else launch<1, 8, MM, false>(out, x, w, alpha, N, K, nullptr);         \
  }
  DISPATCH_M(1) DISPATCH_M(2) DISPATCH_M(3) DISPATCH_M(4)
#undef DISPATCH_M
  return out;
}

// x: BF16 [M<=4, K], quantized per tensor inside the kernel; wscale: the weight's [1] FP32 scale. `scale_out`
// ([1] FP32, optional) receives the activation scale.
torch::Tensor fp8_gemv_quant(torch::Tensor x, torch::Tensor w, torch::Tensor wscale, c10::optional<torch::Tensor> scale_out) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && wscale.is_cuda());
  TORCH_CHECK(x.dtype() == at::kBFloat16 && w.dtype() == at::kFloat8_e4m3fn && wscale.dtype() == at::kFloat);
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && wscale.numel() == 1);
  const uint32_t M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(M >= 1 && M <= kMaxM, "M must be 1..4");
  TORCH_CHECK(K % (32 * kVec) == 0 && K <= kMaxK, "K must be a multiple of 512 and <= 8192");
  TORCH_CHECK(w.size(1) == K);
  float* sp = nullptr;
  if (scale_out.has_value()) {
    TORCH_CHECK(scale_out->is_cuda() && scale_out->dtype() == at::kFloat && scale_out->numel() >= 1);
    sp = scale_out->data_ptr<float>();
  }
  auto out = torch::empty({M, N}, x.options().dtype(at::kBFloat16));
  const bool wide = N >= 8192;
#define DISPATCH_M(MM)                                                     \
  if (M == MM) {                                                           \
    if (wide) launch<2, 8, MM, true>(out, x, w, wscale, N, K, sp);         \
    else launch<1, 8, MM, true>(out, x, w, wscale, N, K, sp);              \
  }
  DISPATCH_M(1) DISPATCH_M(2) DISPATCH_M(3) DISPATCH_M(4)
#undef DISPATCH_M
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_gemv", &fp8_gemv_cuda, "small-M FP8 GEMV (SM120/121)");
  m.def("fp8_gemv_quant", &fp8_gemv_quant, "small-M FP8 GEMV of BF16 activations quantized per tensor in the kernel",
        pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("wscale"), pybind11::arg("scale_out") = pybind11::none());
}
