// SPDX-License-Identifier: Apache-2.0
// Adapted from SGLang qsa_index_q_prep_kernel, commit
// d62b9c2480f3b4b2ff16344c90e6a73e4b1d09fa; see adjacent LICENSE.sglang.
// Direct aux page addressing, no persistent position buffer, fixed geometry,
// Torch-compatible RMS reduction and FP32 RoPE without intermediate BF16 rounding.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>

namespace qsa_query {
using bf16 = __nv_bfloat16;
struct alignas(8) Vec4 { bf16 x[4]; };

template<bool Prepare>
__global__ void query_kernel(const bf16* projected, bf16* aux, const int32_t* pages,
    const int64_t* positions, const bf16* weight, const float* cache, bf16* output,
    float eps) {
  const int64_t token = blockIdx.x;
  const int warp = threadIdx.x/32, lane = threadIdx.x%32;
  if (warp == 0) {
    const Vec4 raw = reinterpret_cast<const Vec4*>(projected+token*640+512)[lane];
    reinterpret_cast<Vec4*>(aux+static_cast<int64_t>(pages[token])*256)[lane] = raw;
  }
  if constexpr (Prepare) {
    const Vec4 v = reinterpret_cast<const Vec4*>(projected+token*640+warp*128)[lane];
    float x[4], ss=0.f;
#pragma unroll
    for (int i=0;i<4;++i) { x[i]=__bfloat162float(v.x[i]); ss+=x[i]*x[i]; }
#pragma unroll
    for (int delta=16;delta>0;delta/=2) ss+=__shfl_down_sync(0xffffffff,ss,delta);
    ss=__shfl_sync(0xffffffff,ss,0);
    const float nf=rsqrtf(ss/128.f+eps);
    __shared__ bf16 norm[4][128];
#pragma unroll
    for (int i=0;i<4;++i) {
      const int d=lane*4+i;
      norm[warp][d]=__float2bfloat16_rn((x[i]*nf)*(1.f+__bfloat162float(weight[d])));
    }
    __syncwarp();
    const float* coeff=cache+positions[token]*64;
    Vec4 out;
#pragma unroll
    for (int i=0;i<4;++i) {
      const int d=lane*4+i;
      float value=__bfloat162float(norm[warp][d]);
      if (d<64) {
        const int partner=d<32?d+32:d-32, c=d%32;
        const float a=value*coeff[c], b=__bfloat162float(norm[warp][partner])*coeff[c+32];
        value=d<32?a-b:a+b;
      }
      out.x[i]=__float2bfloat16_rn(value);
    }
    reinterpret_cast<Vec4*>(output+token*512+warp*128)[lane]=out;
  }
}

void prep(tvm::ffi::TensorView projected, tvm::ffi::TensorView aux,
    tvm::ffi::TensorView pages, tvm::ffi::TensorView positions,
    tvm::ffi::TensorView weight, tvm::ffi::TensorView cache,
    tvm::ffi::TensorView output, float eps) {
  const auto dev=aux.device();
  tvm::ffi::CUDADeviceGuard guard(dev.device_id);
  auto stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type,dev.device_id));
  if (!projected.size(0)) return;
  query_kernel<true><<<projected.size(0),128,0,stream>>>(
      static_cast<const bf16*>(projected.data_ptr()),static_cast<bf16*>(aux.data_ptr()),
      static_cast<const int32_t*>(pages.data_ptr()),static_cast<const int64_t*>(positions.data_ptr()),
      static_cast<const bf16*>(weight.data_ptr()),static_cast<const float*>(cache.data_ptr()),
      static_cast<bf16*>(output.data_ptr()),eps);
  const auto error=cudaGetLastError();
  TVM_FFI_ICHECK(error==cudaSuccess)<<cudaGetErrorString(error);
}

void scatter(tvm::ffi::TensorView projected,tvm::ffi::TensorView aux,tvm::ffi::TensorView pages) {
  const auto dev=aux.device();
  tvm::ffi::CUDADeviceGuard guard(dev.device_id);
  auto stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type,dev.device_id));
  if (!projected.size(0)) return;
  query_kernel<false><<<projected.size(0),32,0,stream>>>(
      static_cast<const bf16*>(projected.data_ptr()),static_cast<bf16*>(aux.data_ptr()),
      static_cast<const int32_t*>(pages.data_ptr()),nullptr,nullptr,nullptr,nullptr,0.f);
  const auto error=cudaGetLastError();
  TVM_FFI_ICHECK(error==cudaSuccess)<<cudaGetErrorString(error);
}
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prep,qsa_query::prep);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(scatter,qsa_query::scatter);
