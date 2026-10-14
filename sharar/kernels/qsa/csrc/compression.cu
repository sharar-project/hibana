// SPDX-License-Identifier: Apache-2.0
// Adapted from SGLang qsa_index_k_compress_kernel, commit
// d62b9c2480f3b4b2ff16344c90e6a73e4b1d09fa (Apache-2.0).
// Changes: interleaved aux addressing, explicit group positions, FP32 RoPE,
// fixed BF16/128/ratio4/rotary64 geometry, conventional same-stream launch order.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>

namespace qsa_local {
using bf16 = __nv_bfloat16;
struct alignas(8) Vec4 { bf16 x[4]; };

__global__ __launch_bounds__(128) void compress_kernel(
    bf16* aux,
    const int32_t* members,
    const int32_t* writes,
    const int64_t* positions,
    const bf16* weight, const
    float* cos_sin,
    int64_t groups, float eps) {

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t g = static_cast<int64_t>(blockIdx.x)*4+warp;

  if (g >= groups) return;

  __shared__ bf16 row[4][128];
  float mean[4];

#pragma unroll
  for (int r=0; r<4; ++r) {
    const int64_t page = members[g*4+r];
    const Vec4 v = reinterpret_cast<const Vec4*>(aux+page*256)[lane];

#pragma unroll
    for (int i=0; i<4; ++i) {
      const float value = __bfloat162float(v.x[i]);
      mean[i] = r == 0 ? value : mean[i]+value;
    }
  }

#pragma unroll
  for (int i=0; i<4; ++i) {
    row[warp][lane*4+i] = __float2bfloat16_rn(mean[i]*0.25f);
    mean[i] = __bfloat162float(row[warp][lane*4+i]);
  }

  __syncwarp();
  // Match Torch's contiguous FP32 mean reduction at D=128: four elements
  // per lane followed by a descending shuffle-down tree and lane-zero broadcast.
  float ss = 0.f;

#pragma unroll
  for (int j=0; j<4; ++j) {
    const float f = __bfloat162float(row[warp][lane*4+j]);
    ss += f*f;
  }

#pragma unroll
  for (int delta=16; delta>0; delta/=2) ss += __shfl_down_sync(0xffffffff, ss, delta);
  ss = __shfl_sync(0xffffffff, ss, 0);
  const float nf = rsqrtf(ss/128.f+eps);

#pragma unroll
  for (int i=0; i<4; ++i) {
    const int d = lane*4+i;
    row[warp][d] = __float2bfloat16_rn((mean[i]*nf)*(1.f+__bfloat162float(weight[d])));
  }
  __syncwarp();

  Vec4 out;
  const float* coeff = cos_sin+positions[g]*64;

#pragma unroll
  for (int i=0; i<4; ++i) {
    const int d = lane*4+i;
    float value = __bfloat162float(row[warp][d]);
    if (d < 64) {
      const int p = d < 32 ? d+32 : d-32;
      const int c = d % 32;
      const float a = value*coeff[c];
      const float b = __bfloat162float(row[warp][p])*coeff[c+32];
      value = d < 32 ? a-b : a+b;
    }
    out.x[i] = __float2bfloat16_rn(value);
  }
  const int64_t page = writes[g];
  reinterpret_cast<Vec4*>(aux+page*256+128)[lane] = out;
}

void compress(tvm::ffi::TensorView aux, tvm::ffi::TensorView members,
              tvm::ffi::TensorView writes, tvm::ffi::TensorView positions,
              tvm::ffi::TensorView weight, tvm::ffi::TensorView cos_sin, float eps) {
  // Python wrapper checks shapes/dtypes. This export is internal to the experiment.
  const auto dev = aux.device();
  tvm::ffi::CUDADeviceGuard guard(dev.device_id);
  auto stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  const int64_t groups = members.size(0);
  if (!groups) return;
  compress_kernel<<<(groups+3)/4,128,0,stream>>>(
      static_cast<bf16*>(aux.data_ptr()), static_cast<const int32_t*>(members.data_ptr()),
      static_cast<const int32_t*>(writes.data_ptr()), static_cast<const int64_t*>(positions.data_ptr()),
      static_cast<const bf16*>(weight.data_ptr()), static_cast<const float*>(cos_sin.data_ptr()), groups, eps);
  const auto error = cudaGetLastError();
  TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
}
} // namespace qsa_local
TVM_FFI_DLL_EXPORT_TYPED_FUNC(compress, qsa_local::compress);
