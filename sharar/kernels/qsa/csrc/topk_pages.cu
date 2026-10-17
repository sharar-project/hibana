// SPDX-License-Identifier: Apache-2.0
// QSA selector: cache a score row, radix-select FP32 top-512, sort only
// the winners, and emit physical pages. No upstream kernel code is copied.
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>

namespace qsa_local {
constexpr int THREADS = 256;
constexpr int TOP = 512;
constexpr int WIDTH = 2051;

__device__ uint32_t score_key(float x) {
  // Finite only. Normalize signed zero so ties use logical group order.
  if (!isfinite(x)) return 0;
  uint32_t bits = __float_as_uint(x == 0.f ? 0.f : x);
  return bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

__global__ __launch_bounds__(THREADS) void topk_pages_kernel(
    const float* scores, int64_t stride, int cols, const int32_t* table,
    const int64_t* base, const int64_t* positions, int32_t* pages, int32_t* counts) {
  extern __shared__ uint32_t keys[];
  __shared__ uint32_t hist[256];
  __shared__ uint64_t winners[TOP];
  __shared__ int valid_count, target, remaining, cursor, ties_seen;
  __shared__ uint32_t prefix, mask, threshold;
  __shared__ int threshold_ties, warp_prefix[8];
  const int t = threadIdx.x, lane = t % 32, warp = t / 32;
  const int64_t row = blockIdx.x;
  if (t == 0) { valid_count = 0; cursor = 0; ties_seen = 0; }
  for (int i=t; i<TOP; i+=THREADS) winners[i]=0;
  for (int i=t; i<WIDTH; i+=THREADS) pages[row*WIDTH+i]=-1;
  __syncthreads();
  int mine=0;
  for (int g=t; g<cols; g+=THREADS) {
    keys[g]=score_key(scores[row*stride+g]);
    mine += keys[g] != 0;
  }
  atomicAdd(&valid_count,mine);
  __syncthreads();
  if (t == 0) {
    target = min(valid_count,TOP);
    remaining=target; prefix=0; mask=0; threshold=0; threshold_ties=0;
  }
  __syncthreads();
  if (valid_count > TOP) {
    // Refine only the bucket containing the Kth element. Four passes retain
    // all FP32 bits; no score quantization or approximate selection.
    for (int shift=24; shift>=0; shift-=8) {
      hist[t]=0;
      __syncthreads();
      for (int g=t; g<cols; g+=THREADS) {
        const uint32_t key=keys[g];
        if (key && (key & mask)==prefix) atomicAdd(hist+((key>>shift)&255),1u);
      }
      __syncthreads();
      if (t==0) {
        int left=remaining;
        for (int bin=255; bin>=0; --bin) {
          if (left > static_cast<int>(hist[bin])) left -= hist[bin];
          else {
            prefix |= static_cast<uint32_t>(bin)<<shift;
            mask |= 255u<<shift;
            remaining=left;
            if (shift==0) threshold_ties=hist[bin];
            break;
          }
        }
      }
      __syncthreads();
    }
    if (t==0) threshold=prefix;
  }
  __syncthreads();
  const bool boundary_tie=valid_count>TOP && threshold_ties>remaining;
  const int greater_count=target-remaining;
  for (int first=0; first<cols; first+=THREADS) {
    const int g=first+t;
    const uint32_t key=g<cols ? keys[g] : 0;
    const bool equal=key && key==threshold;
    if (boundary_tie) {
      // Only boundary ties require deterministic compaction by logical index.
      const uint32_t ballot=__ballot_sync(0xffffffff,equal);
      if (lane==0) warp_prefix[warp]=__popc(ballot);
      __syncthreads();
      if (t==0) {
        int sum=ties_seen;
        for (int w=0; w<8; ++w) { int n=warp_prefix[w]; warp_prefix[w]=sum; sum+=n; }
        ties_seen=sum;
      }
      __syncthreads();
      const int tie_rank=warp_prefix[warp]+__popc(ballot & ((1u<<lane)-1));
      if (equal && tie_rank<remaining)
        winners[greater_count+tie_rank]=(static_cast<uint64_t>(key)<<32)|(0xffffffffu-g);
      __syncthreads();
    }
    const bool take=key && (valid_count<=TOP || key>threshold || (!boundary_tie && equal));
    const uint32_t ballot=__ballot_sync(0xffffffff,take);
    int start=0;
    if (lane==0) start=atomicAdd(&cursor,__popc(ballot));
    start=__shfl_sync(0xffffffff,start,0);
    if (take) {
      const int rank=start+__popc(ballot & ((1u<<lane)-1));
      winners[rank]=(static_cast<uint64_t>(key)<<32)|(0xffffffffu-g);
    }
  }
  __syncthreads();
  // Descending score, ascending logical index. Invalid padding sorts last.
  for (int size=2; size<=TOP; size*=2) {
    for (int distance=size/2; distance; distance/=2) {
      for (int i=t; i<TOP; i+=THREADS) {
        int j=i^distance;
        if (j>i) {
          uint64_t a=winners[i],b=winners[j];
          bool descending=(i&size)==0;
          if (descending ? a<b : a>b) { winners[i]=b; winners[j]=a; }
        }
      }
      __syncthreads();
    }
  }
  const int64_t offset=base[row],position=positions[row];
  for (int i=t; i<target*4; i+=THREADS) {
    const uint32_t group=0xffffffffu-static_cast<uint32_t>(winners[i/4]);
    pages[row*WIDTH+i]=table[offset+static_cast<int64_t>(group)*4+i%4];
  }
  const int tail=(position+1)%4;
  if (t<tail) pages[row*WIDTH+target*4+t]=table[offset+((position+1)/4)*4+t];
  if (t==0) counts[row]=target*4+tail;
}

void topk_pages(tvm::ffi::TensorView scores,tvm::ffi::TensorView table,
    tvm::ffi::TensorView base,tvm::ffi::TensorView positions,
    tvm::ffi::TensorView pages,tvm::ffi::TensorView counts) {
  const auto dev=scores.device();
  tvm::ffi::CUDADeviceGuard guard(dev.device_id);
  auto stream=static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type,dev.device_id));
  if (!scores.size(0)) return;
  topk_pages_kernel<<<scores.size(0),THREADS,scores.size(1)*sizeof(uint32_t),stream>>>(
    static_cast<const float*>(scores.data_ptr()),scores.stride(0),scores.size(1),
    static_cast<const int32_t*>(table.data_ptr()),static_cast<const int64_t*>(base.data_ptr()),
    static_cast<const int64_t*>(positions.data_ptr()),static_cast<int32_t*>(pages.data_ptr()),
    static_cast<int32_t*>(counts.data_ptr()));
  auto error=cudaGetLastError();
  TVM_FFI_ICHECK(error==cudaSuccess)<<cudaGetErrorString(error);
}
} // namespace qsa_local
TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk_pages,qsa_local::topk_pages);
