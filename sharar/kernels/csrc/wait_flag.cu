// SPDX-License-Identifier: Apache-2.0
// A graph-capturable device-side wait on a host flag: the kernel spins (with __nanosleep) until the
// pinned int32 flag reaches `value`, or sets `error` and returns after `timeout_ns`. Lets the host launch
// the next decode graph before its inputs (the PLE rows of the just-sampled token) are ready.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void wait_flag_kernel(volatile int* flag, int value, int* error, long long timeout_ns) {
    long long start = clock64();
    // clock64 ticks at the SM clock; treat ~2 GHz as an upper bound so the timeout is never shorter than asked.
    long long limit = timeout_ns * 2;
    while (*flag < value) {
        if (clock64() - start > limit) { *error = 1; return; }
        __nanosleep(2000);
    }
}

void wait_flag(at::Tensor flag, int64_t value, at::Tensor error, int64_t timeout_ns) {
    TORCH_CHECK(flag.is_pinned() && flag.dtype() == at::kInt && flag.numel() == 1, "flag must be a pinned int32 scalar");
    TORCH_CHECK(error.is_cuda() && error.dtype() == at::kInt, "error must be a CUDA int32 tensor");
    int* dev_flag = nullptr;
    C10_CUDA_CHECK(cudaHostGetDevicePointer((void**)&dev_flag, flag.data_ptr(), 0));
    auto stream = at::cuda::getCurrentCUDAStream();
    wait_flag_kernel<<<1, 1, 0, stream>>>(dev_flag, (int)value, error.data_ptr<int>(), timeout_ns);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wait_flag", &wait_flag, "device-side wait on a pinned host flag");
}
