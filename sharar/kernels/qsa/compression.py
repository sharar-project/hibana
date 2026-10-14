# SPDX-License-Identifier: Apache-2.0
"""Experimental CUDA adaptation; no production dispatch or CuTe dependency."""
from functools import lru_cache
import math
from pathlib import Path

import torch

from .maintenance_plan import CompressionPlan


CUDA_FLAGS = ['-std=c++20', '-O3', '-lineinfo', '--fmad=false',
              '-gencode=arch=compute_121,code=sm_121']


@lru_cache(maxsize=None)
def load_compressor(build_directory=None):
    """Compile outside capture; caller chooses a fresh run-local build directory."""
    import tvm_ffi.cpp
    build = Path(build_directory).resolve() if build_directory is not None else None
    if build is not None:
        build.mkdir(parents=True, exist_ok=True)
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('this initial experiment is compiled only for SM121')
    module = tvm_ffi.cpp.load('qsa_compress_fp32_rope',
        sources=str(Path(__file__).parent/'csrc/compression.cu'),
        extra_cuda_cflags=CUDA_FLAGS, build_directory=str(build) if build is not None else None)
    return Compressor(module)


class Compressor:
    def __init__(self, module):
        self.module = module

    def __call__(self, aux, plan: CompressionPlan, weight, cos_sin, eps=1e-6):
        # Check metadata only; values/ranges and output uniqueness are plan preconditions.
        if aux.ndim != 3 or aux.shape[1:] != (2, 128) or aux.dtype != torch.bfloat16 or not aux.is_contiguous():
            raise ValueError('aux must be contiguous BF16 [pages,2,128]')
        if not aux.is_cuda:
            raise ValueError('compression requires CUDA tensors')
        expected = [(plan.members, (plan.groups, 4), torch.int32),
                    (plan.writes, (plan.groups,), torch.int32),
                    (plan.positions, (plan.groups,), torch.int64),
                    (weight, (128,), torch.bfloat16)]
        for tensor, shape, dtype in expected:
            if tensor.device != aux.device or tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous():
                raise ValueError(f'expected contiguous {dtype} {shape} on {aux.device}')
        if cos_sin.ndim != 2 or cos_sin.shape[1] != 64 or cos_sin.dtype != torch.float32 or cos_sin.device != aux.device or not cos_sin.is_contiguous():
            raise ValueError('cos_sin must be contiguous FP32 [positions,64] on the aux device')
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError('eps must be a finite positive host scalar')
        if plan.groups:
            self.module.compress(aux, plan.members, plan.writes, plan.positions, weight, cos_sin, eps)
        return aux
