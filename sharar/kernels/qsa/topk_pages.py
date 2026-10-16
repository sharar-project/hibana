# SPDX-License-Identifier: Apache-2.0
"""Single-launch exact FP32 radix top-512 + sorted physical page assembly.

Equal scores choose ascending logical group index, including +/-0. This is
deterministic but not Torch's unspecified tie order. Nonfinite scores are
excluded. Scores must already contain the causal mask. Cache layout unchanged.
"""
from functools import lru_cache
from pathlib import Path

import torch
from .compression import CUDA_FLAGS


@lru_cache(maxsize=None)
def load_topk_pages(build_directory=None):
    import tvm_ffi.cpp
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('this experiment is compiled only for SM121')
    build = Path(build_directory).resolve() if build_directory is not None else None
    if build is not None:
        build.mkdir(parents=True, exist_ok=True)
    module = tvm_ffi.cpp.load('qsa_topk_pages',
        sources=str(Path(__file__).parent/'csrc/topk_pages.cu'),
        extra_cuda_cflags=CUDA_FLAGS, build_directory=str(build) if build is not None else None)
    return TopkPages(module)


class TopkPages:
    def __init__(self, module):
        self.module = module

    def __call__(self, scores, table, base, positions, out=None):
        if (scores.ndim != 2 or not scores.is_cuda or scores.dtype != torch.float32
                or scores.stride(1) != 1 or scores.shape[1] > 8192):
            raise ValueError('scores must be CUDA FP32 [rows,groups<=8192], contiguous columns')
        rows = scores.shape[0]
        for tensor, shape, dtype in ((table, None, torch.int32),
                (base, (rows,), torch.int64), (positions, (rows,), torch.int64)):
            if (tensor.device != scores.device or tensor.ndim != 1 or tensor.dtype != dtype
                    or not tensor.is_contiguous() or (shape is not None and tensor.shape != shape)):
                raise ValueError('table/base/positions must be contiguous CUDA int32/int64/int64 vectors')
        if out is None:
            out = (torch.empty((rows,2051),device=scores.device,dtype=torch.int32),
                   torch.empty(rows,device=scores.device,dtype=torch.int32))
        if len(out) != 2:
            raise ValueError('out must contain pages and counts')
        for tensor, shape in zip(out, ((rows,2051),(rows,))):
            if tensor.shape != shape or tensor.dtype != torch.int32 or tensor.device != scores.device or not tensor.is_contiguous():
                raise ValueError('invalid output buffers')
        # Trusted hot-path inputs: nonnegative positions, finite entries only
        # for complete visible groups, and valid physical page-table entries.
        # No device-to-host validation or allocation in the native launch.
        self.module.topk_pages(scores,table,base,positions,*out)
        return out
