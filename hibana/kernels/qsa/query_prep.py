# SPDX-License-Identifier: Apache-2.0
"""QSA index-query prep and raw scatter for the optimized prefill; caller owns output and cache."""
from functools import lru_cache
import math
from pathlib import Path

import torch
from .compression import CUDA_FLAGS


@lru_cache(maxsize=None)
def load_query_prep(build_directory=None):
    import tvm_ffi.cpp
    build = Path(build_directory).resolve() if build_directory is not None else None
    if build is not None:
        build.mkdir(parents=True, exist_ok=True)
    if torch.cuda.get_device_capability() != (12,1):
        raise RuntimeError('QSA query prep is compiled only for SM121')
    module = tvm_ffi.cpp.load('qsa_query_fp32_rope',
        sources=str(Path(__file__).parent/'csrc/query_prep.cu'),
        extra_cuda_cflags=CUDA_FLAGS,build_directory=str(build) if build is not None else None)
    return QueryPrep(module)


class QueryPrep:
    def __init__(self,module): self.module=module

    @staticmethod
    def _check(tensor,shape,dtype,device):
        if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device or not tensor.is_contiguous():
            raise ValueError(f'expected contiguous {dtype} {shape} on {device}')

    def _inputs(self,projected,aux,pages):
        if not aux.is_cuda or aux.ndim != 3 or aux.shape[1:] != (2,128) or aux.dtype != torch.bfloat16 or not aux.is_contiguous():
            raise ValueError('aux must be contiguous CUDA BF16 [pages,2,128]')
        if projected.ndim != 2: raise ValueError('projected must have two dimensions')
        rows=projected.shape[0]
        self._check(projected,(rows,640),torch.bfloat16,aux.device)
        self._check(pages,(rows,),torch.int32,aux.device)
        return rows

    def scatter(self,projected,aux,pages):
        """Dense path: skip query normalization; requires unique valid pages."""
        rows=self._inputs(projected,aux,pages)
        if rows: self.module.scatter(projected,aux,pages)
        return aux

    def __call__(self,projected,aux,pages,positions,weight,cos_sin,output,eps=1e-6):
        """Fixed geometry. Inputs/output must not alias; values/ranges trusted."""
        rows=self._inputs(projected,aux,pages)
        self._check(positions,(rows,),torch.int64,aux.device)
        self._check(weight,(128,),torch.bfloat16,aux.device)
        self._check(output,(rows,4,128),torch.bfloat16,aux.device)
        if cos_sin.ndim != 2 or cos_sin.shape[1] != 64:
            raise ValueError('cos_sin must have shape [positions,64]')
        self._check(cos_sin,cos_sin.shape,torch.float32,aux.device)
        if not math.isfinite(eps) or eps<=0: raise ValueError('eps must be finite and positive')
        if rows: self.module.prep(projected,aux,pages,positions,weight,cos_sin,output,eps)
        return output
