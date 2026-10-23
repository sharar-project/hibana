"""The 27B's verify attention on SM121 over an NVFP4 KV cache (csrc/verify_attn.cu), compiled on first use."""
import functools
import hashlib
import os
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent / "csrc" / "verify_attn.cu"


@functools.cache
def module():
    from torch.utils.cpp_extension import load
    from sharar.engine.config import current
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", current().cuda_arch)
    revision = hashlib.sha256(_SRC.read_bytes()).hexdigest()[:12]
    return load(name=f"sharar_verify_attn_{revision}", sources=[str(_SRC)], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)


def verify_attention(q, cache, scales, kv_indptr, kv_indices, qo_indptr, sm_scale: float, splits: int = 16, out=None):
    """q [tokens, 24, 256] BF16 (at most 16 rows a sequence, qo_indptr [N + 1]); cache (k, v) [pages, 1, 4, 128] uint8
    NVFP4 codes with `scales` (k, v) [pages, 1, 4, 16] E4M3; kv_indptr [N + 1] / kv_indices each sequence's pages, its
    rows last. Returns [tokens, 24, 256] BF16."""
    k, v = cache
    tokens, qh, d = q.shape
    o = torch.empty((tokens, qh, splits, d), dtype=torch.float32, device=q.device)
    lse = torch.empty((tokens, qh, splits), dtype=torch.float32, device=q.device)
    out = torch.empty_like(q) if out is None else out
    module().verify_attn(q, k, v, *scales, kv_indptr, kv_indices, qo_indptr, o, lse, out, sm_scale, splits)
    return out
