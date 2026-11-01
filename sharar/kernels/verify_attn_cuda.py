"""The 27B's verify attention, and a prefill's attention to its cached prefix, on SM121 over an NVFP4 KV cache
(csrc/verify_attn.cu), compiled on first use."""
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


def verify_splits(batch: int, kv_heads: int, sms: int) -> int:
    """Splits of the context for `verify_attention`: the fewest from 2 to 12 that leave the least of a last wave of
    blocks (one a multiprocessor) idle, as the time a wave over the chunk takes. GB10 (48 SMs), 64K: one request 12, two 6,
    four 3 (48 blocks; 8 splits left 128 blocks, two thirds of a wave short: 2.02 -> 1.66 ms), eight and sixteen 3."""
    pairs = batch * kv_heads
    return min(range(2, 13), key=lambda s: (-(-pairs * s // sms) / s, s))


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


PREFIX_ROWS = 128  # flattened (token, head) rows a prefix block takes (csrc/verify_attn.cu PREFIX_WARPS x 16)


def prefix_plan(new_tokens: list[int], prefix_lens: list[int], kv_indices: torch.Tensor, heads: int, kv_heads: int,
                sms: int) -> tuple:
    """Device buffers for `prefix_attention` of a prefill's sequences (`new_tokens` rows each, `prefix_lens` cached pages
    each, `kv_indices` the prefixes' pages laid end to end): (qo_indptr, kv_indptr, kv_indices, blocks, splits). Splits
    of the prefix keep about two blocks a multiprocessor in flight."""
    group = heads // kv_heads
    qo, kv, blocks = [0], [0], []
    for seq, (n, start) in enumerate(zip(new_tokens, prefix_lens)):
        blocks += [(seq, r) for r in range(0, n * group, PREFIX_ROWS)] if start else []
        qo.append(qo[-1] + n)
        kv.append(kv[-1] + start)
    splits = max(1, min(8, -(-2 * sms // max(1, len(blocks) * kv_heads)), -(-max(prefix_lens) // 256)))
    d = kv_indices.device
    as_dev = lambda x: torch.tensor(x, dtype=torch.int32).to(d, non_blocking=True)  # noqa: E731
    return as_dev(qo), as_dev(kv), kv_indices.to(torch.int32), as_dev(blocks).reshape(-1, 2), splits


def prefix_attention(q, cache, scales, plan, sm_scale: float, new_out=None, new_lse=None):
    """A prefill's rows q [tokens, 24, 256] BF16 against their sequences' cached NVFP4 prefixes (`prefix_plan`), merged
    with their own causal attention (`new_out` [tokens, 24, 256] BF16, `new_lse` [tokens, 24] base-2 log-sum-exp, as
    FlashInfer's ragged prefill returns them) when given. Returns [tokens, 24, 256] BF16."""
    qo_indptr, kv_indptr, kv_indices, blocks, splits = plan
    k, v = cache
    tokens, qh, d = q.shape
    o = torch.empty((tokens, qh, splits, d), dtype=torch.float32, device=q.device)
    lse = torch.full((tokens, qh, splits), float("-inf"), dtype=torch.float32, device=q.device)  # rows without a prefix
    out = torch.empty_like(q)
    module().prefix_attn(q, k, v, *scales, kv_indptr, kv_indices, qo_indptr, blocks, o, lse, out, sm_scale, splits,
                         None if new_out is None else new_out.contiguous(), None if new_lse is None else new_lse.contiguous())
    return out
