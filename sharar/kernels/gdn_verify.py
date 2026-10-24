"""The Gated DeltaNet verify recurrence and its commit (csrc/gdn_verify.cu), compiled on first use.

Replaces the vendored Triton recurrence (`fla/fused_sigmoid_gating_recurrent.py`) and replay (`fla/accepted_state.py`)
on FP32 state pools: one lane per value row holds its 128 key entries, so a token's two dot products are in-lane FMA
chains instead of cross-lane reductions over a [128, 32] tile. The commit replays the verify's arithmetic exactly.
"""
from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent / "csrc" / "gdn_verify.cu"


@functools.cache
def module():
    from torch.utils.cpp_extension import load
    from sharar.engine.config import current
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", current().cuda_arch)
    revision = hashlib.sha256(_SRC.read_bytes()).hexdigest()[:12]
    return load(name=f"sharar_gdn_verify_{revision}", sources=[str(_SRC)], extra_cuda_cflags=["-O3", "--use_fast_math"],
                verbose=False)


def supports(k: torch.Tensor, v: torch.Tensor, state: torch.Tensor | None) -> bool:
    """Heads of 128 keys and values over an FP32 state pool (the kernel's shapes)."""
    return (state is not None and state.dtype == torch.float32 and k.shape[-1] == 128 and v.shape[-1] == 128
            and k.is_cuda)


def verify(A_log, dt_bias, softplus_beta, softplus_threshold, q, k, v, a, b, initial_state_source, initial_state_indices,
           cu_seqlens=None, intermediate_states_buffer=None, intermediate_state_indices=None, cache_ring=False,
           ring_uses_scratch=False, replayssm_rawk=None, replayssm_rawv=None, replayssm_g=None, replayssm_beta=None,
           **_unused) -> torch.Tensor:
    """`fused_sigmoid_gating_delta_rule_update` for a verify (L2-normalized q and k, the committed state left as it is):
    q, k [1, T, H, 128], v [1, T, HV, 128], a, b [T, HV]; returns [1, T, HV, 128] BF16."""
    n = (len(cu_seqlens) - 1) if cu_seqlens is not None else initial_state_indices.numel()
    ring = dict(ring_k=replayssm_rawk, ring_v=replayssm_rawv, ring_g=replayssm_g, ring_beta=replayssm_beta,
                ring_indices=intermediate_state_indices if ring_uses_scratch else initial_state_indices) if cache_ring else {}
    return module().gdn_verify(q, k, v, a, b, A_log, dt_bias, softplus_beta, softplus_threshold, k.shape[-1] ** -0.5,
                               initial_state_source, initial_state_indices, cu_seqlens, n,
                               intermediate_states_buffer, intermediate_state_indices if intermediate_states_buffer is not None else None,
                               ring.get("ring_k"), ring.get("ring_v"), ring.get("ring_g"), ring.get("ring_beta"),
                               ring.get("ring_indices"))


def commit(state, raw_k, raw_v, gates, betas, slots, counts, out=None) -> None:
    """The first counts[i] ring tokens of row i over slot slots[i]'s state, every layer (in place, or into `out`
    [layers, B, HV, V, K], the test oracle)."""
    module().gdn_commit(state, raw_k, raw_v, gates, betas, slots, counts, state if out is None else out)
