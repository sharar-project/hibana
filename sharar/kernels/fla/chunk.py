# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch

from sharar.kernels.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sharar.kernels.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from sharar.kernels.fla.chunk_o import chunk_fwd_o
from sharar.kernels.fla.cumsum import chunk_local_cumsum
from sharar.kernels.fla.index import (
    prepare_chunk_indices,
)
from sharar.kernels.fla.l2norm import l2norm_fwd
from sharar.kernels.fla.utils import (
    autocast_custom_fwd,
    input_guard,
)


CHUNK_SIZE = 64


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
):
    g = chunk_local_cumsum(
        g, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )

    # fused kkt + solve_tril + recompute_w_u
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return o, h


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        initial_state_indices: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
    ):
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
        o, h = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        )
        return o.to(q.dtype), h


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
):
    r"""Chunked (64-token) gated delta rule prefill over `N` packed sequences.

    Args:
        q, k (torch.Tensor):
            `[1, T, H, K]` BF16 queries and keys (`B` must be 1 with `cu_seqlens`).
        v (torch.Tensor):
            `[1, T, HV, V]` values; `HV` is a multiple of `H`.
        g (torch.Tensor):
            `[1, T, HV]` FP32 forget gate in log space.
        beta (torch.Tensor):
            `[1, T, HV]` betas.
        scale (Optional[float]):
            Query scale. Default: `1 / sqrt(K)`.
        initial_state (torch.Tensor):
            FP32 state pool `[slots, HV, V, K]`. Each sequence starts from
            `initial_state[initial_state_indices[i]]` and its final state is written
            back there in place; rows with index `-1` are neither read nor written.
        initial_state_indices (torch.Tensor):
            `[N]` int32 pool slot per sequence.
        cu_seqlens (torch.LongTensor):
            `[N + 1]` cumulative sequence lengths.
        use_qk_l2norm_in_kernel (bool):
            L2-normalize q and k before the recurrence.
        chunk_indices, chunk_offsets (Optional[torch.Tensor]):
            Prepared 64-token chunk metadata for these exact sequences. The
            caller owns its lifetime. Omit both for the standalone kernel path.

    Returns:
        (o, None, h):
            `o` is the `[1, T, HV, V]` output in q's dtype; `h` is the
            `[1, NT, HV, V, K]` state entering each chunk (k's dtype), chunks laid out per
            sequence (`NT` = total chunks over all sequences).
    """
    assert q.dtype == k.dtype == v.dtype
    assert (
        q.dtype != torch.float32
    ), "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H]."

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if (
            initial_state_indices is not None
            and initial_state_indices.shape[0] != len(cu_seqlens) - 1
        ):
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state_indices.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, h = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        initial_state_indices,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        chunk_indices,
        chunk_offsets,
    )
    return o, None, h
