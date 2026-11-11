# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# and https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py

from typing import List, Optional

import torch
import triton
import triton.language as tl

from hibana.kernels._compat import is_arch_support_pdl

PAD_SLOT_ID = -1


@triton.jit()
def _causal_conv1d_fwd_kernel(  # continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    initial_states_ptr,  # conv_states_ptr
    cache_indices_ptr,  # conv_state_indices_ptr
    has_initial_states_ptr,
    query_start_loc_ptr,
    o_ptr,  # (dim, cu_seqlen), or the packed q | k | v blocks when KEY_DIM > 0
    # Matrix dimensions
    dim: tl.constexpr,
    seqlen: tl.int32,  # cu_seqlen
    num_cache_lines: tl.constexpr,
    # Strides
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.constexpr,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    KERNEL_WIDTH: tl.constexpr,  # 4: the only width the wrapper accepts
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KEY_DIM: tl.constexpr = 0,  # hibana: > 0 writes the [q | k | v] features as three contiguous [seqlen, width] blocks
):
    total_tokens = seqlen  # hibana: `seqlen` becomes this sequence's length below; the packed blocks span the batch
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1

    # one program handles one chunk in a single sequence
    # rather than mixing sequences - to make updating initial_states across sequences efficiently

    # single-sequence id
    idx_seq = tl.program_id(0)
    chunk_offset = tl.program_id(1)

    # BLOCK_N elements along the feature-dimension (channel)
    idx_feats = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    if segment_len <= 0:
        return

    # base of the sequence
    x_base = (
        x_ptr
        + sequence_start_index.to(tl.int64) * stride_x_token
        + idx_feats * stride_x_dim
    )  # [BLOCK_N,]

    if IS_CONTINUOUS_BATCHING:
        # cache_idx
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        # cache_idx
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return
    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    # Does 2 things:
    # 1. READ prior-block init-state data - [done by every Triton programs]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        # read from conv_states
        load_init_state = False
        if HAS_INITIAL_STATES:
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            # load from conv_states
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
            conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
            conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            # first chunk without prior tokens: they are zeros
            col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # STEP 2:
        # here prepare data for updating conv_state
        if (
            state_len <= seqlen
        ):  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
            # just read from 'x'
            # copy 'x' data to conv_state
            # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
            idx_tokens_last = (seqlen - state_len) + tl.arange(
                0, NP2_STATELEN
            )  # [BLOCK_M]
            x_ptrs = (
                x_ptr
                + (
                    (sequence_start_index + idx_tokens_last).to(tl.int64)
                    * stride_x_token
                )[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )  # [BLOCK_M,BLOCK_N,]
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            conv_states_ptrs_target = (
                conv_states_base[None, :]
                + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            )

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, new_conv_state, mask)

        else:
            if load_init_state:
                # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_state_batch_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (
                    (conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                tl.debug_barrier()  # need this due to the bug in tl.where not enforcing this when data is the result of another tl.load
                new_conv_state = tl.where(
                    mask, conv_state, loaded_x
                )  # BUG in 'tl.where'  which requires a barrier before this
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:  # load_init_state == False
                # update conv_state by shifting left, BUT
                # set cols prior to 'x' as zeros + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        # read prior-token data from `x`
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1).to(tl.int64) * stride_x_token
        mask_w = idx_feats < dim
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset.to(tl.int64) * stride_x_token  # starting of chunk

    # PRE-LOAD WEIGHTS
    mask_w = idx_feats < dim
    w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
    w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
    w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
    w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
    w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
    w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if j == 1:
                matrix_w = w_col1
                matrix_x = col1
            elif j == 2:
                matrix_w = w_col2
                matrix_x = col2
            elif j == 3:
                matrix_w = w_col3
                x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        col0 = col1
        col1 = col2
        col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (
            idx_feats < dim
        )  # token-index  # feature-index
        if KEY_DIM > 0:
            # hibana: q (features [0, K)), k ([K, 2K)) and v ([2K, dim)) each in its own row-major block, so the
            # delta rule reads them without copies; the values are the same.
            section = tl.where(idx_feats < KEY_DIM, 0, tl.where(idx_feats < 2 * KEY_DIM, KEY_DIM, 2 * KEY_DIM))
            width = tl.where(idx_feats < 2 * KEY_DIM, KEY_DIM, dim - 2 * KEY_DIM)
            token = (sequence_start_index + token_offset + idx_token).to(tl.int64)
            o_ptrs = o_ptr + section.to(tl.int64) * total_tokens + token * width + (idx_feats - section)
        else:
            o_ptrs = (
                o_ptr
                + (sequence_start_index + token_offset + idx_token).to(tl.int64)
                * stride_o_token
                + (idx_feats * stride_o_dim)
            )

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    key_dim: int = 0,
):
    """Varlen causal convolution of packed sequences, with conv-state write-back.

    x: (dim, cu_seq_len), the sequences concatenated along tokens.
    weight: (dim, 4)
    conv_states: (slots, dim, >= 3), updated in place: sequence i reads
        conv_states[cache_indices[i]] as its prior tokens when has_initial_state[i],
        and leaves its last three inputs there.
    query_start_loc: (batch + 1,) int32 cumulative sequence lengths, starting at 0.
    seq_lens_cpu: the batch's sequence lengths (host list).
    cache_indices: (batch,) int32 conv-state slot per sequence; pad_slot_id entries are skipped.
    has_initial_state: (batch,) bool.
    activation: None, "silu" or "swish".
    key_dim: hibana: when > 0, the output's storage holds q (features [0, key_dim)), k ([key_dim, 2 key_dim)) and
        v (the rest) as three contiguous [cu_seq_len, width] blocks instead of the (dim, cu_seq_len) layout.

    out: same shape as `x`
    """
    out = torch.empty_like(x)

    dim, cu_seqlen = x.shape
    _, width = weight.shape
    assert width == 4, "the kernel is specialized for a width-4 convolution"
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    # conv_states can hold more slots than the batch; cache_indices map sequences to slots.
    num_cache_lines = conv_states.size(0)
    assert (
        num_cache_lines == conv_states.shape[0]
        and dim == conv_states.shape[1]
        and width - 1 <= conv_states.shape[2]
    )
    stride_istate_seq = conv_states.stride(0)
    stride_istate_dim = conv_states.stride(1)
    stride_istate_token = conv_states.stride(2)
    stride_o_dim = out.stride(0)
    stride_o_token = out.stride(1)

    def grid(META):
        max_seq_len = max(seq_lens_cpu)
        return (
            len(seq_lens_cpu),  # batch_size
            (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_fwd_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        out,
        # Matrix dimensions
        dim,
        cu_seqlen,
        num_cache_lines,
        # stride
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_INITIAL_STATES=has_initial_state is not None,
        IS_CONTINUOUS_BATCHING=cache_indices is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        BLOCK_M=8,
        BLOCK_N=256,
        num_stages=2,
        KEY_DIM=key_dim,
    )
    return out


@triton.jit()
def _causal_conv1d_update_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    conv_state_ptr,
    conv_state_indices_ptr,
    intermediate_conv_window_ptr,
    intermediate_state_indices_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_inter_seq: tl.constexpr,
    stride_inter_step: tl.constexpr,
    stride_inter_dim: tl.constexpr,
    stride_inter_win: tl.constexpr,
    stride_intermediate_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    KERNEL_WIDTH: tl.constexpr,  # 4: the only width the wrapper accepts
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SAVE_INTERMEDIATE: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_CONTINUOUS_BATCHING:
        conv_state_batch_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(tl.int64)
        if SAVE_INTERMEDIATE:
            intermediate_state_batch_coord = tl.load(
                intermediate_state_indices_ptr
                + idx_seq * stride_intermediate_state_indices
            ).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    # STEP 1: READ init_state data
    conv_states_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base
    conv_states_ptrs = prior_tokens  # [BLOCK_N]
    col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
    col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
    col2 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

    # The conv_state updates works in a sliding window manner,
    # at each forward pass, the tokens are shift by seqlen, so we
    # load since idx_tokens + seqlen.
    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + seqlen) * stride_conv_state_tok)[:, None]
    )  # [BLOCK_M, BLOCK_N]
    mask = (
        (conv_state_batch_coord < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # [BLOCK_N]

    x_ptrs = (
        x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    )  # [BLOCK_M, BLOCK_N]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_state_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]
    conv_state_ptrs_target = (
        conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]
    )  # [BLOCK_M, BLOCK_N]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4: PRE-LOAD WEIGHTS for the BLOCK_N features
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
    w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
    w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
    w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
    w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
    w_col3 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token
    for idx_token in tl.static_range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0

        for j in tl.static_range(KERNEL_WIDTH):
            if j == 1:
                matrix_w = w_col1
                matrix_x = col1
            elif j == 2:
                matrix_w = w_col2
                matrix_x = col2
            elif j == 3:
                matrix_w = w_col3
                x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        col0 = col1
        col1 = col2
        col2 = matrix_x

        if SAVE_INTERMEDIATE:
            # Save the window state after consuming this token
            # Layout: [seq(cache line), step, dim, win(K-1)]
            base_ptr = (
                intermediate_conv_window_ptr
                + intermediate_state_batch_coord * stride_inter_seq
                + idx_token * stride_inter_step
                + idx_feats * stride_inter_dim
            )
            tl.store(base_ptr + 0 * stride_inter_win, col0, mask=mask_w)
            tl.store(base_ptr + 1 * stride_inter_win, col1, mask=mask_w)
            tl.store(base_ptr + 2 * stride_inter_win, col2, mask=mask_w)

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (idx_seq) * stride_o_seq
            + idx_token * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    activation: Optional[str] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (slots, dim, state_len), where state_len >= width - 1; updated in place
    weight: (dim, 4)
    activation: None, "silu" or "swish"
    conv_state_indices: (batch,), dtype int32
        The conv_state slot of each row; pad_slot_id rows are skipped.
    intermediate_conv_window, intermediate_state_indices:
        When given, the window after every token is stored at
        intermediate_conv_window[intermediate_state_indices[i], step] ([.., step, dim, width - 1]).
    out: (batch, dim) or (batch, dim, seqlen)
    """
    assert activation in (None, "silu", "swish")
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    assert width == 4, "the kernel is specialized for a width-4 convolution"
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    out = torch.empty_like(x)
    stride_w_dim, stride_w_width = weight.stride()

    stride_x_seq, stride_x_dim, stride_x_token = x.stride()  # X (batch, dim, seqlen)

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    stride_intermediate_state_indices = (
        intermediate_state_indices.stride(0)
        if intermediate_state_indices is not None
        else 0
    )
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    # prepare intermediate buffer strides if provided
    if intermediate_conv_window is not None:
        stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
            intermediate_conv_window.stride(0),
            intermediate_conv_window.stride(1),
            intermediate_conv_window.stride(2),
            intermediate_conv_window.stride(3),
        )
    else:
        stride_inter_seq = stride_inter_step = stride_inter_dim = stride_inter_win = 0

    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}

    _causal_conv1d_update_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        conv_state,
        conv_state_indices,
        intermediate_conv_window if intermediate_conv_window is not None else x,
        intermediate_state_indices,
        out,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_inter_seq,
        stride_inter_step,
        stride_inter_dim,
        stride_inter_win,
        stride_intermediate_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=256,
        SAVE_INTERMEDIATE=intermediate_conv_window is not None,
        **pdl_kwargs,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out
