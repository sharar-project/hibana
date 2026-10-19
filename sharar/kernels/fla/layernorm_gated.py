# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/layernorm_gated.py
# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# Forward only: the gated RMS norm (norm before gate, no bias) of the GDN layers, CUDA only.

from functools import lru_cache

import torch
import triton
import triton.language as tl

from sharar.kernels import stable
from sharar.kernels._compat import cdiv, device_context, is_arch_support_pdl, next_power_of_2

# Maximum rows per Triton block for layernorm gated kernel
MAX_ROWS_PER_BLOCK = 4


@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Z,  # pointer to the other branch
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_z_row,
    M,  # number of rows in X
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_Z: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    # Map the program id to the starting row of X and Y it should compute.
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)

    # Create 2D tile: [ROWS_PER_BLOCK, BLOCK_N]
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    # Compute offsets for 2D tile
    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * N

    # Base pointers
    X_base = X + row_offsets + col_offsets
    Y_base = Y + rows[:, None] * stride_y_row + col_offsets

    # Create mask for valid rows and columns
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    # Load input data with 2D tile
    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    # Mean square per row (reduce along axis 1)
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]

    rstd = tl.rsqrt(var + eps)  # Shape: [ROWS_PER_BLOCK]

    # Store rstd for each row
    rstd_offsets = group * M + rows
    rstd_mask = rows < M
    tl.store(Rstd + rstd_offsets, rstd, mask=rstd_mask)

    # Weights broadcast across rows
    w_offsets = cols + group * N
    w_mask = cols < N
    w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    x_hat = x * rstd[:, None]

    y = x_hat * w[None, :]

    if HAS_Z:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            y *= tl.sigmoid(z)

    # Write output
    tl.store(Y_base, y, mask=mask)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


@lru_cache
def _get_sm_count(device: torch.device) -> int:
    """Get and cache the SM count for a given device."""
    props = torch.cuda.get_device_properties(device)
    return props.multi_processor_count


def calc_rows_per_block(M: int, device: torch.device) -> int:
    # Use a constant value when the row count must not affect kernel numerics.
    if stable.enabled():
        return MAX_ROWS_PER_BLOCK
    sm_count = _get_sm_count(device)
    rows_per_block = next_power_of_2(cdiv(M, 2 * sm_count))
    rows_per_block = min(rows_per_block, MAX_ROWS_PER_BLOCK)
    return rows_per_block


def _layer_norm_fwd(
    x,
    weight,
    eps,
    z=None,
    group_size=None,
    activation: str = "swish",
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    out = torch.empty_like(x)
    assert out.stride(-1) == 1
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    # Calculate rows per block based on SM count
    rows_per_block = calc_rows_per_block(M, x.device)
    # Update grid to use rows_per_block
    grid = (cdiv(M, rows_per_block), ngroups)
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    with device_context(x.device):
        _layer_norm_fwd_1pass_kernel[grid](
            x,
            out,
            weight,
            z,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_N=BLOCK_N,
            ROWS_PER_BLOCK=rows_per_block,
            HAS_Z=z is not None,
            num_warps=num_warps,
            ACTIVATION=activation,
            **pdl_kwargs,
        )
    return out, rstd


def rms_norm_gated(
    *,
    x,
    weight,
    z=None,
    eps=1e-6,
    group_size=None,
    activation: str = "swish",
):
    """Grouped RMS norm of x, then times activation(z) when z is given ("swish"/"silu" or "sigmoid")."""

    x_shape_og = x.shape
    # reshape input data into 2D tensor
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        assert z.shape == x_shape_og
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
    weight = weight.contiguous()
    y, rstd = _layer_norm_fwd(
        x,
        weight,
        eps,
        z=z,
        group_size=group_size,
        activation=activation,
    )
    return y.reshape(x_shape_og)
