"""Serving dispatch for indexed GDN prefill; decode/verify are unchanged.

CuTe uses packed FP32 states. Gather/scatter preserve arbitrary pool slots;
explicit checkpoint metadata maps its after-interval snapshots to prefix slots.
Never silently fall back after a launch/compile error.
"""
from functools import lru_cache

import torch

from hibana.engine.config import current


@lru_cache(maxsize=None)
def _sm12x(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) in ((12, 0), (12, 1))


def backend(q, k, v, state) -> str:
    policy = current().gdn_prefill
    if policy not in ("auto", "cute", "triton"):
        raise ValueError(f"invalid SHARAR_GDN_PREFILL: {policy!r}")
    supported = (
        q.is_cuda and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.shape[-1] == k.shape[-1] == v.shape[-1] == 128
        and q.shape[2] == k.shape[2] and v.shape[2] % q.shape[2] == 0
        and state.dtype == torch.float32 and state.is_contiguous()
        and _sm12x(q.device.index)
    )
    # Stable numerics stay on Triton. Explicit cute is a diagnostic
    # override, not a promise of bit identity with Triton's 64-token algorithm.
    if policy == "cute" and not supported:
        raise ValueError("CuTe GDN requires SM120/121, BF16 D128 and FP32 state")
    if policy == "triton" or (policy == "auto" and (not supported or current().stable_numerics)):
        return "triton"
    return "cute"


def prefill(*, q, k, v, g, beta, state, plan, triton_kernel=None):
    if backend(q, k, v, state) == "triton":
        if triton_kernel is None:
            from hibana.kernels.fla.chunk import chunk_gated_delta_rule as triton_kernel
        core, _, history = triton_kernel(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=state,
            initial_state_indices=plan.cache_indices,
            cu_seqlens=plan.query_start_loc, chunk_indices=plan.chunk_indices,
            chunk_offsets=plan.chunk_offsets,
            use_qk_l2norm_in_kernel=True,
        )
        for slot, _, chunk in plan.checkpoints:
            state[slot] = history[0, chunk].to(state.dtype)
        return core

    from hibana.kernels.fla.l2norm import l2norm_fwd
    from hibana.kernels.cute.gdn.delta_rule_sm120 import delta_rule_prefill_dsl

    slots = plan.cache_indices.long()
    packed = state.index_select(0, slots).contiguous()
    snapshots = torch.empty(
        (plan.checkpoint_count, *packed.shape[1:]), device=state.device,
        dtype=torch.float32,
    ) if plan.checkpoint_count else None
    out = torch.empty_like(v[0])
    delta_rule_prefill_dsl(
        out, packed, l2norm_fwd(q.contiguous())[0],
        l2norm_fwd(k.contiguous())[0], v[0].contiguous(), packed,
        g[0].exp().contiguous(), beta[0].float().contiguous(),
        plan.cute_cu_seqlens, q.shape[-1] ** -0.5,
        state_checkpoints=snapshots,
        checkpoint_cu_starts=plan.checkpoint_cu_starts if snapshots is not None else None,
        checkpoint_every_n_tokens=plan.checkpoint_interval,
        chunk_size=32, coalesced_state_io=True,
        indexed_checkpoints=True,
    )
    state.index_copy_(0, slots, packed)
    if snapshots is not None:
        state.index_copy_(0, plan.checkpoint_destinations,
                          snapshots.index_select(0, plan.checkpoint_indices))
    return out.unsqueeze(0)
