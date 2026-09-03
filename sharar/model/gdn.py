"""Gated DeltaNet linear-attention layer (48 of the 64 layers).

Prefill mirrors SGLang's GDNAttnBackend.forward_extend with the Triton kernels:
FP8 qkv/z projection, bf16 b/a projection, causal Conv1D with state write-back,
fused gating, chunked gated delta rule with state write-back, gated RMSNorm,
FP8 output projection. Decode mirrors forward_decode: Conv1D state update and
the packed fused recurrent kernel.
"""

from __future__ import annotations

import torch

from sharar import config
from sharar.engine.context import ForwardContext
from sharar.kernels import ops
from sharar.kernels.conv1d import causal_conv1d_fn, causal_conv1d_update
from sharar.kernels.fla.chunk import chunk_gated_delta_rule
from sharar.kernels.fla.fused_gdn_gating import fused_gdn_gating
from sharar.kernels.fla.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
from sharar.kernels.fla.layernorm_gated import rms_norm_gated
from sharar.kernels.fp8_quant import quant_fp8
from sharar.memory.state_pool import CONV_DIM, CONV_STATE_LEN
from sharar.weights.modelopt import Fp8Linear, load_fp8_linear
from sharar.weights.safetensors import SnapshotWeights


class GatedDeltaNet:
    def __init__(self, w: SnapshotWeights, prefix: str, layer_id: int, device: torch.device):
        self.layer_id = layer_id
        self.device = device
        self.num_k_heads = config.LINEAR_NUM_KEY_HEADS
        self.num_v_heads = config.LINEAR_NUM_VALUE_HEADS
        self.head_k_dim = config.LINEAR_KEY_HEAD_DIM
        self.head_v_dim = config.LINEAR_VALUE_HEAD_DIM
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        assert self.conv_dim == CONV_DIM
        # The reference fuses [q | k | v | z] into one FP8 projection whose shards
        # are requantized to a shared scale. Keep those exact weights but run
        # two GEMMs so both outputs are contiguous with no copy.
        fused = load_fp8_linear(w, [f"{prefix}.in_proj_qkv", f"{prefix}.in_proj_z"], device)
        self.in_qkvz = fused
        self.in_qkv = Fp8Linear(fused.weight[: self.conv_dim].contiguous(), fused.weight_scale, fused.input_scale)
        self.in_z = Fp8Linear(fused.weight[self.conv_dim :].contiguous(), fused.weight_scale, fused.input_scale)
        del fused.weight
        self.in_qkvz = None
        self.in_ba = torch.cat([w.get(f"{prefix}.in_proj_b.weight"), w.get(f"{prefix}.in_proj_a.weight")], dim=0).to(device)
        self.conv_weight = w.get(f"{prefix}.conv1d.weight").to(device).view(self.conv_dim, config.LINEAR_CONV_KERNEL)
        self.A_log = w.get(f"{prefix}.A_log").to(device).float().contiguous()
        self.dt_bias = w.get(f"{prefix}.dt_bias").to(device).contiguous()
        # Per-head gated RMSNorm as a grouped norm over [T, value_dim] so the
        # z gate can stay a strided view of the qkvz projection (no copy).
        self.norm_weight = w.get(f"{prefix}.norm.weight").to(device).repeat(self.num_v_heads).contiguous()
        self.out = load_fp8_linear(w, [f"{prefix}.out_proj"], device)
        self._scratch_conv: torch.Tensor | None = None
        self._scratch_ssm: torch.Tensor | None = None

    def _states(self, ctx: ForwardContext | None) -> tuple[torch.Tensor, torch.Tensor]:
        if ctx is not None and ctx.state_pool is not None:
            return ctx.state_pool.conv[self.layer_id], ctx.state_pool.ssm[self.layer_id]
        # Stand-alone use (parity tools): one private slot, cleared each call.
        if self._scratch_conv is None:
            self._scratch_conv = torch.zeros((1, self.conv_dim, CONV_STATE_LEN), dtype=torch.bfloat16, device=self.device)
            self._scratch_ssm = torch.zeros((1, self.num_v_heads, self.head_v_dim, self.head_k_dim), dtype=torch.float32, device=self.device)
        self._scratch_conv.zero_()
        self._scratch_ssm.zero_()
        return self._scratch_conv, self._scratch_ssm

    @property
    def input_scale(self) -> torch.Tensor:
        return self.in_qkv.input_scale

    def _project(self, x: torch.Tensor, x_q: torch.Tensor | None = None):
        if x_q is None:
            x_q = quant_fp8(x, self.in_qkv.input_scale)
        mixed_qkv = ops.fp8_linear(None, self.in_qkv, x_q=x_q)
        z = ops.fp8_linear(None, self.in_z, x_q=x_q)
        ba = x @ self.in_ba.T
        b, a = ba.split([self.num_v_heads, self.num_v_heads], dim=-1)
        return mixed_qkv, z, b.contiguous(), a.contiguous()

    def _finish(self, core: torch.Tensor, z: torch.Tensor, t: int) -> torch.Tensor:
        core = rms_norm_gated(
            x=core.reshape(t, self.value_dim), weight=self.norm_weight, bias=None, z=z,
            eps=config.RMS_NORM_EPS, group_size=self.head_v_dim, norm_before_gate=True,
            is_rms_norm=True, activation="swish",
        )
        return ops.fp8_linear(None, self.out, x_q=quant_fp8(core, self.out.input_scale))

    def prefill(self, x: torch.Tensor, ctx: ForwardContext | None = None, x_q: torch.Tensor | None = None) -> torch.Tensor:
        """Fresh-sequence prefill into the context's slot. The chunk kernel reads
        the initial state from the pool, so the slot must be cleared beforehand
        (StatePool.alloc does; the scratch path does)."""
        t = x.shape[0]
        device = x.device
        conv_states, ssm_states = self._states(ctx)
        slot = ctx.slot if (ctx is not None and ctx.state_pool is not None) else 0
        mixed_qkv, z, b, a = self._project(x, x_q)
        query_start_loc = torch.tensor([0, t], dtype=torch.int32, device=device)
        cache_indices = torch.tensor([slot], dtype=torch.int32, device=device)
        has_prefix = ctx is not None and ctx.prefix_len > 0
        checkpoints = list(ctx.checkpoint_positions) if ctx is not None else []
        if checkpoints:
            # conv snapshot = the last 3 pre-conv inputs before the boundary
            prefix = ctx.prefix_len
            for pos, cslot in zip(checkpoints, ctx.checkpoint_slots):
                local = pos - prefix
                conv_states[cslot] = mixed_qkv[local - CONV_STATE_LEN : local].transpose(0, 1)
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            self.conv_weight,
            None,
            activation="silu",
            conv_states=conv_states,
            has_initial_state=torch.tensor([has_prefix], device=device),
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=[t],
        ).transpose(0, 1)[:t]
        q, k, v = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(1, t, self.num_k_heads, self.head_k_dim)
        k = k.reshape(1, t, self.num_k_heads, self.head_k_dim)
        v = v.reshape(1, t, self.num_v_heads, self.head_v_dim)
        g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
        core, _, h = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        if checkpoints:
            # h[0, c] is the state entering chunk c, i.e. after 64*c new tokens.
            prefix = ctx.prefix_len
            for pos, cslot in zip(checkpoints, ctx.checkpoint_slots):
                local = pos - prefix
                assert local % 64 == 0 and 0 < local < t, (pos, prefix, t)
                ssm_states[cslot] = h[0, local // 64].to(ssm_states.dtype)
        return self._finish(core, z, t)

    def decode(self, x: torch.Tensor, ctx: ForwardContext, x_q: torch.Tensor | None = None) -> torch.Tensor:
        b_sz = x.shape[0]
        conv_states = ctx.state_pool.conv[self.layer_id]
        ssm_states = ctx.state_pool.ssm[self.layer_id]
        mixed_qkv, z, b, a = self._project(x, x_q)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv, conv_states, self.conv_weight, None, "silu", conv_state_indices=ctx.slots
        )
        out = mixed_qkv.new_empty(b_sz, 1, self.num_v_heads, self.head_v_dim)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=ctx.slots,
            use_qk_l2norm_in_kernel=True,
        )
        return self._finish(out, z, b_sz)

    def forward(self, x: torch.Tensor, ctx: ForwardContext, x_q: torch.Tensor | None = None) -> torch.Tensor:
        if ctx.mode == "prefill":
            return self.prefill(x, ctx, x_q)
        return self.decode(x, ctx, x_q)
