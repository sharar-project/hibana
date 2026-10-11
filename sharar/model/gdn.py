"""Gated DeltaNet linear-attention layer (48 of the 64 layers).

Prefill mirrors SGLang's GDNAttnBackend.forward_extend with the Triton kernels:
FP8 qkv/z projection, bf16 b/a projection, causal Conv1D with state write-back,
fused gating, chunked gated delta rule with state write-back, gated RMSNorm,
FP8 output projection. Decode mirrors forward_decode: Conv1D state update and
the packed fused recurrent kernel.
"""

from __future__ import annotations

import torch

from sharar.target import QWEN38, Target
from sharar.engine.context import ForwardContext
from sharar.kernels import ops, stable
from sharar.kernels.conv1d import causal_conv1d_fn, causal_conv1d_update
from sharar.kernels.fla.chunk import chunk_gated_delta_rule
from sharar.kernels.fla.fused_gdn_gating import fused_gdn_gating
from sharar.kernels.fla.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
from sharar.kernels.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
from sharar.kernels.fla.layernorm_gated import rms_norm_gated
from sharar.kernels.fp8_quant import quant_fp8
from sharar.kernels.ssm_window import materialize, quantize, window_commit, window_verify
from sharar.kernels.verify_conv import verify_convolve
from sharar.cache.state_pool import CONV_STATE_LEN
from sharar.model.gdn_plan import prepare_gdn_prefill
from sharar.engine.upload import upload
from sharar.weights.modelopt import Fp8Linear, load_fp8_linear
from sharar.weights.sharding import ROW_PARALLEL, ColumnParallel
from sharar.weights.safetensors import SnapshotWeights


def _window(ctx):
    """The quantized pool's views (`cache/ssm_format.py`), or None for an FP32 pool."""
    return getattr(ctx.state_pool, "window", None) if ctx is not None else None


class GatedDeltaNet:
    fused_window_norm = True  # the window kernel ends with this layer's gated RMS norm and FP8 out-projection input

    def __init__(self, w: SnapshotWeights, prefix: str, layer_id: int, device: torch.device, target: Target = QWEN38):
        self.layer_id = layer_id
        self.device = device
        self.eps = target.rms_norm_eps
        self.num_k_heads = target.linear_num_key_heads
        self.num_v_heads = target.linear_num_value_heads
        self.head_k_dim = target.linear_key_head_dim
        self.head_v_dim = target.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        # Head counts are this rank's (`Target.shard`): every per-head tensor keeps its rows of q, k, v (and z, b,
        # a), whole heads of each, so a value head stays with its key head; out_proj is row-parallel.
        qkv_heads = ColumnParallel((self.key_dim, self.key_dim, self.value_dim))
        v_heads = ColumnParallel((self.num_v_heads,))
        # The reference fuses [q | k | v | z] into one FP8 projection whose shards
        # are requantized to a shared scale. Keep that layout but run
        # two GEMMs so both outputs are contiguous with no copy.
        fused = load_fp8_linear(w, [f"{prefix}.in_proj_qkv", f"{prefix}.in_proj_z"], device,
                                ColumnParallel((self.key_dim, self.key_dim, self.value_dim, self.value_dim)))
        self.in_qkvz = fused
        self.in_qkv = Fp8Linear(fused.weight[: self.conv_dim].contiguous(), fused.weight_scale, fused.input_scale, shard=qkv_heads)
        self.in_z = Fp8Linear(fused.weight[self.conv_dim :].contiguous(), fused.weight_scale, fused.input_scale,
                              shard=ColumnParallel((self.value_dim,)))
        del fused.weight
        self.in_qkvz = None
        self.in_ba = torch.cat([v_heads(w.get(f"{prefix}.in_proj_b.weight")), v_heads(w.get(f"{prefix}.in_proj_a.weight"))], dim=0).to(device)
        self.conv_weight = qkv_heads(w.get(f"{prefix}.conv1d.weight").view(-1, target.linear_conv_kernel)).contiguous().to(device)
        self.A_log = v_heads(w.get(f"{prefix}.A_log")).to(device).float().contiguous()
        self.dt_bias = v_heads(w.get(f"{prefix}.dt_bias")).to(device).contiguous()
        # Per-head gated RMSNorm as a grouped norm over [T, value_dim] so the
        # z gate can stay a strided view of the qkvz projection (no copy).
        self.norm_weight = w.get(f"{prefix}.norm.weight").to(device).repeat(self.num_v_heads).contiguous()
        self.out = load_fp8_linear(w, [f"{prefix}.out_proj"], device, ROW_PARALLEL)
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
        b, a = self._project_ba(x)
        return mixed_qkv, z, b, a

    def _project_ba(self, x):
        if stable.enabled():
            from sharar.kernels.ba_proj import ba_project
            ba = ba_project(x, self.in_ba)  # M-invariant arithmetic; cuBLAS's choice changes with M
        else:
            ba = x @ self.in_ba.T
        b, a = ba.split([self.num_v_heads, self.num_v_heads], dim=-1)
        # Every gating/recurrent kernel consumes the token strides explicitly.
        return b, a

    def _finish(self, core: torch.Tensor, z: torch.Tensor, t: int, *, verify: bool = False) -> torch.Tensor:
        return ops.fp8_linear(None, self.out, x_q=self._output_input(core, z, t, verify=verify))

    def _output_input(self, core, z, t, *, verify=False):
        if verify:
            from sharar.kernels.verify_norm import verify_gated_norm
            normalized = verify_gated_norm(core, z, self.norm_weight, self.eps, self.head_v_dim)
            return quant_fp8(normalized.reshape(t, self.value_dim), self.out.input_scale)
        core = rms_norm_gated(
            x=core.reshape(t, self.value_dim), weight=self.norm_weight, bias=None, z=z,
            eps=self.eps, group_size=self.head_v_dim, norm_before_gate=True,
            is_rms_norm=True, activation="swish",
        )
        return quant_fp8(core, self.out.input_scale)

    def prefill(self, x: torch.Tensor, ctx: ForwardContext | None = None, x_q: torch.Tensor | None = None) -> torch.Tensor:
        """Prefill one sequence, or several (ctx.batched) laid end to end.
        Each sequence continues from the state in its slot when it has a prefix
        (cached or earlier chunks); the chunk kernel reads the initial state
        from the pool, so fresh slots must be cleared (StatePool.alloc does)."""
        mixed_qkv, z, b, a = self._project(x, x_q)
        return self._finish(self._prefill_core(mixed_qkv, b, a, ctx), z, x.shape[0])

    def _prefill_core(self, mixed_qkv, b, a, ctx):
        t, device = mixed_qkv.shape[0], mixed_qkv.device
        conv_states, ssm_states = self._states(ctx)
        plan = prepare_gdn_prefill(ctx, t, device)
        window = _window(ctx)
        if window is not None:
            # Quantized pool: run the chunked kernel on FP32 copies of the slots' states, then quantize the final states
            # and the checkpoints back as new boundary states.
            ssm_states = materialize(window.layout, window.layer[self.layer_id], plan.cache_indices)
            state_indices = torch.arange(ssm_states.shape[0], dtype=plan.cache_indices.dtype, device=device)
        else:
            state_indices = plan.cache_indices
        # Snapshot the last three pre-convolution inputs at each boundary.
        for slot, end, _ in plan.checkpoints:
            conv_states[slot] = mixed_qkv[end - CONV_STATE_LEN:end].transpose(0, 1)
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            self.conv_weight,
            None,
            activation="silu",
            conv_states=conv_states,
            has_initial_state=plan.has_initial_state,
            cache_indices=plan.cache_indices,
            query_start_loc=plan.query_start_loc,
            seq_lens_cpu=plan.seq_lens,
        ).transpose(0, 1)[:t]
        q, k, v = self._split_heads(mixed_qkv.unsqueeze(0))
        g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
        core, _, h = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=ssm_states,
            initial_state_indices=state_indices,
            cu_seqlens=plan.query_start_loc,
            chunk_indices=plan.chunk_indices,
            chunk_offsets=plan.chunk_offsets,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        # h[0, c] is the state entering chunk c, laid out per sequence.
        if window is not None:
            views = window.layer[self.layer_id]
            quantize(window.layout, views, ssm_states, plan.cache_indices)
            if plan.checkpoints:
                quantize(window.layout, views, h[0, [chunk for _, _, chunk in plan.checkpoints]],
                         upload([slot for slot, _, _ in plan.checkpoints], torch.int32, device))
            return core
        for slot, _, chunk in plan.checkpoints:
            ssm_states[slot] = h[0, chunk].to(ssm_states.dtype)
        return core

    def decode(self, x: torch.Tensor, ctx: ForwardContext, x_q: torch.Tensor | None = None) -> torch.Tensor:
        b_sz = x.shape[0]
        conv_states = ctx.state_pool.conv[self.layer_id]
        ssm_states = ctx.state_pool.ssm[self.layer_id]
        mixed_qkv, z, b, a = self._project(x, x_q)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv, conv_states, self.conv_weight, None, "silu", conv_state_indices=ctx.slots
        )
        if _window(ctx) is not None:
            return self._window_out(self._window_decode(mixed_qkv, z, b, a, ctx), z, b_sz)
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

    def _window_decode(self, mixed_qkv, z, b, a, ctx):
        """One token per row on a quantized pool, as a verify of one token accepted whole: the window recurrence writes
        the token's record to a one-step ring, and the commit appends it to the slot's window (folding a full one).
        Returns the kernel's output, which `_window_out` finishes."""
        window = ctx.state_pool.window
        b_sz, d = mixed_qkv.shape[0], mixed_qkv.device
        q, k, v = self._split_heads(mixed_qkv.unsqueeze(0))
        ring_k = torch.empty((1, b_sz, self.num_k_heads, 1, self.head_k_dim), dtype=torch.float16, device=d)
        ring_u = torch.empty((1, b_sz, self.num_v_heads, 1, self.head_v_dim), dtype=torch.float16, device=d)
        ring_g = torch.empty((1, b_sz, self.num_v_heads, 1), dtype=torch.float32, device=d)
        x_q = window_verify(window.layout, window.layer[self.layer_id], q, k, v, a, b, self.A_log, self.dt_bias,
                            ctx.slots, 1, ring_k[0], ring_u[0], ring_g[0], norm=self._window_norm(z))
        window_commit(window.layout, window.single[self.layer_id], ring_k, ring_u, ring_g, ctx.slots,
                      torch.ones(b_sz, dtype=torch.int32, device=d))
        return x_q

    def _window_norm(self, z):
        """The window kernel's epilogue: this layer's gated RMS norm and the out projection's input quantization."""
        return (z, self.norm_weight, self.eps, self.out.input_scale) if self.fused_window_norm else None

    def _window_out(self, out, z, t, *, verify=False):
        """The layer output from the window kernel's: the out projection of its FP8 input, or `_finish` of its core output."""
        if self.fused_window_norm:
            return ops.fp8_linear(None, self.out, x_q=out)
        return self._finish(out, z, t, verify=verify)

    def _split_heads(self, conv):
        """[1, T, conv_dim] -> q/k [1, T, H, K], v [1, T, HV, V] (views)."""
        t = conv.shape[1]
        q, k, v = torch.split(conv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        return (q.reshape(1, t, self.num_k_heads, self.head_k_dim), k.reshape(1, t, self.num_k_heads, self.head_k_dim),
                v.reshape(1, t, self.num_v_heads, self.head_v_dim))

    def verify(self, x: torch.Tensor, ctx: ForwardContext, x_q: torch.Tensor | None = None) -> torch.Tensor:
        """Speculative verify: B sequences of S = k+1 tokens each. Fused conv
        reads the committed window and records candidate windows; the recurrent kernel
        starts from the committed SSM state, leaves it untouched, and records
        the state after every token so the accept step can commit any prefix."""
        mixed_qkv, z, b, a = self._project(x, x_q)
        core = self._verify_core(mixed_qkv, z, b, a, ctx)
        if _window(ctx) is not None:
            return self._window_out(core, z, x.shape[0], verify=True)
        return self._finish(core, z, x.shape[0], verify=True)

    def _verify_core(self, mixed_qkv, z, b, a, ctx):
        s_len = ctx.verify_steps
        b_sz = mixed_qkv.shape[0] // s_len
        conv_states = ctx.state_pool.conv[self.layer_id]
        ssm_states = ctx.state_pool.ssm[self.layer_id]
        conv = verify_convolve(mixed_qkv, conv_states, self.conv_weight, ctx.seq_slots,
                               ctx.verify_conv_windows[self.layer_id], s_len)
        q, k, v = self._split_heads(conv.reshape(1, b_sz * s_len, self.conv_dim))
        window = _window(ctx)
        if window is not None:  # records into the round's ring; `_window_out` finishes the output
            ring = ctx.verify_replay[self.layer_id]
            return window_verify(window.layout, window.layer[self.layer_id], q, k, v, a, b, self.A_log, self.dt_bias,
                                 ctx.seq_slots, s_len, ring["ring_k"], ring["ring_u"], ring["ring_g"],
                                 norm=self._window_norm(z))
        core = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log, dt_bias=self.dt_bias, softplus_beta=1.0, softplus_threshold=20.0,
            q=q, k=k, v=v, a=a, b=b,
            initial_state_source=ssm_states, initial_state_indices=ctx.seq_slots,
            use_qk_l2norm_in_kernel=True, cu_seqlens=ctx.cu_seqlens,
            disable_state_update=True,
            intermediate_states_buffer=ctx.inter_ssm.get(self.layer_id),
            intermediate_state_indices=ctx.verify_scratch,
            cache_steps=s_len,
            ring_uses_scratch=bool(ctx.verify_replay),
            **ctx.verify_replay.get(self.layer_id, {}),
        )
        return core


    def forward(self, x: torch.Tensor, ctx: ForwardContext, x_q: torch.Tensor | None = None) -> torch.Tensor:
        if ctx.mode == "prefill":
            return self.prefill(x, ctx, x_q)
        if ctx.mode == "verify":
            return self.verify(x, ctx, x_q)
        return self.decode(x, ctx, x_q)
