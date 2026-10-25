# SPDX-License-Identifier: Apache-2.0
"""Nemotron-H's Mamba-2 mixer (40 of the 88 blocks) on the engine's recurrent-layer contract.

FP8 in_proj -> z | xBC | dt; causal conv (kernel 4, bias, SiLU) over xBC; the SSD recurrence over 128 heads of 64
with 8 B/C groups of state 128 (`kernels/mamba2.py`); y * silu(z) normalized in groups of 1024, quantized into the
FP8 out_proj. The pool's conv state is [slots, 10240, 3] and its SSM state Mamba's [slots, heads, head dim, state]
FP32 (FP16 under SHARAR_SSM_STATE=fp16_sr, rounded stochastically on every store, keyed on the token position
and this layer's index among the Mamba layers; a per-window quantized byte row under int8 / int4), so prefix-cache
checkpoints, scratch slots and the SSD tier carry it as they carry the GDN's. Prefill continues
each sequence from its slot and writes interior checkpoints; decode advances the slot in place; verify leaves it and
records either every step's state (`snapshots`) or the compact inputs the accept step replays (`reconstruct`).
"""

from __future__ import annotations

import torch

from sharar.engine.context import ForwardContext
from sharar.kernels import mamba2, ops
from sharar.weights.modelopt import load_fp8_linear


class NemotronMamba:
    def __init__(self, w, prefix: str, layer_id: int, device: torch.device, target):  # noqa: D107
        self.layer_id = layer_id
        self.index = list(target.gdn_layer_ids).index(layer_id)  # the stochastic rounding's layer key (`ssm_all` row)
        self.device = device
        g = lambda name: w.get(f"{prefix}.{name}").to(device)  # noqa: E731
        self.params = mamba2.prepare_params(g("conv1d.weight"), g("conv1d.bias"), g("A_log"), g("D"), g("dt_bias"),
                                            g("norm.weight"), target.rms_norm_eps, groups=target.linear_num_key_heads,
                                            state=target.linear_key_head_dim)
        self.in_proj = load_fp8_linear(w, [f"{prefix}.in_proj"], device)  # [z 8192 | xBC 10240 | dt 128, 4096]
        self.out_proj = load_fp8_linear(w, [f"{prefix}.out_proj"], device)
        self.d_inner = self.params.d_inner
        self._scratch = None

    @property
    def bytes(self) -> int:
        p = self.params
        return (self.in_proj.bytes + self.out_proj.bytes
                + sum(t.numel() * t.element_size() for t in (p.conv_weight, p.conv_bias, p.A, p.D, p.dt_bias, p.norm_weight)))

    @property
    def input_scale(self) -> torch.Tensor:
        return self.in_proj.input_scale

    def _states(self, ctx: ForwardContext | None):
        if ctx is not None and ctx.state_pool is not None:
            return ctx.state_pool.conv[self.layer_id], ctx.state_pool.ssm[self.layer_id]
        # Stand-alone use (parity tools): one private slot, cleared each call.
        if self._scratch is None:
            p = self.params
            self._scratch = (torch.zeros((1, p.conv_dim, 3), dtype=torch.bfloat16, device=self.device),
                             torch.zeros((1, p.heads, p.head_dim, p.state), dtype=torch.float32, device=self.device))
        for t in self._scratch:
            t.zero_()
        return self._scratch

    def _out(self, y: torch.Tensor, zxbcdt: torch.Tensor) -> torch.Tensor:
        x_q = mamba2.gated_rmsnorm(y, zxbcdt[:, :self.d_inner], self.params, self.out_proj.input_scale)
        return ops.fp8_linear(None, self.out_proj, x_q=x_q)

    def forward(self, x: torch.Tensor, ctx: ForwardContext, x_q: torch.Tensor | None = None) -> torch.Tensor:
        zxbcdt = ops.fp8_linear(x, self.in_proj, x_q=x_q)
        conv, ssm = self._states(ctx)
        pool = ctx.state_pool if ctx is not None else None
        window = getattr(pool, "window", None)
        storage = dict(layer=self.index,
                       window=(window.layout, window.layer[self.layer_id]) if window is not None else None)
        if ctx is None or ctx.mode == "prefill":
            y = mamba2.prefill(zxbcdt, self.params, conv, ssm, mamba2.prepare_prefill(ctx, x.shape[0], x.device),
                               **storage)
        elif ctx.mode == "verify":
            y = mamba2.verify(zxbcdt, self.params, conv, ssm, ctx.seq_slots, ctx.verify_steps,
                              ctx.verify_conv_windows.get(self.layer_id),
                              snapshots=ctx.inter_ssm.get(self.layer_id), snapshot_indices=ctx.verify_scratch,
                              replay=ctx.verify_replay.get(self.layer_id), positions=ctx.positions, **storage)
        else:
            y = mamba2.decode(zxbcdt, self.params, conv, ssm, ctx.slots, ctx.positions, **storage)
        return self._out(y, zxbcdt)
