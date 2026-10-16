# SPDX-License-Identifier: Apache-2.0
"""Flash-Next's in-checkpoint MTP layer as the speculative draft (the `mtp` proposer's head).

For token x_i and the target's hyper stream h_{i-1} (the [T, hc*H] residual before the global
mixer, `forward()["norm"]`), the head fuses u = fc_embedding(norm_e(embed(x_i))) broadcast over
the streams plus fc_hidden applied per stream to norm_h(h_{i-1}), runs one hyper-connected
decoder layer (dense attention over its own BF16 KV, BF16 routed experts, shared expert) and
returns the layer's hyper stream d; `logits()` applies the MTP's own mixer and the target's
lm_head. Chained drafting feeds d back as the next hidden. The attention, the draft KV and the
call surface are the shared MTP layer's (`model/mtp_layer.py`).
"""

from __future__ import annotations


import torch
import torch.nn.functional as F

from sharar.engine.config import current
from sharar.model.flashnext.hyper import HyperConnection
from sharar.model.flashnext.linear import TrunkLinear
from sharar.model.mtp_layer import MtpLayer
from sharar.model.norm import GemmaRMSNorm

PREFIX = "mtp"


class FlashNextMtpHead(MtpLayer):
    DRAFT_HEADS = True  # set_draft_head: reduced vocabulary and/or 3-bit shortlist

    def __init__(self, w, device: torch.device, model, num_pages: int, precision: str | None = None, padding_rows: int = 0):
        # Tensor parallel: the draft layer is replicated, whole on every rank (built from the whole geometry), and
        # drafts through the target's vocabulary-parallel lm_head (`VocabParallelHead`: each rank reads its rows).
        t = self.target = getattr(model, "full_target", model.target)
        self.tensor_parallel = t is not model.target
        self.embed, self.lm_head = model.embed, model.lm_head
        from sharar.proposers.draft_head import TargetHead, VocabParallelHead
        # replaced by set_draft_head
        self.draft_head = VocabParallelHead(self.lm_head) if self.tensor_parallel else TargetHead(self.lm_head)
        self.hidden_size, self.hc = t.hidden_size, t.hc_count
        self.hidden_width = self.hc * self.hidden_size  # the "hidden" the proposer carries per row
        # SHARAR_FLASHNEXT_MTP_TRUNK: the draft head's own precision, bf16 or fp8 (default: the trunk's).
        precision = precision or current().flashnext_mtp_trunk or model.precision
        eps = t.rms_norm_eps
        g = lambda name: w.get(f"{PREFIX}.{name}").to(device)  # noqa: E731
        self.fc_e = TrunkLinear.from_bf16(g("fc_embedding.weight"), precision)
        self.fc_h = TrunkLinear.from_bf16(g("fc_hidden.weight"), precision)
        self.norm_e = GemmaRMSNorm(g("pre_fc_norm_embedding.weight"), eps)
        self.norm_h = GemmaRMSNorm(g("pre_fc_norm_hidden.weight"), eps)  # over the whole hc*H vector
        p = f"{PREFIX}.layers.0"
        mixers = current().flashnext_mixers
        self.attn_hc = HyperConnection.load(w, f"{p}.attn_hyper_connection", self.hc, self.hidden_size, eps, device, precision=mixers)
        self.mlp_hc = HyperConnection.load(w, f"{p}.mlp_hyper_connection", self.hc, self.hidden_size, eps, device, precision=mixers)
        self.mixer = HyperConnection.load(w, f"{PREFIX}.hyper_connection_mixer", self.hc, self.hidden_size, eps, device, combine=False, precision=mixers)
        self._attention_setup(w, f"{p}.self_attn", device, t, model.rope, precision, num_pages, padding_rows)
        # Routed experts, fused [E, 2I, H] (gate rows then up rows) and [E, H, I]: stored FP8 with one
        # scale per expert matrix, upcast per token for the ten gathered experts (5.2 GB BF16 would not
        # fit beside the target on the GB10); loaded expert by expert to avoid a 3.4 GB host temporary.
        self.router = w.get(f"{p}.mlp.gate.weight").to(device).contiguous()
        self.gate_up, self.gate_up_scale = self._fp8_experts(w, f"{p}.mlp.experts.gate_up_proj", device)
        self.down, self.down_scale = self._fp8_experts(w, f"{p}.mlp.experts.down_proj", device)
        self.experts, self.topk, self.inter = t.num_experts, t.experts_per_token, t.moe_intermediate_size
        self.shared_gate = w.get(f"{p}.mlp.shared_expert_gate.weight").to(device).float().reshape(1, -1)
        self.shared_gate_up = TrunkLinear.load(w, [f"{p}.mlp.shared_expert.gate_proj.weight", f"{p}.mlp.shared_expert.up_proj.weight"], device, precision)
        self.shared_down = TrunkLinear.load(w, [f"{p}.mlp.shared_expert.down_proj.weight"], device, precision)

    @staticmethod
    def _fp8_experts(w, name: str, device) -> tuple[torch.Tensor, torch.Tensor]:
        from safetensors import safe_open
        shard = w._map[name]
        with safe_open(str(w.snap / shard), framework="pt", device="cpu") as handle:
            view = handle.get_slice(name)
            e, *rest = view.get_shape()
            out = torch.empty((e, *rest), dtype=torch.float8_e4m3fn, device=device)
            scale = torch.empty((e,), dtype=torch.float32, device=device)
            for i in range(e):
                m = view[i:i + 1][0].float()
                s = (m.abs().amax() / 448.0).clamp_min(1e-12)
                out[i].copy_((m / s).clamp(-448, 448).to(torch.float8_e4m3fn))
                scale[i] = s
        return out, scale

    @property
    def bytes(self) -> int:
        weights = (self.fc_e.bytes + self.fc_h.bytes + self.attn.bytes + self.attn_hc.bytes + self.mlp_hc.bytes + self.mixer.bytes
                   + self.router.numel() * 2 + self.gate_up.numel() + self.down.numel() + self.shared_gate_up.bytes + self.shared_down.bytes)
        return weights + self.kv.bytes + self.draft_head.bytes

    # ---- pieces ----
    def _fuse(self, tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        e = self.fc_e(self.norm_e(F.embedding(tokens, self.embed)))  # [T, H]
        h = self.fc_h(self.norm_h(hidden).view(-1, self.hc, self.hidden_size))  # [T, hc, H]
        return (h + e.unsqueeze(-2)).reshape(-1, self.hidden_width)

    def _moe(self, x: torch.Tensor) -> torch.Tensor:
        from sharar.kernels import flashnext_fused as fk
        ids, weights = fk.route_topk(x @ self.router.T, self.topk)
        flat = ids.reshape(-1)
        # routed row r reads input row r // topk; the down projection applies silu_mul to its [gate | up] rows itself
        gu = fk.expert_gemv_fused(self.gate_up, self.gate_up_scale, x, flat, x_div=self.topk)  # [M*topk, 2I]
        y = fk.expert_gemv_fused(self.down, self.down_scale, gu, flat, act=fk.ACT_SILU_MUL)  # [M*topk, H]
        shared = self.shared_down(self.shared_gate_up(x), act=fk.ACT_SILU_MUL)
        return fk.moe_combine_routed(y, weights, shared, x, self.shared_gate.reshape(-1))

    def _attention_input(self, u: torch.Tensor) -> torch.Tensor:
        return self.attn_hc.mix(u)[0]

    def _layer(self, u: torch.Tensor, positions: torch.Tensor, write_pages: torch.Tensor, wrapper) -> torch.Tensor:
        h, res = self.attn_hc.mix(u)
        a = self._attention(h, positions, write_pages, wrapper)
        u = self.attn_hc.combine(a, res)
        h, res = self.mlp_hc.mix(u)
        return self.mlp_hc.combine(self._moe(h), res)

    # ---- the vocabulary projection ----
    @torch.inference_mode()
    def logits(self, d: torch.Tensor) -> torch.Tensor:
        from sharar.distributed import all_gather
        return all_gather(self.lm_head(self.mixer.mix(d)[0]))  # the whole vocabulary on every rank

    def head_input(self, d: torch.Tensor) -> torch.Tensor:
        """What the vocabulary projection reads: the MTP's own mixer over the hyper stream."""
        return self.mixer.mix(d)[0]

    def set_draft_head(self, ids: torch.Tensor | None = None, fmt: str = "fp8", rescore: int = 0) -> None:
        """Draft through a head of its own (`proposers/draft_head.py`): over these token ids only
        (len(ids) rows of the shared lm_head instead of the vocabulary) and/or a 3-bit copy whose
        top `rescore` candidates the shared lm_head rescores."""
        from sharar.proposers.draft_head import ShortlistHead, VocabParallelHead
        if self.tensor_parallel:
            if ids is not None:
                raise ValueError("a reduced draft vocabulary (SHARAR_DRAFT_VOCAB) is not served tensor-parallel")
            self.draft_head = VocabParallelHead(self.lm_head, fmt, rescore)
            return
        self.draft_head = ShortlistHead(self.lm_head, ids, fmt, rescore)
