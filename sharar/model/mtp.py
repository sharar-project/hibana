"""The 27B's MTP head as the speculative draft (`model/mtp_layer.py` holds what every target's MTP layer shares).

Structure (from the reference Qwen3_5ForCausalLMMTP): for a token x_i at position i and the target's post-final-norm
hidden state h_{i-1} of the previous position,
  u = fc(cat(norm_e(embed(x_i)), norm_h(h_{i-1})))
  d = final_norm(attention_decoder_layer(u))      # one full-attention layer, dense MLP
  logits = lm_head(d)                              # the target's NVFP4 lm_head
which predicts x_{i+1}. The projections are BF16 (`precision="fp8"` quantizes them at load, as Flash-Next's).
"""

from __future__ import annotations

import torch

from sharar.kernels import ops
from sharar.model.flashnext.linear import TrunkLinear
from sharar.model.mtp_layer import MtpLayer
from sharar.model.norm import GemmaRMSNorm

PREFIX = "mtp"


class MtpHead(MtpLayer):
    DRAFT_HEADS = False  # only the target's NVFP4 head (a shortlist head needs FP8 rows)

    def __init__(self, w, device: torch.device, model, num_pages: int, precision: str | None = None, padding_rows: int = 0):
        t = self.target = model.target
        self.embed, self.lm_head = model.embed, model.lm_head
        self.hidden_width = t.hidden_size  # the hidden row the proposer carries
        from sharar.proposers.draft_head import TargetHead
        self.draft_head = TargetHead(self.logits)  # the target's NVFP4 head
        precision = precision or "bf16"
        eps = t.rms_norm_eps
        g = lambda name: w.get(f"{PREFIX}.{name}").to(device)  # noqa: E731
        p = f"{PREFIX}.layers.0"
        self.fc = TrunkLinear.from_bf16(g("fc.weight"), precision)  # [H, 2H] over cat(embedding, hidden)
        self.norm_e = GemmaRMSNorm(g("pre_fc_norm_embedding.weight"), eps)
        self.norm_h = GemmaRMSNorm(g("pre_fc_norm_hidden.weight"), eps)
        self.input_norm = GemmaRMSNorm(g("layers.0.input_layernorm.weight"), eps)
        self.post_norm = GemmaRMSNorm(g("layers.0.post_attention_layernorm.weight"), eps)
        self.final_norm = GemmaRMSNorm(g("norm.weight"), eps)
        self.gate_up = TrunkLinear.load(w, [f"{p}.mlp.gate_proj.weight", f"{p}.mlp.up_proj.weight"], device, precision)
        self.down = TrunkLinear.load(w, [f"{p}.mlp.down_proj.weight"], device, precision)
        self._attention_setup(w, f"{p}.self_attn", device, t, model.rope, precision, num_pages, padding_rows)

    @property
    def bytes(self) -> int:
        return self.fc.bytes + self.attn.bytes + self.gate_up.bytes + self.down.bytes + self.kv.bytes

    def _fuse(self, tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([self.norm_e(torch.nn.functional.embedding(tokens, self.embed)), self.norm_h(hidden)], dim=-1))

    def _attention_input(self, u: torch.Tensor) -> torch.Tensor:
        return self.input_norm(u)

    def _layer(self, u: torch.Tensor, positions: torch.Tensor, write_pages: torch.Tensor, wrapper) -> torch.Tensor:
        from sharar.kernels.flashnext_fused import ACT_SILU_MUL
        a = self._attention(self._attention_input(u), positions, write_pages, wrapper)
        x, resid = self.post_norm.fused_add(a, u)
        return self.final_norm(self.down(self.gate_up(x), act=ACT_SILU_MUL) + resid)

    @torch.inference_mode()
    def logits(self, d: torch.Tensor) -> torch.Tensor:
        return ops.nvfp4_linear(d, self.lm_head)

    def head_input(self, d: torch.Tensor) -> torch.Tensor:
        return d
