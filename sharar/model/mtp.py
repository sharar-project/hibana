"""The checkpoint's MTP head as the speculative draft model.

Structure (from the reference Qwen3_5ForCausalLMMTP): for a token x_i at
position i and the target's post-final-norm hidden state h_{i-1} of the
previous position,
  u = fc(cat(norm_e(embed(x_i)), norm_h(h_{i-1})))
  d = final_norm(attention_decoder_layer(u))      # one bf16 full-attention layer
  logits = lm_head(d)                              # the target's NVFP4 lm_head
which predicts x_{i+1}. Chained drafting feeds d back as the next hidden. The
layer keeps its own bf16 paged KV cache, addressed by the same page ids as the
target's pages for the same positions.
"""

from __future__ import annotations

import flashinfer
import torch

from sharar import config
from sharar.kernels import ops
from sharar.kernels.qk_norm_rope import qk_norm_rope
from sharar.memory.kv_pool import KvPool
from sharar.model.norm import GemmaRMSNorm
from sharar.model.rope import PartialRotary
from sharar.weights.modelopt import Nvfp4Linear
from sharar.weights.safetensors import SnapshotWeights
from sharar.kernels import plan_guard

PREFIX = "mtp"


class MtpHead:
    def __init__(self, w: SnapshotWeights, device: torch.device, embed: torch.Tensor, lm_head: Nvfp4Linear, rope: PartialRotary, num_pages: int):
        self.device = device
        self.embed = embed
        self.lm_head = lm_head
        self.rope = rope
        eps = config.RMS_NORM_EPS
        g = lambda name: w.get(f"{PREFIX}.{name}").to(device)  # noqa: E731
        self.fc = g("fc.weight")  # [5120, 10240]
        self.norm_e = GemmaRMSNorm(g("pre_fc_norm_embedding.weight"), eps)
        self.norm_h = GemmaRMSNorm(g("pre_fc_norm_hidden.weight"), eps)
        self.input_norm = GemmaRMSNorm(g("layers.0.input_layernorm.weight"), eps)
        self.post_norm = GemmaRMSNorm(g("layers.0.post_attention_layernorm.weight"), eps)
        self.final_norm = GemmaRMSNorm(g("norm.weight"), eps)
        self.q_norm_w = g("layers.0.self_attn.q_norm.weight").contiguous()
        self.k_norm_w = g("layers.0.self_attn.k_norm.weight").contiguous()
        self.w_qkv = torch.cat([g("layers.0.self_attn.q_proj.weight"), g("layers.0.self_attn.k_proj.weight"), g("layers.0.self_attn.v_proj.weight")], 0).contiguous()
        self.w_o = g("layers.0.self_attn.o_proj.weight").contiguous()
        self.w_gate_up = torch.cat([g("layers.0.mlp.gate_proj.weight"), g("layers.0.mlp.up_proj.weight")], 0).contiguous()
        self.w_down = g("layers.0.mlp.down_proj.weight").contiguous()
        self.num_heads = config.NUM_ATTENTION_HEADS
        self.num_kv_heads = config.NUM_KV_HEADS
        self.head_dim = config.HEAD_DIM
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.kv = KvPool(num_pages, device, layer_ids=[0], dtype=torch.bfloat16)
        self._ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self._attn = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self._ws, "NHD")

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.fc, self.w_qkv, self.w_o, self.w_gate_up, self.w_down)) + self.kv.bytes

    def plan(self, qo_lens: list[int], kv_pages: list[torch.Tensor]) -> None:
        d = self.device
        # Host index pointers: plan() reads them on the CPU (a device tensor would sync).
        qo_indptr = torch.tensor([0] + list(torch.tensor(qo_lens).cumsum(0).tolist()), dtype=torch.int32)
        kv_indptr = torch.tensor([0] + list(torch.tensor([p.shape[0] for p in kv_pages]).cumsum(0).tolist()), dtype=torch.int32)
        plan_guard.plan(self._attn,
            qo_indptr, kv_indptr, torch.cat(kv_pages), torch.ones(len(qo_lens), dtype=torch.int32),
            self.num_heads, self.num_kv_heads, self.head_dim, 1,
            causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, sm_scale=self.head_dim**-0.5,
        )

    @torch.inference_mode()
    def write_kv(self, tokens: torch.Tensor, hidden: torch.Tensor, positions: torch.Tensor, write_pages: torch.Tensor) -> None:
        """Commit target-conditioned KV without running draft attention/MLP.

        A full-accept bonus input has no corresponding draft forward, so it
        also needs this write before any subsequent draft reads its page.
        """
        e = self.norm_e(torch.nn.functional.embedding(tokens, self.embed))
        u = torch.cat([e, self.norm_h(hidden)], dim=-1) @ self.fc.T
        qkv = self.input_norm(u) @ self.w_qkv.T
        _, k = qk_norm_rope(qkv, positions, self.rope.cos_sin, self.q_norm_w, self.k_norm_w,
                            self.num_heads, self.num_kv_heads, self.head_dim, self.rope.rotary_dim, config.RMS_NORM_EPS)
        v = qkv[:, self.q_size * 2 + self.kv_size:].view(-1, self.num_kv_heads, self.head_dim).contiguous()
        self.kv.write(0, write_pages, k, v)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, hidden: torch.Tensor, positions: torch.Tensor, write_pages: torch.Tensor, qo_lens: list[int], kv_pages: list[torch.Tensor]) -> torch.Tensor:
        """tokens/hidden/positions/write_pages: [T] rows laid out per sequence
        (qo_lens); kv_pages[i] are all pages sequence i attends to (prefix and
        its own rows, in order). Returns the draft hidden d [T, 5120] (post
        final norm); logits come from `logits()`."""
        t = tokens.shape[0]
        e = self.norm_e(torch.nn.functional.embedding(tokens, self.embed))
        h = self.norm_h(hidden)
        u = torch.cat([e, h], dim=-1) @ self.fc.T
        # decoder layer
        x = self.input_norm(u)
        qkv = x @ self.w_qkv.T
        gate = qkv[:, : self.q_size * 2].view(t, self.num_heads, 2 * self.head_dim)[..., self.head_dim :]
        q, k = qk_norm_rope(qkv, positions, self.rope.cos_sin, self.q_norm_w, self.k_norm_w, self.num_heads, self.num_kv_heads, self.head_dim, self.rope.rotary_dim, config.RMS_NORM_EPS)
        v = qkv[:, self.q_size * 2 + self.kv_size :].view(t, self.num_kv_heads, self.head_dim).contiguous()
        self.kv.write(0, write_pages, k, v)
        self.plan(qo_lens, kv_pages)
        attn = self._attn.run(q, self.kv.cache(0)).view(t, self.num_heads, self.head_dim)
        attn = (attn * torch.sigmoid(gate)).reshape(t, self.q_size)
        a_out = attn @ self.w_o.T
        x, resid = self.post_norm.fused_add(a_out, u)
        gu = x @ self.w_gate_up.T
        m = torch.nn.functional.silu(gu[:, : config.INTERMEDIATE_SIZE]) * gu[:, config.INTERMEDIATE_SIZE :]
        m = m @ self.w_down.T
        out = self.final_norm(m + resid)
        return out

    @torch.inference_mode()
    def forward_static(self, tokens: torch.Tensor, hidden: torch.Tensor, positions: torch.Tensor, write_pages: torch.Tensor) -> torch.Tensor:
        """One row per sequence with the attention wrapper already planned
        (graph capture path); same math as forward()."""
        t = tokens.shape[0]
        e = self.norm_e(torch.nn.functional.embedding(tokens, self.embed))
        h = self.norm_h(hidden)
        u = torch.cat([e, h], dim=-1) @ self.fc.T
        x = self.input_norm(u)
        qkv = x @ self.w_qkv.T
        gate = qkv[:, : self.q_size * 2].view(t, self.num_heads, 2 * self.head_dim)[..., self.head_dim :]
        q, k = qk_norm_rope(qkv, positions, self.rope.cos_sin, self.q_norm_w, self.k_norm_w, self.num_heads, self.num_kv_heads, self.head_dim, self.rope.rotary_dim, config.RMS_NORM_EPS)
        v = qkv[:, self.q_size * 2 + self.kv_size :].view(t, self.num_kv_heads, self.head_dim).contiguous()
        self.kv.write(0, write_pages, k, v)
        attn = self._attn.run(q, self.kv.cache(0)).view(t, self.num_heads, self.head_dim)
        attn = (attn * torch.sigmoid(gate)).reshape(t, self.q_size)
        a_out = attn @ self.w_o.T
        x, resid = self.post_norm.fused_add(a_out, u)
        gu = x @ self.w_gate_up.T
        m = torch.nn.functional.silu(gu[:, : config.INTERMEDIATE_SIZE]) * gu[:, config.INTERMEDIATE_SIZE :]
        m = m @ self.w_down.T
        return self.final_norm(m + resid)

    @torch.inference_mode()
    def logits(self, d: torch.Tensor) -> torch.Tensor:
        return ops.nvfp4_linear(d, self.lm_head)
