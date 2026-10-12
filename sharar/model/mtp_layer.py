"""The checkpoint's MTP layer as the speculative draft: what every target's MTP layer shares (#50).

For token x_i and the target's hidden h_{i-1} of the previous position, a layer fuses u from the embedding of x_i and
h_{i-1}, runs one decoder layer over its own BF16 paged KV (addressed by the same page ids as the target's pages for
the same positions) and returns the draft hidden d; `logits()` projects d through the target's vocabulary head, and
chained drafting feeds d back as the next hidden. The shared parts are the attention (the target's gated attention
module with FP8-or-BF16 projections, `model/flashnext/attention.py`, its q/k norm, RoPE and KV write fused in one
launch), the draft KV pool and its attention plan, and the call surface `proposers/mtp.py` and its graphs drive. A
target's subclass supplies `_fuse` (how u is formed), `_attention_input` and `_layer` (what wraps the attention: the
27B's residual and dense MLP, Flash-Next's hyper connections and experts), `logits` and `head_input`.
"""

from __future__ import annotations

import flashinfer
import torch

from sharar.cache.kv_pool import KvPool
from sharar.kernels import ops


class MtpLayer:
    DRAFT_HEADS = False  # whether set_draft_head is supported (a shortlist head reads FP8 lm_head rows)

    def _attention_setup(self, w, prefix: str, device: torch.device, target, rope, precision: str, num_pages: int,
                         padding_rows: int) -> None:
        """The attention module, the draft KV pool (padded draft-graph rows write its padding pages) and its plan."""
        from sharar.kernels.attention_plan import AttentionGeometry, AttentionPlanner
        from sharar.model.flashnext.attention import FlashNextAttention
        self.device, self.rope = device, rope
        self.ragged = ops.RaggedAttention(device, target.num_attention_heads, target.num_kv_heads, target.head_dim)
        self.attn = FlashNextAttention(w, prefix, 0, device, rope, self.ragged, target, precision, sharded=False)
        self.kv = KvPool(num_pages, device, layer_ids=[0], dtype=torch.bfloat16, target=target, padding_rows=padding_rows)
        self._ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self._attn = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self._ws, "NHD")
        self.attention = AttentionPlanner(AttentionGeometry.of_target(target, 1, torch.bfloat16))  # BF16 draft KV, page 1

    def plan(self, qo_lens: list[int], kv_pages: list[torch.Tensor]) -> None:
        # Host index pointers: plan() reads them on the CPU (a device tensor would sync).
        qo_indptr = torch.tensor([0] + list(torch.tensor(qo_lens).cumsum(0).tolist()), dtype=torch.int32)
        kv_indptr = torch.tensor([0] + list(torch.tensor([p.shape[0] for p in kv_pages]).cumsum(0).tolist()), dtype=torch.int32)
        self.attention.paged_prefill(self._attn, qo_indptr, kv_indptr, torch.cat(kv_pages), torch.ones(len(qo_lens), dtype=torch.int32), causal=True)

    def _qkv_write(self, h: torch.Tensor, positions: torch.Tensor, write_pages: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The attention's q/k norm and RoPE, v, and the draft KV write in one launch (the bits of
        `Attention._split_qkv` + `KvPool.write`): (q [T, Hq, D], the qkv rows)."""
        from sharar.kernels.flashnext_fused import qkv_epilogue
        a = self.attn
        qkv = a.qkv(h)
        k_cache, v_cache = self.kv.cache(0)
        q = qkv_epilogue(qkv, positions, a.rope.cos_sin, a.q_norm.weight, a.k_norm.weight, k_cache, v_cache, write_pages,
                         a.num_heads, a.num_kv_heads, a.head_dim, a.rope.rotary_dim, a.eps)
        return q, qkv

    def _attention(self, h: torch.Tensor, positions: torch.Tensor, write_pages: torch.Tensor, wrapper) -> torch.Tensor:
        """`Attention.prefill` in its verify mode (each row attends over its sequence's draft KV) with the fused epilogue;
        the gated output projection."""
        a = self.attn
        t = h.shape[0]
        q, qkv = self._qkv_write(h, positions, write_pages)
        attn = wrapper.run(q, self.kv.cache(0)).view(t, a.q_size)
        gate = qkv[:, : a.q_size * 2].view(t, a.num_heads, 2 * a.head_dim)[..., a.head_dim:]
        return a._finish(attn, gate)

    # ---- what a target supplies ----
    def _fuse(self, tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _attention_input(self, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _layer(self, u: torch.Tensor, positions: torch.Tensor, write_pages: torch.Tensor, wrapper) -> torch.Tensor:
        raise NotImplementedError

    # ---- the surface proposers/mtp.py drives ----
    @torch.inference_mode()
    def write_kv(self, tokens: torch.Tensor, hidden: torch.Tensor, positions: torch.Tensor, write_pages: torch.Tensor) -> None:
        """Commit target-conditioned KV for rows the draft never runs (prefill ingestion, a full-accept bonus input)."""
        self._qkv_write(self._attention_input(self._fuse(tokens, hidden)), positions, write_pages)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, hidden: torch.Tensor, positions: torch.Tensor, write_pages: torch.Tensor,
                qo_lens: list[int], kv_pages: list[torch.Tensor]) -> torch.Tensor:
        """tokens/hidden/positions/write_pages: [T] rows laid out per sequence (qo_lens); kv_pages[i] are all pages
        sequence i attends to (its prefix and its own rows, in order). Returns the draft hidden d."""
        self.plan(qo_lens, kv_pages)
        return self._layer(self._fuse(tokens, hidden), positions, write_pages, self._attn)

    @torch.inference_mode()
    def forward_static(self, tokens: torch.Tensor, hidden: torch.Tensor, positions: torch.Tensor, write_pages: torch.Tensor) -> torch.Tensor:
        """One row per sequence with the attention wrapper already planned (graph capture); the same math as forward()."""
        return self._layer(self._fuse(tokens, hidden), positions, write_pages, self._attn)
