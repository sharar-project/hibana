# SPDX-License-Identifier: Apache-2.0
"""Flash-Next's sparse-attention indexer (Qwen Sparse Attention), on the engine's pages.

Per attention layer the checkpoint ships `index_qk_proj` (4 query heads and one
key head of 128) and Gemma norms for the index query and the compressed key.
Every token's raw index key is kept in its KV page (page aux row 0). When a
token completes a group of four (position % 4 == 3) the group's compressed key
is written to that token's page (row 1): the FP32 mean of the four raw keys,
Gemma-normed, rotated at the group's first position. A query at position p
sees the (p + 1) // 4 complete groups; past 512 of them it scores each group
as relu(q . k) summed over the heads / sqrt(128), keeps the top 512 and adds
its own pending tail, which is at most 2,051 tokens. The semantics are
SGLang's `qsa` package (checked in `tests/unit/test_qsa_reference.py` and
`tests/gpu/test_qsa.py`); the attention itself is `kernels/qsa/attention.py`.

Keys are maintained on every forward; selection runs only when some row is
past the budget, otherwise attention stays the dense path it is equal to.
Selection is `kernels/qsa/selection.py`: fixed shapes, so graphs replay it too.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sharar.model.flashnext.reference import gemma_rms, rope_neox


class FlashNextIndexer:
    def __init__(self, w, prefix: str, layer_id: int, device: torch.device, target):
        self.name = f"qsa.{layer_id}"
        self.heads, self.dim = target.indexer_n_heads, target.indexer_head_dim
        self.ratio, self.budget = target.indexer_compress_ratio, target.indexer_budget
        self.width = self.budget + self.ratio - 1  # selected tokens per row at most
        self.eps = target.rms_norm_eps
        self.rotary_dim, self.theta = target.rotary_dim, target.rope_theta
        p = f"{prefix}.indexer"
        self.qk = w.get(f"{p}.index_qk_proj.weight").to(device)  # BF16 [5 * 128, hidden], as the checkpoint ships it
        self.q_norm = w.get(f"{p}.q_layernorm.weight").to(device)
        self.k_norm = w.get(f"{p}.k_layernorm.weight").to(device)
        self._prefill = None
        self._prefill_backend = "original"

    def configure_prefill(self, target, cos_sin, *, sharded: bool = True) -> None:
        from sharar.engine.config import current
        from sharar.distributed import get_tp_info
        cfg = current()
        self._prefill = None
        # Replicated MTP attention is outside the target prefill setting.
        self._prefill_backend = cfg.qsa_prefill_backend if sharded else "original"
        if self._prefill_backend == "original":
            return
        tp_size = get_tp_info().size
        geometry = (target.num_attention_heads, target.num_kv_heads, target.head_dim)
        supported = ((tp_size == 1 and geometry == (24, 2, 256))
                     or (tp_size == 2 and geometry == (12, 1, 256)))
        index_geometry = (self.heads, self.dim, self.ratio, self.budget, self.rotary_dim)
        reason = None
        if cfg.stable_numerics:
            reason = "stable numerics is enabled"
        elif not supported or index_geometry != (4, 128, 4, 2048, 64):
            reason = f"unsupported TP={tp_size} attention/index geometry {geometry}/{index_geometry}"
        elif not self.qk.is_cuda or torch.cuda.get_device_capability(self.qk.device) != (12, 1):
            reason = "requires CUDA on SM121"
        if reason is not None:
            if self._prefill_backend == "optimized":
                raise ValueError(f"SHARAR_QSA_PREFILL_BACKEND=optimized: {reason}")
            return
        from sharar.kernels.qsa.prefill import PrefillKernels
        self._prefill = PrefillKernels(cos_sin)

    @property
    def bytes(self) -> int:
        return self.qk.numel() * self.qk.element_size()

    def register(self, kv_pool) -> None:
        kv_pool.register_page_aux(self.name, (2, self.dim), torch.bfloat16)  # [raw, compressed] per page

    def run(self, x: torch.Tensor, ctx, token_pages: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Store this forward's index keys and complete its groups; return the
        selected pages per row ([T, width] int32, valid first) and counts ([T]),
        or None when every row sees at most the budget (dense attention)."""
        if ctx.extras is not None:
            ctx.extras.qsa_optimized_prefill=False
        optimized=getattr(self,"_prefill",None)
        if optimized is not None and optimized.eligible(x,ctx,self):
            return optimized.run(self,x,ctx,token_pages)
        if getattr(self, "_prefill_backend", "original") == "optimized" and ctx.mode == "prefill":
            from sharar.kernels.qsa.prefill import PrefillKernels
            if PrefillKernels.in_scope(x, ctx, self):
                raise ValueError("SHARAR_QSA_PREFILL_BACKEND=optimized: unsupported sparse eager prefill; "
                                 "requires CUDA BF16 inputs, contiguous BF16 index weights, "
                                 "matching sequence metadata and context <=32768")
        h, d = self.heads, self.dim
        aux = ctx.kv_pool.page_aux[self.name]
        qk = F.linear(x, self.qk)
        aux[token_pages.long(), 0] = qk[:, h * d:]
        seqs = ctx.extras.qsa_seqs
        self._compress(aux, seqs, x.device, x.dtype)
        if all(first + n - 1 < self.width for first, n, _ in seqs):
            return None
        table = torch.cat([pages.to(torch.int32) for _, _, pages in seqs])
        indptr = torch.tensor([0] + [len(pages) for _, _, pages in seqs], dtype=torch.int64).cumsum(0)
        base = torch.repeat_interleave(indptr[:-1], torch.tensor([n for _, n, _ in seqs])).to(device=x.device, dtype=torch.int32)
        gcap = max((first + n) // self.ratio for first, n, _ in seqs)
        parts = [self.select(qk[a:a + 256], ctx.positions[a:a + 256], aux, table, base[a:a + 256], gcap)
                 for a in range(0, x.shape[0], 256)]  # [rows, groups] scores in row chunks
        return torch.cat([p for p, _ in parts]), torch.cat([c for _, c in parts])

    def select(self, qk: torch.Tensor, positions: torch.Tensor, aux: torch.Tensor, table: torch.Tensor, base: torch.Tensor,
               gcap: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Selection for rows at `positions` from their index projections `qk`; `base` is each
        row's offset in the flat page `table`, `gcap` the groups the widest row may see."""
        from sharar.kernels.qsa.selection import select_pages
        t, h, d = qk.shape[0], self.heads, self.dim
        q = gemma_rms(qk[:, : h * d].reshape(t, h, d), self.q_norm, self.eps)
        q = rope_neox(q, positions, self.rotary_dim, self.theta)
        return select_pages(q, aux, table, base, positions, self.ratio, self.budget, gcap, d ** -0.5)

    def maintain(self, x: torch.Tensor, ctx, token_pages: torch.Tensor) -> torch.Tensor:
        """Graph-safe key maintenance: store every row's raw key and compress every
        row's group through itself from `ctx.extras.qsa_members`; rows that do not complete a
        group write their result to the dummy page (`qsa_writes`). Returns the index
        projection (the queries of `select`)."""
        h, d = self.heads, self.dim
        aux = ctx.kv_pool.page_aux[self.name]
        qk = F.linear(x, self.qk)
        aux[token_pages.long(), 0] = qk[:, h * d:]
        side = ctx.extras
        keys = aux[side.qsa_members.long(), 0].float().mean(dim=1).to(x.dtype)
        keys = gemma_rms(keys.unsqueeze(1), self.k_norm, self.eps)
        first = (ctx.positions - (self.ratio - 1)).clamp_min(0)
        aux[side.qsa_writes.long(), 1] = rope_neox(keys, first, self.rotary_dim, self.theta)[:, 0]
        return qk

    @staticmethod
    def static_rows(seqs, ratio: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """(members [rows, ratio], writes [rows]) int32 for `maintain` from qsa_seqs-style
        (first position, new tokens, pages) per sequence."""
        members, writes = [], []
        back = torch.arange(ratio - 1, -1, -1, device=device)
        for first, n, pages in seqs:
            pos = torch.arange(first, first + n, device=device)
            members.append(pages[(pos[:, None] - back).clamp_min(0)])
            writes.append(torch.where(pos % ratio == ratio - 1, pages[pos], torch.zeros_like(pages[pos])))
        return torch.cat(members).to(torch.int32), torch.cat(writes).to(torch.int32)

    def _compress(self, aux, seqs, device, dtype) -> None:
        r = self.ratio
        ends = [(p, s) for s, (first, n, _) in enumerate(seqs) for p in range(first, first + n) if p % r == r - 1]
        if not ends:
            return
        members = torch.cat([seqs[s][2][p - r + 1:p + 1].long().unsqueeze(0) for p, s in ends])  # [G, ratio] pages
        keys = aux[members, 0].float().mean(dim=1).to(dtype)  # FP32 mean, stored in the activation dtype
        keys = gemma_rms(keys.unsqueeze(1), self.k_norm, self.eps)
        first_positions = torch.tensor([p - r + 1 for p, _ in ends], dtype=torch.int64, device=device)
        aux[members[:, -1], 1] = rope_neox(keys, first_positions, self.rotary_dim, self.theta)[:, 0]
