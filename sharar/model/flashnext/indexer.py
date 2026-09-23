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
against the reference forward); the attention itself is `kernels/qsa.py`.

Keys are maintained on every forward; selection runs only when some row is
past the budget, otherwise attention stays the dense path it is equal to.
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

    @property
    def bytes(self) -> int:
        return self.qk.numel() * self.qk.element_size()

    def register(self, kv_pool) -> None:
        kv_pool.register_page_aux(self.name, (2, self.dim), torch.bfloat16)  # [raw, compressed] per page

    def run(self, x: torch.Tensor, ctx, token_pages: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Store this forward's index keys and complete its groups; return the
        selected pages per row ([T, width] int32, valid first) and counts ([T]),
        or None when every row sees at most the budget (dense attention)."""
        t, h, d, r = x.shape[0], self.heads, self.dim, self.ratio
        aux = ctx.kv_pool.page_aux[self.name]
        qk = F.linear(x, self.qk)
        aux[token_pages.long(), 0] = qk[:, h * d:]
        self._compress(aux, ctx.qsa_seqs, x.device, x.dtype)
        if all(first + n - 1 < self.width for first, n, _ in ctx.qsa_seqs):
            return None
        q = gemma_rms(qk[:, : h * d].reshape(t, h, d), self.q_norm, self.eps)
        q = rope_neox(q, ctx.positions, self.rotary_dim, self.theta)
        pages = torch.full((t, self.width), -1, dtype=torch.int32, device=x.device)
        row = 0
        for first, n, seq_pages in ctx.qsa_seqs:
            pages[row:row + n] = self._select(q[row:row + n], aux, first, n, seq_pages.long())
            row += n
        return pages, (pages >= 0).sum(1, dtype=torch.int32)

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

    def _select(self, q: torch.Tensor, aux, first: int, n: int, seq_pages: torch.Tensor) -> torch.Tensor:
        r, d = self.ratio, self.dim
        device = q.device
        positions = torch.arange(first, first + n, device=device)
        visible = (positions + 1) // r
        groups = (first + n) // r  # complete groups visible to the last row
        top = min(self.budget // r, groups)
        rows = torch.full((n, self.width), -1, dtype=torch.int64, device=device)
        if top:
            keys = aux[seq_pages[r - 1: groups * r: r], 1].float()  # [G, d]
            chosen = torch.empty((n, top), dtype=torch.int64, device=device)
            for a in range(0, n, 256):  # [rows, groups] scores in row chunks
                b = min(n, a + 256)
                logits = torch.zeros((b - a, groups), dtype=torch.float32, device=device)
                for head in range(self.heads):
                    logits += torch.relu(q[a:b, head].float() @ keys.T)
                logits /= d ** 0.5
                logits.masked_fill_(torch.arange(groups, device=device)[None, :] >= visible[a:b, None], -float("inf"))
                values, idx = logits.topk(top, dim=-1)
                chosen[a:b] = torch.where(torch.isfinite(values), idx, torch.full_like(idx, -1))
            members = chosen.unsqueeze(-1) * r + torch.arange(r, device=device)  # [n, top, r] positions
            members = torch.where(chosen.unsqueeze(-1) >= 0, members, torch.full_like(members, -1)).reshape(n, top * r)
            rows[:, : top * r] = members
        tail = (visible * r).unsqueeze(1) + torch.arange(r - 1, device=device)  # the pending group, itself included
        tail = torch.where(tail <= positions.unsqueeze(1), tail, torch.full_like(tail, -1))
        rows[:, self.width - (r - 1):] = tail
        order = torch.argsort((rows < 0).to(torch.int8), dim=1, stable=True)  # valid entries first, in order
        rows = rows.gather(1, order)
        return torch.where(rows >= 0, seq_pages[rows.clamp_min(0)], torch.full_like(rows, -1)).to(torch.int32)
