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
`tests/gpu/test_qsa.py`); the attention itself is `kernels/qsa.py`.

Keys are maintained on every forward; selection runs only when some row is
past the budget, otherwise attention stays the dense path it is equal to.
Selection is `kernels/qsa_select.py`: fixed shapes, so graphs replay it too.
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
        from sharar.kernels.qsa_select import select_pages
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

    _staging: dict = {}  # device -> (pinned int64 index buffer, event of its last upload): `static_rows`

    @staticmethod
    def static_rows(seqs, ratio: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """(members [rows, ratio], writes [rows]) int32 for `maintain` from qsa_seqs-style
        (first position, new tokens, pages) per sequence.

        One gather from the sequences' concatenated pages, its indices computed on the host and uploaded from a pinned
        buffer: a handful of launches whatever the batch. The loop per sequence it replaces queued ~8 small kernels per
        sequence behind the draft graphs; at 128 requests the host blocked ~39 ms on the full launch queue, so the
        verification's PLE rows were read only after the drafts finished."""
        import numpy as np
        firsts = np.fromiter((first for first, _, _ in seqs), dtype=np.int64, count=len(seqs))
        counts = np.fromiter((n for _, n, _ in seqs), dtype=np.int64, count=len(seqs))
        lengths = np.fromiter((pages.shape[0] for _, _, pages in seqs), dtype=np.int64, count=len(seqs))
        base = np.repeat(np.concatenate([[0], np.cumsum(lengths)[:-1]]), counts)  # each row's sequence in the concatenation
        pos = np.repeat(firsts, counts) + (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts))
        rows = pos.shape[0]
        members = base[:, None] + np.maximum(pos[:, None] - np.arange(ratio - 1, -1, -1)[None, :], 0)  # [rows, ratio]
        written = pos % ratio == ratio - 1
        index = np.empty((rows, ratio + 2), dtype=np.int64)
        index[:, :ratio], index[:, ratio], index[:, ratio + 1] = members, base + pos, written
        pages = torch.cat([p for _, _, p in seqs])
        if pages.device.type != "cuda":
            flat = torch.from_numpy(index).to(pages.device)
        else:
            buf, event = FlashNextIndexer._staging.get(pages.device, (None, None))
            if buf is None or buf.numel() < index.size:
                buf, event = torch.empty(max(index.size, 1 << 16), dtype=torch.int64, pin_memory=True), torch.cuda.Event()
                FlashNextIndexer._staging[pages.device] = (buf, event)
            else:
                event.synchronize()  # the previous upload has read the buffer
            buf[:index.size].numpy()[:] = index.reshape(-1)
            flat = buf[:index.size].to(pages.device, non_blocking=True)
            event.record()
        flat = flat.view(rows, ratio + 2)
        selected = pages[flat[:, :ratio + 1]]  # members, then each row's own page
        writes = torch.where(flat[:, ratio + 1] != 0, selected[:, ratio], torch.zeros_like(selected[:, ratio]))
        return selected[:, :ratio].to(torch.int32), writes.to(torch.int32)

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
