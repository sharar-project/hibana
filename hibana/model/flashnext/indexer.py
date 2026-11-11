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

from hibana.model.flashnext.reference import gemma_rms, rope_neox


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
        self.cos_sin = None  # the attention's rotary table (`configure_prefill`), for the graph path's kernels

    def configure_prefill(self, target, cos_sin, *, sharded: bool = True) -> None:
        from hibana.engine.config import current
        from hibana.distributed import get_tp_info
        cfg = current()
        self._prefill = None
        self.cos_sin = cos_sin
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
        from hibana.kernels.qsa.prefill import PrefillKernels
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
            from hibana.kernels.qsa.prefill import PrefillKernels
            if PrefillKernels.in_scope(x, ctx, self):
                raise ValueError("SHARAR_QSA_PREFILL_BACKEND=optimized: unsupported sparse eager prefill; "
                                 "requires CUDA BF16 inputs, contiguous BF16 index weights, "
                                 "matching sequence metadata and context <=32768")
        h, d = self.heads, self.dim
        aux = ctx.kv_pool.page_aux[self.name]
        qk = F.linear(x, self.qk)
        aux[token_pages.long(), 0] = qk[:, h * d:]
        plan = self._plan(ctx.extras, x.device)
        self._compress(aux, ctx.extras.qsa_seqs, x.device, x.dtype, plan)
        if plan["dense"]:
            return None
        # scores and top-k in chunks of 256 rows (a bounded score buffer); the queries' norm and rotation, and the
        # selection after the top-k, for every row at once (row by row the same bits)
        # (eager prefill on SM121: the same selection from faster kernels, `kernels/qsa/score_exact.py`)
        segments = [n for _, n, _ in ctx.extras.qsa_seqs] if getattr(ctx, "mode", None) == "prefill" else None
        return self.select(qk, ctx.positions, aux, plan["table"], plan["base"], plan["gcap"], chunk=256, segments=segments)

    def select(self, qk: torch.Tensor, positions: torch.Tensor, aux: torch.Tensor, table: torch.Tensor, base: torch.Tensor,
               gcap: int, chunk: int | None = None, segments: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Selection for rows at `positions` from their index projections `qk`; `base` is each
        row's offset in the flat page `table`, `gcap` the groups the widest row may see. `segments`: each sequence's
        row count, in row order, when eager rows may take the SM121 kernels of the same selection (`score_exact.py`)."""
        from hibana.kernels.qsa.selection import select_pages
        t, h, d = qk.shape[0], self.heads, self.dim
        q = gemma_rms(qk[:, : h * d].reshape(t, h, d), self.q_norm, self.eps)
        q = rope_neox(q, positions, self.rotary_dim, self.theta)
        if segments is not None:
            from hibana.kernels.qsa import score_exact
            if score_exact.supported(q, aux, self.ratio, self.budget):
                return score_exact.select_exact(q, aux, table, base, positions, gcap, d ** -0.5, segments, chunk or 256)
        return select_pages(q, aux, table, base, positions, self.ratio, self.budget, gcap, d ** -0.5, chunk)

    def replay(self, x: torch.Tensor, ctx, token_pages: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The graph path: `maintain`, then `select` in a selecting graph (`ctx.extras.qsa_table`), else None (dense).
        On CUDA with the rotary table, two launches after the projection (`kernels/qsa/decode_prep.py`, the same
        bits)."""
        from hibana.kernels.qsa import decode_prep
        side = ctx.extras
        aux = ctx.kv_pool.page_aux[self.name]
        sparse = side.qsa_table is not None
        cos_sin = getattr(self, "cos_sin", None)
        if not (x.is_cuda and decode_prep.supported(self, cos_sin)):
            qk = self.maintain(x, ctx, token_pages)
            if not sparse:
                return None
            return self.select(qk, ctx.positions, aux, side.qsa_table, side.qsa_indptr[side.qsa_row_seq], side.qsa_gcap)
        qk = F.linear(x, self.qk)
        prepped = decode_prep.prep(qk, aux, token_pages, ctx.positions, self.q_norm, cos_sin, self.heads, self.eps, sparse,
                                   side.qsa_indptr, side.qsa_row_seq)
        decode_prep.compress(aux, side.qsa_members, side.qsa_writes, ctx.positions, self.k_norm, cos_sin, self.eps)
        if not sparse:
            return None
        from hibana.kernels.qsa.selection import select_pages
        q, base = prepped
        return select_pages(q, aux, side.qsa_table, base, ctx.positions, self.ratio, self.budget, side.qsa_gcap, self.dim ** -0.5)

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

    def _plan(self, side, device) -> dict:
        """The forward's indexer layout, the same for every attention layer: computed by the first and kept on the side
        inputs. The group completions (`_groups`) and the selection's page table."""
        plan = getattr(side, "_qsa_plan", None)
        if plan is not None:
            return plan
        seqs, r = side.qsa_seqs, self.ratio
        plan = self._groups(seqs, device)
        plan["dense"] = all(first + n - 1 < self.width for first, n, _ in seqs)
        if not plan["dense"]:
            plan["table"] = seqs[0][2].to(torch.int32) if len(seqs) == 1 else torch.cat([p.to(torch.int32) for _, _, p in seqs])
            indptr = torch.tensor([0] + [len(p) for _, _, p in seqs], dtype=torch.int64).cumsum(0)
            plan["base"] = torch.repeat_interleave(indptr[:-1], torch.tensor([n for _, n, _ in seqs])).to(device=device, dtype=torch.int32)
            plan["gcap"] = max((first + n) // r for first, n, _ in seqs)
        side._qsa_plan = plan
        return plan

    def _groups(self, seqs, device) -> dict:
        """The groups the sequences' new tokens complete: their member pages [G, ratio] (one gather, indices from the
        host) and first positions [G], or members None."""
        import numpy as np
        r = self.ratio
        pages = seqs[0][2] if len(seqs) == 1 else torch.cat([p for _, _, p in seqs])
        offsets = np.cumsum([0] + [len(p) for _, _, p in seqs[:-1]])
        ends = [np.arange(first + (r - 1 - first) % r, first + n, r, dtype=np.int64) for first, n, _ in seqs]
        index = np.concatenate([o + e[:, None] - np.arange(r - 1, -1, -1) for o, e in zip(offsets, ends)])
        if not index.shape[0]:
            return {"members": None}
        return {"members": pages[torch.from_numpy(index).to(device)].long(),
                "first": torch.from_numpy(np.concatenate(ends) - (r - 1)).to(device)}

    def _compress(self, aux, seqs, device, dtype, groups: dict | None = None) -> None:
        """Write the compressed key of every group the sequences' new tokens complete; `groups`: `_groups(seqs)`."""
        groups = groups if groups is not None else self._groups(seqs, device)
        members = groups["members"]
        if members is None:
            return
        keys = aux[members, 0].float().mean(dim=1).to(dtype)  # FP32 mean, stored in the activation dtype
        keys = gemma_rms(keys.unsqueeze(1), self.k_norm, self.eps)
        aux[members[:, -1], 1] = rope_neox(keys, groups["first"], self.rotary_dim, self.theta)[:, 0]
