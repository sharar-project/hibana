"""Draft heads: how a draft body turns its hidden rows into token choices.

Every speculator ends in a vocabulary projection: the MTP heads take the argmax,
DFlash2's selector takes the top-k with values. The
default is the target's own head (`TargetHead`). A `ShortlistHead` reads fewer bytes: a subset
of the head's rows (a reduced draft vocabulary), a 3-bit or 2-bit copy of them, or both,
with the leading candidates rescored against the target's FP8 rows. The target
still verifies with its own head, so under exact acceptance a cheaper draft head
changes how many drafts are accepted, never what is emitted.
"""

from __future__ import annotations

from typing import Callable

import torch

FORMATS = ("fp8", "w3", "w2")  # SHARAR_DRAFT_HEAD
RESCORE = {"w3": 8, "w2": 16}  # SHARAR_DRAFT_HEAD_RESCORE's default per format: the 2-bit top 8 miss the FP8 argmax 0.6% of steps, the top 16 0.2%


def confidence(logits: torch.Tensor) -> torch.Tensor:
    """[..., V] logits -> FP32 [..., 2]: the top-1 softmax probability and its margin over the top-2."""
    top = torch.softmax(logits.float(), dim=-1).topk(2, dim=-1).values
    return torch.stack([top[..., 0], top[..., 0] - top[..., 1]], dim=-1)


class TargetHead:
    """The target's vocabulary projection as a draft head: `logits_fn(x)` over every row."""

    rows_changed = False

    def __init__(self, logits_fn: Callable[[torch.Tensor], torch.Tensor]):
        self.logits_fn = logits_fn

    @property
    def bytes(self) -> int:
        return 0  # the target's own head, already resident

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits_fn(x)

    def argmax(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits_fn(x).argmax(-1)

    def argmax_confidence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The argmax and the head's `confidence` in it (SHARAR_TRACE_CONFIDENCE)."""
        logits = self.logits_fn(x)
        return logits.argmax(-1), confidence(logits)

    def topk(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(values, global ids), each [..., k]."""
        return self.logits_fn(x).topk(k, dim=-1)

    def candidates(self, x: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The head's r best tokens: (global ids int64 [..., r], their FP32 logits [..., r]), descending. A sampled MTP
        draft's law lives on them (proposers/mtp.py)."""
        values, ids = self.logits_fn(x).topk(r, dim=-1)  # then FP32: exact for BF16, and r values instead of the vocabulary
        return ids, values.float()


def _weight_scale(lin) -> torch.Tensor | None:
    """An FP8 head's one weight scale: a `TrunkLinear`'s `scale` (Flash-Next; None when it is BF16) or an
    `Fp8Linear`'s `weight_scale` (Nemotron's checkpoint head, which also carries a static input scale)."""
    return lin.scale if hasattr(lin, "scale") else lin.weight_scale


class ShortlistHead:
    """The rows `ids` of an FP8 lm_head (Flash-Next's `TrunkLinear`; every row when `ids` is None, also of Nemotron's
    `Fp8Linear`), as they are or as a 3-bit (`fmt="w3"`) or 2-bit (`"w2"`) copy; with `rescore` > 1 the copy's logits only pick
    that many candidates and the target head's own FP8 rows choose among them."""

    rows_changed = True

    def __init__(self, lm_head, ids: torch.Tensor | None = None, fmt: str = "fp8", rescore: int = 0):
        from hibana.model.flashnext.linear import TrunkLinear
        if fmt not in FORMATS:
            raise ValueError(f"draft head format must be fp8, w3 or w2, not {fmt!r}")
        if not hasattr(lm_head, "weight") or lm_head.weight.dim() != 2 or lm_head.weight.dtype == torch.uint8:
            raise ValueError("a shortlist draft head reads rows of an FP8 lm_head (Flash-Next, Nemotron); the 27B's NVFP4 head is not supported yet")
        if ids is not None and not isinstance(lm_head, TrunkLinear):
            raise ValueError("a reduced draft vocabulary reads rows of a TrunkLinear lm_head (Flash-Next)")
        self.fmt = fmt
        self.ids = None if ids is None else ids.to(lm_head.weight.device, torch.int64).unique(sorted=True)
        self.real_rows = None if self.ids is None else self.ids.numel()  # rows past it repeat the last id (_pad_rows)
        if fmt != "fp8" and self.ids is not None:
            self.ids = _pad_rows(self.ids, 128)
        rows = lm_head if self.ids is None else TrunkLinear(lm_head.weight[self.ids].contiguous(), lm_head.scale)
        self.head, self.w3 = (None, pack_head(rows, fmt)) if fmt != "fp8" else (rows, None)
        # Shortlist: the low-bit logits pick `rescore` candidates and the target head's own FP8 rows
        # (already resident) choose among them, so the pick is the full head's whenever its
        # argmax is on the shortlist.
        self.rescore, self.source = (rescore, lm_head) if fmt != "fp8" and rescore > 1 else (0, None)

    @property
    def bytes(self) -> int:
        """Bytes this head adds beside the target's lm_head (0 when it is the target's own)."""
        if self.w3 is not None:
            return self.w3.bytes
        return 0 if self.ids is None else self.head.bytes

    def _logits(self, flat: torch.Tensor) -> torch.Tensor:
        if self.w3 is not None:
            from hibana.kernels import w3
            if flat.shape[0] <= w3.MAX_ROWS:  # one part: no concatenating copy of the [M, V] logits
                return w3.w3_linear(flat.contiguous(), self.w3)
            return torch.cat([w3.w3_linear(part.contiguous(), self.w3) for part in flat.split(w3.MAX_ROWS)])
        return self.head(flat)

    def _global(self, local: torch.Tensor) -> torch.Tensor:
        return local if self.ids is None else self.ids[local]

    def argmax(self, x: torch.Tensor) -> torch.Tensor:
        return self._argmax(x.reshape(-1, x.shape[-1]))[0]

    def argmax_confidence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The argmax and the `confidence` of this head's own logits (the 3-bit or shortlisted rows)."""
        token, logits = self._argmax(x.reshape(-1, x.shape[-1]))
        return token, confidence(logits)

    def _argmax(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self._logits(flat)
        if self.rescore:
            cand, exact = self._shortlist(flat, logits, self.rescore)
            return cand.gather(-1, exact.argmax(-1, keepdim=True)).squeeze(-1), logits
        return self._global(logits.argmax(-1)), logits

    def topk(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        flat = x.reshape(-1, x.shape[-1])
        logits = self._logits(flat)
        if self.rescore:
            cand, exact = self._shortlist(flat, logits, max(self.rescore, k))
            values, order = exact.topk(k, dim=-1)
            scale = _weight_scale(self.source)
            scale = scale.float().reshape(()) if scale is not None else 1.0
            values, ids = (values * scale).to(torch.bfloat16), cand.gather(-1, order)
        else:
            values, local = logits.topk(k, dim=-1)
            ids = self._global(local)
        return values.view(*x.shape[:-1], k), ids.view(*x.shape[:-1], k)

    def candidates(self, x: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The head's r best tokens, (global ids int64 [..., r], FP32 logits [..., r]) descending: with rescoring, the
        best r of the shortlist (the max(rescore, r) best by the low-bit logits) by the target head's FP8 rows, weight
        scale applied, so the shortlist holds the `argmax` pick; else the r best rows of this head. Padded rows
        (repeats of the last id) are never candidates, so the ids are unique per row."""
        flat = x.reshape(-1, x.shape[-1])
        logits = self._logits(flat)
        if self.real_rows is not None and logits.shape[-1] > self.real_rows:
            logits = logits.float().masked_fill(torch.arange(logits.shape[-1], device=logits.device) >= self.real_rows, float('-inf'))
        if self.rescore:
            cand, exact = self._shortlist(flat, logits, max(self.rescore, r))
            values, order = exact.topk(r, dim=-1)
            scale = _weight_scale(self.source)
            values, ids = values * (scale.float().reshape(()) if scale is not None else 1.0), cand.gather(-1, order)
        else:
            values, local = logits.topk(r, dim=-1)
            values, ids = values.float(), self._global(local)
        return ids.view(*x.shape[:-1], r), values.view(*x.shape[:-1], r)

    def _shortlist(self, x: torch.Tensor, logits: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The shortlist's global ids [M, r] and their logits by the target head's FP8 rows, without the
        common weight scale [M, r]."""
        cand = self._global(logits.topk(r, dim=-1).indices)
        rows = self.source.weight[cand].float()  # [M, r, K] of the target's head
        return cand, torch.bmm(rows, x.float().unsqueeze(-1)).squeeze(-1)


def _pad_rows(ids: torch.Tensor, multiple: int) -> torch.Tensor:
    """Repeat the last id up to a multiple of `multiple` rows (the 3-bit kernel tiles 128 rows);
    a repeated row ties with itself and argmax keeps the first, so the pick is unchanged."""
    short = -ids.numel() % multiple
    return ids if not short else torch.cat([ids, ids[-1:].expand(short)])


def pack_head(lin, fmt: str = "w3", rows_per_chunk: int = 16384):
    """3-bit (`w3`) or 2-bit (`w2`: four levels, fitted group scales) copy of an FP8 head [N, K] (`TrunkLinear` or
    `Fp8Linear`), dequantized a slice at a time (the whole lm_head in FP32 would be a 2.5 GB temporary)."""
    scale = _weight_scale(lin)
    return pack_rows(lambda r0, r1: lin.weight[r0:r1].float() * scale.float() if scale is not None else lin.weight[r0:r1].float(),
                     lin.out_features, lin.in_features, fmt, rows_per_chunk)


def pack_rows(read, rows: int, cols: int, fmt: str = "w3", rows_per_chunk: int = 16384):
    """`pack_head` over any head: `read(r0, r1)` returns rows [r0, r1) dequantized to FP32."""
    from hibana.kernels import w3
    codebook, fit = ("lloyd2", True) if fmt == "w2" else (w3.DEFAULT_CODEBOOK, False)
    parts = [w3.pack(read(r0, min(r0 + rows_per_chunk, rows)), codebook, fit_scales=fit) for r0 in range(0, rows, rows_per_chunk)]
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    return w3.W3Weight(torch.cat([p.planes for p in parts]), torch.cat([p.scales for p in parts]), torch.cat([p.row_scale for p in parts]),
                       rows, cols, first.codebook, first.lut)


class VocabParallelHead:
    """A draft head over a vocabulary-parallel lm_head (tensor parallel): each rank scores only its rows of the
    vocabulary, from its FP8 rows (`fmt="fp8"`) or a 3-bit or 2-bit copy of them (`"w3"`, `"w2"`), and the ranks agree on the
    pick through two small collectives instead of every rank reading the whole head each draft step.

    Without rescoring the pick is the best of every rank's best. With `rescore` r > 1 each rank proposes its r
    best by the 3-bit logits, the r best of those are the shortlist, and each candidate's FP8 logit (computed by
    the rank that holds its row) chooses among them: what `ShortlistHead` picks over the whole head, since a row's
    3-bit copy and FP8 logit do not depend on which rank holds it."""

    rows_changed = True

    def __init__(self, lm_head, fmt: str = "fp8", rescore: int = 0):
        from hibana.distributed import get_tp_info
        if fmt not in FORMATS:
            raise ValueError(f"draft head format must be fp8, w3 or w2, not {fmt!r}")
        tp = get_tp_info()
        self.lm_head, self.fmt = lm_head, fmt
        self.local = lm_head.out_features  # this rank's rows: [rank * local, (rank + 1) * local)
        self.offset = tp.rank * self.local
        self.w3 = pack_head(lm_head, fmt) if fmt != "fp8" else None
        self.rescore = rescore if fmt != "fp8" and rescore > 1 else 0

    @property
    def bytes(self) -> int:
        return self.w3.bytes if self.w3 is not None else 0

    def _logits(self, flat: torch.Tensor) -> torch.Tensor:
        if self.w3 is not None:
            from hibana.kernels import w3
            return torch.cat([w3.w3_linear(part.contiguous(), self.w3) for part in flat.split(w3.MAX_ROWS)])
        return self.lm_head(flat)

    def _gather(self, x: torch.Tensor) -> torch.Tensor:
        """[M, n] from every rank -> [M, size * n] in rank order."""
        from hibana.distributed import all_gather
        return all_gather(x.contiguous())

    def _pick(self, flat: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        from hibana.distributed import all_reduce
        r = max(self.rescore, 1)
        values, local = logits.topk(r, dim=-1)
        values, ids = self._gather(values.float()), self._gather(local + self.offset)  # [M, size * r]
        if not self.rescore:
            return ids.gather(-1, values.argmax(-1, keepdim=True)).squeeze(-1)
        top = values.topk(r, dim=-1).indices
        cand = ids.gather(-1, top)  # the shortlist's global ids [M, r]
        mine = (cand >= self.offset) & (cand < self.offset + self.local)
        rows = self.lm_head.weight[(cand - self.offset).clamp(0, self.local - 1)].float()  # [M, r, K]
        exact = torch.bmm(rows, flat.float().unsqueeze(-1)).squeeze(-1) * mine  # each candidate's logit on its own rank
        exact = all_reduce(exact)  # FP32: NCCL; exactly one rank adds each candidate's logit to zeros
        return cand.gather(-1, exact.argmax(-1, keepdim=True)).squeeze(-1)

    def argmax(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        return self._pick(flat, self._logits(flat)).view(x.shape[:-1])

    def candidates(self, x: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`ShortlistHead.candidates` across the ranks: each proposes its max(rescore, r) best, the best of those are the
        shortlist, and with rescoring each candidate's FP8 logit (from the rank holding its row, weight scale applied)
        orders them. Every rank returns the same (ids [..., r], FP32 logits [..., r])."""
        from hibana.distributed import all_reduce
        flat = x.reshape(-1, x.shape[-1])
        width = max(self.rescore, r)
        values, local = self._logits(flat).topk(width, dim=-1)
        values, ids = self._gather(values.float()), self._gather(local + self.offset)  # [M, size * width]
        top_v, top = values.topk(width, dim=-1)
        cand = ids.gather(-1, top)
        if not self.rescore:
            out_ids, out_v = cand[:, :r], top_v[:, :r]
        else:
            mine = (cand >= self.offset) & (cand < self.offset + self.local)
            rows = self.lm_head.weight[(cand - self.offset).clamp(0, self.local - 1)].float()
            exact = all_reduce(torch.bmm(rows, flat.float().unsqueeze(-1)).squeeze(-1) * mine)
            scale = _weight_scale(self.lm_head)
            out_v, order = (exact * (scale.float().reshape(()) if scale is not None else 1.0)).topk(r, dim=-1)
            out_ids = cand.gather(-1, order)
        return out_ids.view(*x.shape[:-1], r), out_v.view(*x.shape[:-1], r)

    def argmax_confidence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The argmax and the softmax top-1 probability and margin over the whole vocabulary: each rank's
        log-sum-exp and best two, combined."""
        flat = x.reshape(-1, x.shape[-1])
        logits = self._logits(flat)
        local = logits.float()
        stats = self._gather(torch.cat([torch.logsumexp(local, -1, keepdim=True), local.topk(2, dim=-1).values], -1))
        size = stats.shape[-1] // 3
        stats = stats.view(-1, size, 3)
        lse = torch.logsumexp(stats[..., 0], -1)
        top = stats[..., 1:].reshape(-1, 2 * size).topk(2, dim=-1).values
        probs = torch.exp(top - lse.unsqueeze(-1))
        return self._pick(flat, logits).view(x.shape[:-1]), torch.stack([probs[:, 0], probs[:, 0] - probs[:, 1]], -1)
