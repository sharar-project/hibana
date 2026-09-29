"""Draft heads: how a draft body turns its hidden rows into token choices (#45).

Every speculator ends in a vocabulary projection: the MTP heads take the argmax,
DFlash2's selector takes the top-k with values, DSpark takes full logits. The
default is the target's own head (`TargetHead`), which is what every proposer
used before this module existed. A `ShortlistHead` reads fewer bytes: a subset
of the head's rows (a reduced draft vocabulary), a 3-bit copy of them, or both,
with the leading candidates rescored against the target's FP8 rows. The target
still verifies with its own head, so under exact acceptance a cheaper draft head
changes how many drafts are accepted, never what is emitted.
"""

from __future__ import annotations

from typing import Callable

import torch


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


class ShortlistHead:
    """The rows `ids` of an FP8 lm_head (`TrunkLinear`; every row when `ids` is None), as they are
    or as a 3-bit copy (`fmt="w3"`); with `rescore` > 1 the 3-bit logits only pick that many
    candidates and the target head's own FP8 rows choose among them."""

    rows_changed = True

    def __init__(self, lm_head, ids: torch.Tensor | None = None, fmt: str = "fp8", rescore: int = 0):
        from sharar.model.flashnext.linear import TrunkLinear
        if fmt not in ("fp8", "w3"):
            raise ValueError(f"draft head format must be fp8 or w3, not {fmt!r}")
        if not hasattr(lm_head, "weight") or lm_head.weight.dim() != 2:
            raise ValueError("a shortlist draft head reads rows of a TrunkLinear lm_head (Flash-Next); the 27B's NVFP4 head is not supported yet")
        self.fmt = fmt
        self.ids = None if ids is None else ids.to(lm_head.weight.device, torch.int64).unique(sorted=True)
        if fmt == "w3" and self.ids is not None:
            self.ids = _pad_rows(self.ids, 128)
        rows = lm_head if self.ids is None else TrunkLinear(lm_head.weight[self.ids].contiguous(), lm_head.scale)
        self.head, self.w3 = (None, pack_head(rows)) if fmt == "w3" else (rows, None)
        # Shortlist: the 3-bit logits pick `rescore` candidates and the target head's own FP8 rows
        # (already resident) choose among them, so the pick is the full head's whenever its
        # argmax is on the shortlist.
        self.rescore, self.source = (rescore, lm_head) if fmt == "w3" and rescore > 1 else (0, None)

    @property
    def bytes(self) -> int:
        """Bytes this head adds beside the target's lm_head (0 when it is the target's own)."""
        if self.w3 is not None:
            return self.w3.bytes
        return 0 if self.ids is None else self.head.bytes

    def _logits(self, flat: torch.Tensor) -> torch.Tensor:
        if self.w3 is not None:
            from sharar.kernels import w3
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
            scale = self.source.scale.float().reshape(()) if self.source.scale is not None else 1.0
            values, ids = (values * scale).to(torch.bfloat16), cand.gather(-1, order)
        else:
            values, local = logits.topk(k, dim=-1)
            ids = self._global(local)
        return values.view(*x.shape[:-1], k), ids.view(*x.shape[:-1], k)

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


def pack_head(lin, rows_per_chunk: int = 16384):
    """3-bit copy of an FP8 `TrunkLinear` [N, K], dequantized a slice at a time (the whole lm_head
    in FP32 would be a 2.5 GB temporary)."""
    from sharar.kernels import w3
    parts = [w3.pack((lin.weight[r0:r0 + rows_per_chunk].float() * lin.scale.float()) if lin.scale is not None
                     else lin.weight[r0:r0 + rows_per_chunk].float())
             for r0 in range(0, lin.out_features, rows_per_chunk)]
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    return w3.W3Weight(torch.cat([p.planes for p in parts]), torch.cat([p.scales for p in parts]), torch.cat([p.row_scale for p in parts]),
                       lin.out_features, lin.in_features, first.codebook, first.lut)
