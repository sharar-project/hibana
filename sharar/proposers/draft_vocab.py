"""Reduced draft vocabulary: the ids a draft head may propose.

The MTP head reads the whole lm_head every draft step (248K rows); drafting over
the most likely N ids reads N rows instead. The target still verifies over the
full vocabulary, so this changes speed, never output: a token the draft cannot
propose is one the target emits on its own. Ranking: every special/added token,
then ids in the tokenizer's construction order (byte-level BPE assigns ids in
merge order, a frequency ranking over its training corpus).

`SHARAR_DRAFT_HEAD=w3` gives the draft head a 3-bit copy of those rows (or of the
whole head): 0.26 GB instead of the FP8 head's 0.63 GB per draft step (#35).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def draft_vocab(snapshot: Path, size: int, vocab_size: int) -> torch.Tensor:
    tok = json.loads((Path(snapshot) / "tokenizer.json").read_text())
    specials = [a["id"] for a in tok.get("added_tokens", []) if a["id"] < vocab_size]
    ranked = list(dict.fromkeys(specials + list(range(vocab_size))))
    return torch.tensor(sorted(ranked[:size]), dtype=torch.int64)


class ReducedHead:
    """A draft-only lm_head: the rows `ids` of the target's head (every row when `ids` is None),
    as they are or as a 3-bit copy (`fmt="w3"`, `sharar/kernels/w3.py`); argmax over them, as
    global ids. The target verifies with its own full head, so under exact acceptance a cheaper
    draft head changes how many drafts are accepted, never what is emitted."""

    def __init__(self, lm_head, ids: torch.Tensor | None = None, fmt: str = "fp8", rescore: int = 0):
        from sharar.model.flashnext.linear import TrunkLinear
        if fmt not in ("fp8", "w3"):
            raise ValueError(f"draft head format must be fp8 or w3, not {fmt!r}")
        self.fmt = fmt
        self.ids = None if ids is None else ids.to(lm_head.weight.device, torch.int64).unique(sorted=True)
        if fmt == "w3" and self.ids is not None:
            self.ids = _pad_rows(self.ids, 128)
        rows = lm_head if self.ids is None else TrunkLinear(lm_head.weight[self.ids].contiguous(), lm_head.scale)
        self.head, self.w3 = (None, pack_head(rows)) if fmt == "w3" else (rows, None)
        # Shortlist: the 3-bit logits pick `rescore` candidates and the target head's own rows
        # (FP8, already resident) choose among them, so the pick is the full head's whenever its
        # argmax is on the shortlist.
        self.rescore, self.source = (rescore, lm_head) if fmt == "w3" and rescore > 1 else (0, None)

    @property
    def bytes(self) -> int:
        """Bytes this head adds beside the target's lm_head (0 when it is the target's own)."""
        if self.w3 is not None:
            return self.w3.bytes
        return 0 if self.ids is None else self.head.bytes

    def argmax(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        if self.w3 is not None:
            from sharar.kernels import w3
            logits = torch.cat([w3.w3_linear(part.contiguous(), self.w3) for part in flat.split(w3.MAX_ROWS)])
            if self.rescore:
                return self._rescore(flat, logits)
        else:
            logits = self.head(flat)
        pick = logits.argmax(-1)
        return pick if self.ids is None else self.ids[pick]

    def _rescore(self, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        cand = logits.topk(self.rescore, dim=-1).indices  # [M, R] rows of this head
        cand = cand if self.ids is None else self.ids[cand]  # global ids
        rows = self.source.weight[cand].float()  # [M, R, K] of the target's head
        exact = torch.bmm(rows, x.float().unsqueeze(-1)).squeeze(-1)  # [M, R]; the weight scale is common to all rows
        return cand.gather(-1, exact.argmax(-1, keepdim=True)).squeeze(-1)


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
