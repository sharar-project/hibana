"""Reduced draft vocabulary: the ids a draft head may propose.

The MTP head reads the whole lm_head every draft step (248K rows); drafting over
the most likely N ids reads N rows instead. The target still verifies over the
full vocabulary, so this changes speed, never output: a token the draft cannot
propose is one the target emits on its own. Ranking: every special/added token,
then ids in the tokenizer's construction order (byte-level BPE assigns ids in
merge order, a frequency ranking over its training corpus).
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
    """The rows `ids` of an lm_head (`TrunkLinear`, same scale): argmax over them, as global ids."""

    def __init__(self, lm_head, ids: torch.Tensor):
        from sharar.model.flashnext.linear import TrunkLinear
        self.ids = ids.to(lm_head.weight.device, torch.int64).unique(sorted=True)
        self.head = TrunkLinear(lm_head.weight[self.ids].contiguous(), lm_head.scale)

    def argmax(self, x: torch.Tensor) -> torch.Tensor:
        return self.ids[self.head(x).argmax(-1)]
