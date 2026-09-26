"""Reduced draft vocabulary: the ids a draft head may propose.

The MTP head reads the whole lm_head every draft step (248K rows); drafting over
the most likely N ids reads N rows instead. The target still verifies over the
full vocabulary, so this changes speed, never output: a token the draft cannot
propose is one the target emits on its own. Ranking: every special/added token,
then ids in the tokenizer's construction order (byte-level BPE assigns ids in
merge order, a frequency ranking over its training corpus).

The head that reads those rows is `proposers/draft_head.py` (`ShortlistHead`);
`SHARAR_DRAFT_HEAD=w3` gives it a 3-bit copy (0.26 GB instead of the FP8 head's
0.63 GB per draft step, #35).
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
