"""Tokenizer for the pinned snapshot, via the `tokenizers` library (no transformers)."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer


class SnapshotTokenizer:
    def __init__(self, snap: Path):
        self._tok = Tokenizer.from_file(str(snap / "tokenizer.json"))

    def encode(self, text: str) -> list[int]:
        # The chat template already carries every special token, so never add any.
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """`encode`, and each token's (start, end) character span in `text`."""
        enc = self._tok.encode(text, add_special_tokens=False)
        return enc.ids, enc.offsets

    def without_specials(self, ids: set[int]) -> SnapshotTokenizer:
        """This tokenizer with the added tokens `ids` matched in text no more: their strings encode as text.

        The entries stay, their content rewritten to a string no real text carries: dropping them
        re-registers the added tokens by position and shifts every later one's id (Flash-Next's
        `<think>` would encode as `<|file_sep|>`, three places down, and every special-prompt chat
        render would reach the model as a different token sequence)."""
        spec = json.loads(self._tok.to_str())
        for t in spec["added_tokens"]:
            if t["id"] in ids:
                t["content"] = f"\ue000disabled-{t['id']}\ue001"
        out = SnapshotTokenizer.__new__(SnapshotTokenizer)
        out._tok = Tokenizer.from_str(json.dumps(spec))
        return out

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)
