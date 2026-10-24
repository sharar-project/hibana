"""Tokenizer for the pinned snapshot, via the `tokenizers` library (no transformers)."""

from __future__ import annotations

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

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)
