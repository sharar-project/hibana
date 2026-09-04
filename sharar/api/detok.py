"""Incremental detokenization with stop strings.

Decodes the full output each time (cheap for the output lengths served
here) and emits only the new suffix, holding back a trailing replacement
character that marks an incomplete UTF-8 sequence. Stop strings are matched
on the visible text; the text is cut at the match and the request finishes.
"""

from __future__ import annotations

from sharar.tokenizer import SnapshotTokenizer


class Detokenizer:
    def __init__(self, tok: SnapshotTokenizer, stop: list[str] | None = None, skip_special: bool = True):
        self.tok = tok
        self.stop = [s for s in (stop or []) if s]
        self.skip_special = skip_special
        self.text = ""
        self.stopped = False

    def update(self, output_ids: list[int]) -> str:
        """Returns the newly visible text since the last call ("" if none)."""
        if self.stopped:
            return ""
        full = self.tok.decode(output_ids, skip_special_tokens=self.skip_special)
        if full.endswith("�"):
            full = full[:-1]
        for s in self.stop:
            idx = full.find(s)
            if idx >= 0:
                full = full[:idx]
                self.stopped = True
                break
        if len(full) <= len(self.text) and not self.stopped:
            return ""
        delta = full[len(self.text):] if full.startswith(self.text) else full
        self.text = full
        return delta
