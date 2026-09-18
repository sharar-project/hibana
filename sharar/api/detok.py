"""Incremental detokenization with stop strings.

The tokenizer runs on a bounded window, not the whole output: ``ids[prefix:read)``
are tokens whose text was already emitted (kept as context so byte-level
merges across the boundary decode the same way), and ``ids[read:]`` are the
new tokens. The new suffix is the window's decode minus the decode of its
emitted part, so each round costs the window, not the output length. A
trailing replacement character marks an incomplete UTF-8 sequence and holds
the window back until the next token completes it. Stop strings are matched
on the visible text; the text is cut at the match and the request finishes.
"""

from __future__ import annotations

from sharar.tokenizer import SnapshotTokenizer

REPLACEMENT = "�"


class Detokenizer:
    def __init__(self, tok: SnapshotTokenizer, stop: list[str] | None = None, skip_special: bool = True):
        self.tok = tok
        self.stop = [s for s in (stop or []) if s]
        self.skip_special = skip_special
        self.text = ""
        self.stopped = False
        self._prefix = 0  # ids before this are never decoded again
        self._read = 0  # ids in [prefix, read) were emitted; ids from here on are new

    def _decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=self.skip_special) if ids else ""

    def update(self, output_ids: list[int]) -> str:
        """Returns the newly visible text since the last call ("" if none)."""
        if self.stopped or len(output_ids) <= self._read:
            return ""
        emitted = self._decode(output_ids[self._prefix:self._read])
        window = self._decode(output_ids[self._prefix:])
        if window.endswith(REPLACEMENT) or len(window) <= len(emitted) or not window.startswith(emitted):
            return ""  # incomplete character or no new text yet: retry with more tokens
        delta = window[len(emitted):]
        self._prefix, self._read = self._read, len(output_ids)
        full = self.text + delta
        for s in self.stop:
            idx = full.find(s, max(0, len(self.text) - len(s) + 1))
            if idx >= 0:
                full = full[:idx]
                self.stopped = True
                break
        delta = full[len(self.text):]
        self.text = full
        return delta
