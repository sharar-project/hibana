"""Split Qwen3 thinking from content, incrementally.

With thinking enabled the template's generation prompt ends with "<think>\\n",
so the model output starts inside the reasoning block and "</think>" closes it.
With thinking disabled the template already closed an empty block, so the
output is plain content. Both cases, and an explicit "<think>" at the start of
the output, are handled.
"""

from __future__ import annotations

THINK_START = "<think>"
THINK_END = "</think>"


class ReasoningParser:
    def __init__(self, in_reasoning: bool):
        self.in_reasoning = in_reasoning
        self.reasoning = ""
        self.content = ""
        self._pending = ""  # tail that might be the start of a tag
        self._seen_any = False

    def feed(self, delta: str) -> tuple[str, str]:
        """Returns (reasoning_delta, content_delta) for this piece of text."""
        text = self._pending + delta
        self._pending = ""
        r_out, c_out = "", ""
        if not self._seen_any and text:
            self._seen_any = True
            if text.startswith(THINK_START):
                self.in_reasoning = True
                text = text[len(THINK_START):]
            elif THINK_START.startswith(text):
                self._pending = text
                return "", ""
        while text:
            if self.in_reasoning:
                idx = text.find(THINK_END)
                if idx >= 0:
                    r_out += text[:idx]
                    text = text[idx + len(THINK_END):]
                    self.in_reasoning = False
                    continue
                keep = _partial_suffix(text, THINK_END)
                r_out += text[: len(text) - keep]
                self._pending = text[len(text) - keep:]
                text = ""
            else:
                c_out += text
                text = ""
        self.reasoning += r_out
        self.content += c_out
        return r_out, c_out

    def finish(self) -> tuple[str, str]:
        """Flush any held-back tail as whichever side we are on."""
        tail, self._pending = self._pending, ""
        if not tail:
            return "", ""
        if self.in_reasoning:
            self.reasoning += tail
            return tail, ""
        self.content += tail
        return "", tail


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of text that is a proper prefix of tag."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0
