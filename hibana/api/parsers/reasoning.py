"""Split Qwen3 thinking from content, incrementally.

With thinking enabled the template's generation prompt ends with "<think>\\n",
so the model output starts inside the reasoning block and "</think>" closes it.
With thinking disabled the template already closed an empty block, so the
output is plain content. Both cases, and an explicit "<think>" at the start of
the output, are handled.

With tools offered, a model can also write its tool call inside a reasoning
block it never closes, and end the turn there (Flash-Next, twice in
308 Claude Code turns). Reasoning from a "<tool_call>" on is held back: a later
"</think>" releases it as reasoning (a call drafted while thinking), and if the
output ends first while it holds a complete call, it becomes content, where
the tool parser finds the call.
"""

from __future__ import annotations

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_START = "<tool_call>"
TOOL_END = "</tool_call>"


class ReasoningParser:
    def __init__(self, in_reasoning: bool, tools: bool = False):
        self.in_reasoning = in_reasoning
        self.tools = tools
        self.reasoning = ""
        self.content = ""
        self._pending = ""  # reasoning or content not emitted yet: a possible tag start, or a held tool call
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
                hold = text.find(TOOL_START) if self.tools else -1
                if hold < 0:
                    hold = len(text) - max(_partial_suffix(text, THINK_END),
                                           _partial_suffix(text, TOOL_START) if self.tools else 0)
                r_out += text[:hold]
                self._pending = text[hold:]
                text = ""
            else:
                c_out += text
                text = ""
        self.reasoning += r_out
        self.content += c_out
        return r_out, c_out

    def finish(self) -> tuple[str, str]:
        """Flush any held-back tail as whichever side we are on; a complete tool call held in reasoning is content."""
        tail, self._pending = self._pending, ""
        if not tail:
            return "", ""
        if self.in_reasoning and not (tail.startswith(TOOL_START) and TOOL_END in tail):
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
