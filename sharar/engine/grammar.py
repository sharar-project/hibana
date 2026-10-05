"""Constrained output: a request's grammar as token masks, with speculative decoding (#16).

A request may carry a `Constraint` (SamplingParams.constraint): a compiled xgrammar grammar for a JSON schema, any
JSON, or forced tool calls. Its `Request` owns a `GrammarState`, which advances on every emitted token (`push`) and
supplies the allowed-token mask wherever a token is chosen:

- plain sampling (first token after prefill, decode): the mask of the current state (`Sampler.sample`);
- a speculative round: after verification, the drafted tokens are walked through the grammar on the host, one mask
  per verified row (the state after the drafted prefix before it), the target's logits for those rows are masked,
  and acceptance runs on the masked logits. A drafted token the grammar forbids is never the masked argmax (nor has
  target probability), so acceptance stops there and the correction comes from the masked row. Drafting and the
  verify graphs are unchanged.

Reasoning is free: when the output starts inside a thinking block, the grammar applies from the first token after
`</think>`. The grammar allows leading whitespace and, once complete, only the stop tokens.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass

import torch

ROLLBACK = 64  # drafted tokens a walk may take back (a verify block is at most 16)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Constraint:
    """What a request's output must match. `compiled` is shareable; each request builds its own matcher from it."""
    compiled: object  # xgrammar.CompiledGrammar
    kind: str  # "json_schema", "json_object" or "tool_call" (reported, and for tests)
    armed: bool = True  # False: the output starts inside a thinking block; the grammar starts after `think_end`
    think_end: int | None = None


class GrammarState:
    """One request's position in its grammar."""

    def __init__(self, constraint: Constraint):
        import xgrammar as xgr
        self.matcher = xgr.GrammarMatcher(constraint.compiled, max_rollback_tokens=ROLLBACK)
        self.armed = constraint.armed
        self.think_end = constraint.think_end

    def accept(self, token: int) -> bool:
        """Advance by one emitted token; False if the grammar forbids it (the request is then finished)."""
        if not self.armed:
            if token == self.think_end:
                self.armed = True
            return True
        try:
            return self.matcher.accept_token(token)
        except Exception:  # noqa: BLE001 - a grammar failure ends this request, never the engine
            log.exception("grammar matcher failed to accept token %d", token)
            return False

    def fill(self, bitmask: torch.Tensor, row: int) -> None:
        """The allowed next tokens into bitmask[row] (int32 words, bit i of word w is token 32w + i)."""
        if self.armed and not self.matcher.is_terminated():
            self.matcher.fill_next_token_bitmask(bitmask, row)
        else:  # thinking, or the grammar already took its stop token (the request ends with it)
            bitmask[row].fill_(-1)

    def fill_draft(self, drafted: list[int], bitmask: torch.Tensor, row: int) -> None:
        """Masks for a verify block: row + j gets the state after drafted[:j], for j = 0 .. len(drafted). Rows after
        the first drafted token the grammar forbids allow everything (acceptance never reaches them). The state is
        restored afterwards."""
        armed, taken = self.armed, 0
        for j in range(len(drafted) + 1):
            self.fill(bitmask, row + j)
            if j == len(drafted):
                break
            token = drafted[j]
            if not self.armed:
                if token == self.think_end:
                    self.armed = True
                continue
            if not self.matcher.accept_token(token):
                bitmask[row + j + 1:row + len(drafted) + 1].fill_(-1)
                break
            taken += 1
            if self.matcher.is_terminated():  # a drafted stop token: the output would end here
                bitmask[row + j + 1:row + len(drafted) + 1].fill_(-1)
                break
        if taken:
            self.matcher.rollback(taken)
        self.armed = armed


def host_bitmask(rows: int, vocab: int) -> torch.Tensor:
    """An all-allowed host bitmask for `rows` rows, in pinned memory when there is a GPU: a copy from pageable memory
    waits for all queued GPU work (the verify graph) before the host can queue the masking and acceptance."""
    mask = torch.empty((rows, (vocab + 31) // 32), dtype=torch.int32, pin_memory=torch.cuda.is_available())
    return mask.fill_(-1)


def apply_bitmask(logits: torch.Tensor, bitmask: torch.Tensor) -> None:
    """Set the logits of forbidden tokens to -inf in place. logits [rows, V] on any device; bitmask [rows, W] int32."""
    rows, vocab = logits.shape
    # Bytes of the little-endian words: byte c holds tokens 8c .. 8c + 7, bit i of it token 8c + i.
    data = bitmask.contiguous().view(torch.uint8).to(logits.device, non_blocking=True)
    shifts = torch.arange(8, device=logits.device, dtype=torch.uint8)
    allowed = ((data.unsqueeze(-1) >> shifts) & 1).view(rows, -1)[:, :vocab].bool()
    logits.masked_fill_(~allowed, float("-inf"))


def sample_masks(reqs, vocab: int) -> torch.Tensor | None:
    """Host bitmask [len(reqs), W] for one token per request, or None when no request is constrained."""
    if not any(getattr(r, "grammar", None) is not None for r in reqs):
        return None
    bitmask = host_bitmask(len(reqs), vocab)
    for i, r in enumerate(reqs):
        if r.grammar is not None:
            _guarded(r, lambda: r.grammar.fill(bitmask, i), bitmask[i:i + 1])
    return bitmask


def verify_masks(reqs, drafted: list[list[int]], vocab: int) -> torch.Tensor | None:
    """Host bitmask [b * (k + 1), W] for a verify block, or None when no request is constrained."""
    if not any(getattr(r, "grammar", None) is not None for r in reqs):
        return None
    s = len(drafted[0]) + 1
    bitmask = host_bitmask(len(reqs) * s, vocab)
    for i, r in enumerate(reqs):
        if r.grammar is not None:
            _guarded(r, lambda: r.grammar.fill_draft(drafted[i], bitmask, i * s), bitmask[i * s:(i + 1) * s])
    return bitmask


def _guarded(req, fill, rows: torch.Tensor) -> None:
    """Run a mask fill; on a grammar failure leave the request's rows unmasked (its next token then fails the
    grammar and finishes it with reason "grammar") instead of failing the engine step for every request."""
    try:
        fill()
    except Exception:  # noqa: BLE001
        log.exception("grammar mask failed for request %s", getattr(req, "request_id", "?"))
        rows.fill_(-1)


class Grammars:
    """Builds constraints for one tokenizer: xgrammar's tokenizer table and compiler, made on first use, and
    compiled grammars cached by their source."""

    # Whitespace is bounded: before the output (the newlines after </think>) and in each run inside JSON. Unbounded
    # whitespace let a sampled request drift into it until max_tokens (Flash-Next, t=0.7, 2026-10-05).
    WS = "root ::= [ \\t\\n]{0,4}"
    MAX_WS = 24  # one JSON whitespace run: a newline and deep indentation

    def __init__(self, vocab: list[str], vocab_size: int, stop_ids, think_end: int | None, byte_level: bool = True):
        """vocab: token strings by id, byte-level BPE encoded (Qwen) or, for tests, plain text (byte_level=False)."""
        self._vocab, self.vocab_size, self.stop_ids, self.think_end = vocab, vocab_size, list(stop_ids), think_end
        self.byte_level = byte_level
        self._compiler = None
        self._lock = threading.Lock()

    @classmethod
    def from_tokenizer(cls, tokenizer, vocab_size: int, stop_ids) -> "Grammars":
        """`tokenizer` is a `tokenizers.Tokenizer` (byte-level BPE, as Qwen's)."""
        ids = tokenizer.get_vocab(with_added_tokens=True)
        vocab = [""] * (max(ids.values()) + 1)
        for text, i in ids.items():
            vocab[i] = text
        return cls(vocab, vocab_size, stop_ids, tokenizer.token_to_id("</think>"))

    @property
    def compiler(self):
        with self._lock:
            if self._compiler is None:
                import xgrammar as xgr
                kind = xgr.VocabType.BYTE_LEVEL if self.byte_level else xgr.VocabType.RAW
                info = xgr.TokenizerInfo(self._vocab, vocab_type=kind, vocab_size=self.vocab_size,
                                         stop_token_ids=self.stop_ids, add_prefix_space=False)
                self._compiler = xgr.GrammarCompiler(info, cache_enabled=True)
            return self._compiler

    def _compile(self, grammar):
        import xgrammar as xgr
        return self.compiler.compile_grammar(xgr.Grammar.concat(xgr.Grammar.from_ebnf(self.WS), grammar))

    def json_schema(self, schema: dict, armed: bool) -> Constraint:
        import xgrammar as xgr
        try:
            grammar = xgr.Grammar.from_json_schema(json.dumps(schema), any_whitespace=True, max_whitespace_cnt=self.MAX_WS)
        except Exception as exc:  # noqa: BLE001 - xgrammar raises RuntimeError for schemas it cannot compile
            raise ValueError(f"unsupported JSON schema: {exc}") from exc
        return Constraint(self._compile(grammar), "json_schema", armed, self.think_end)

    def json_object(self, armed: bool) -> Constraint:
        import xgrammar as xgr
        # any JSON object (OpenAI's json_object), with the same whitespace bound
        grammar = xgr.Grammar.from_json_schema(json.dumps({"type": "object"}), any_whitespace=True, strict_mode=False,
                                               max_whitespace_cnt=self.MAX_WS)
        return Constraint(self._compile(grammar), "json_object", armed, self.think_end)

    def tool_call(self, tools: list[dict], name: str | None, armed: bool) -> Constraint:
        """Qwen3.8's XML tool calls (api/parsers/tools.py): one call to `name`, or one or more calls to any tool.
        Function and parameter names come from the schemas; values are free text without `</`."""
        import xgrammar as xgr
        calls = []
        for tool in tools:
            fn = tool.get("function", tool)
            if name is not None and fn.get("name") != name:
                continue
            params = list(((fn.get("parameters") or {}).get("properties") or {}))
            pname = " | ".join(_ebnf_string(p) for p in params) or '""'
            param = (f'"<parameter=" ({pname}) ">\\n" value "\\n</parameter>\\n"') if params else '""'
            calls.append(f'"<tool_call>\\n<function={_ebnf_body(fn["name"])}>\\n" ({param})* "</function>\\n</tool_call>"')
        if not calls:
            raise ValueError(f"tool_choice names {name!r}, which is not among the tools")
        call = " | ".join(f"({c})" for c in calls)
        root = "root ::= call" if name is not None else 'root ::= call ([ \\n]* call)*'
        ebnf = f'{root}\ncall ::= {call}\nvalue ::= ([^<] | "<" [^/])*'
        return Constraint(self._compile(xgr.Grammar.from_ebnf(ebnf)), "tool_call", armed, self.think_end)


def _ebnf_body(text: str) -> str:
    return json.dumps(text)[1:-1]


def _ebnf_string(text: str) -> str:
    return json.dumps(text)
