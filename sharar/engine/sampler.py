"""Token sampling: greedy fast path, then temperature / top-k / top-p.

Defaults mirror the checkpoint's generation_config (temperature 1.0, top_k 20,
top_p 0.95) when a request does not set them; temperature 0 means greedy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import secrets

import torch


class InvalidRequest(ValueError):
    """Input rejected before admission or any GPU work."""


@dataclass
class SamplingParams:
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_k: int = 20
    top_p: float = 0.95
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = (248046, 248044)  # <|im_end|>, <|endoftext|>
    ignore_eos: bool = False
    trace: bool = True  # may this request be written to the opt-in trace sink
    # Relaxed speculative acceptance (greedy requests only): a drafted token is
    # also accepted when the target ranks it within its top accept_top_k and
    # its logit is within accept_margin of the argmax (a log-probability
    # ratio). Off unless both are set; outputs then are not the target's exact
    # greedy continuation.
    accept_top_k: int = 1
    accept_margin: float = 0.0

    def validate(self) -> None:
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise InvalidRequest("max_new_tokens must be a positive integer")
        for name, value, valid in (
            ("temperature", self.temperature, lambda x: x >= 0),
            ("top_p", self.top_p, lambda x: 0 < x <= 1),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or not valid(value):
                raise InvalidRequest(f"invalid {name}: {value!r}")
        if type(self.top_k) is not int or self.top_k < -1:
            raise InvalidRequest("top_k must be an integer >= -1")
        if self.seed is not None and (type(self.seed) is not int or not -(2**63) <= self.seed < 2**64):
            raise InvalidRequest("seed must be an integer in [-2**63, 2**64)")
        if type(self.ignore_eos) is not bool:
            raise InvalidRequest("ignore_eos must be a boolean")
        if type(self.trace) is not bool:
            raise InvalidRequest("trace must be a boolean")
        if type(self.accept_top_k) is not int or self.accept_top_k < 1:
            raise InvalidRequest("accept_top_k must be an integer >= 1")
        if type(self.accept_margin) not in (int, float) or not math.isfinite(self.accept_margin) or self.accept_margin < 0:
            raise InvalidRequest("accept_margin must be a finite number >= 0")
        if self.relaxed and not self.greedy:
            raise InvalidRequest("relaxed acceptance (accept_top_k > 1 with accept_margin > 0) requires temperature 0")

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0

    @property
    def relaxed(self) -> bool:
        return self.accept_top_k > 1 and self.accept_margin > 0


class Sampler:
    def __init__(self, device: torch.device):
        self.device = device

    def generator(self, seed: int | None) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(secrets.randbits(64) if seed is None else seed)

    def sample(self, logits: torch.Tensor, params: list[SamplingParams], generators: list[torch.Generator | None]) -> torch.Tensor:
        """logits: [B, V] (bf16 or fp32). Returns [B] int64 token ids."""
        if all(p.greedy for p in params):
            return logits.argmax(dim=-1)
        out = torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)
        for i, p in enumerate(params):
            row = logits[i].float()
            if p.greedy:
                out[i] = row.argmax()
                continue
            row = row / p.temperature
            if p.top_k > 0 and p.top_k < row.numel():
                kth = row.topk(p.top_k).values[-1]
                row = torch.where(row < kth, torch.full_like(row, float("-inf")), row)
            probs = torch.softmax(row, dim=-1)
            if 0.0 < p.top_p < 1.0:
                sorted_probs, sorted_idx = probs.sort(descending=True)
                cum = sorted_probs.cumsum(dim=-1)
                keep = cum - sorted_probs < p.top_p
                sorted_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
                probs = torch.zeros_like(probs).scatter(0, sorted_idx, sorted_probs)
                probs = probs / probs.sum()
            out[i] = torch.multinomial(probs, 1, generator=generators[i])[0]
        return out
