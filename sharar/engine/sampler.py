"""Token sampling: greedy fast path, then temperature / top-k / top-p.

Defaults mirror the checkpoint's generation_config (temperature 1.0, top_k 20,
top_p 0.95) when a request does not set them; temperature 0 means greedy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_k: int = 20
    top_p: float = 0.95
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = (248046, 248044)  # <|im_end|>, <|endoftext|>
    ignore_eos: bool = False

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


class Sampler:
    def __init__(self, device: torch.device):
        self.device = device
        self._generators: dict[int, torch.Generator] = {}

    def _gen(self, seed: int | None) -> torch.Generator | None:
        if seed is None:
            return None
        g = self._generators.get(seed)
        if g is None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
            self._generators[seed] = g
        return g

    def sample(self, logits: torch.Tensor, params: list[SamplingParams]) -> torch.Tensor:
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
            out[i] = torch.multinomial(probs, 1, generator=self._gen(p.seed))[0]
        return out
