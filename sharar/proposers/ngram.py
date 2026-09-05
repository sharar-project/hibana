"""Bounded suffix lookup: a feature-free proposer that exercises the contract.

Only complete k-token continuations qualify. Missing matches use the target's
ordinary decode path. This is a contract/reference adapter, not a speed claim.
"""

from __future__ import annotations

import torch

from sharar.engine.contracts import ProposerContext, ProposerRequirements


class NGramProposer:
    requirements = ProposerRequirements(name="ngram")

    def __init__(self, context: ProposerContext, k: int = 3, max_ngram: int = 8, window: int = 4096):
        if not 1 <= k <= 4 or max_ngram < 1 or window < k + 1:
            raise ValueError("invalid n-gram bounds")
        self.context, self.k = context, k
        self.max_ngram, self.window = max_ngram, window

    def candidates(self, req) -> list[int] | None:
        remaining = max(0, self.window - len(req.output_ids))
        history = (req.prompt_ids[-remaining:] if remaining else []) + req.output_ids[-self.window:]
        end = len(history)
        for size in range(min(self.max_ngram, end - self.k), 0, -1):
            suffix = history[-size:]
            for pos in range(end - size - self.k, -1, -1):
                if history[pos:pos + size] == suffix:
                    return history[pos + size:pos + size + self.k]
        return None

    def ready(self, req):
        return self.candidates(req) is not None

    def propose(self, requests, pages, inputs, *, graphs):
        candidates = [self.candidates(r) for r in requests]
        if any(row is None for row in candidates):
            raise ValueError("n-gram proposal lost its matching prefix")
        return torch.tensor(candidates, dtype=torch.int64, device=self.context.device)

    def ingest_prefill(self, req, ids, start, result):
        pass

    def start(self, req, token):
        pass

    def accepted_batch(self, requests, emitted, features):
        pass

    def release(self, req):
        req.proposer_state = None

    def prepare_release(self, req):
        pass

    def capture_graphs(self, log=None):
        pass  # lookup runs on the CPU; it adds no target feature/graph capture
