"""Training samples from request traces.

A sample is one served request: the prompt followed by the target's own
output, plus the projected context features the drafter attended to while
serving (the last window of the prompt and every committed response
position). Only response positions are supervised. The newest fraction of the
traces is the held-out set, so promotion is judged on what the user did most
recently. Sequences longer than ``max_length`` keep their tail.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from sharar.engine.traces import iter_records, load_features


@dataclass(frozen=True)
class Sample:
    request_id: int
    finish: float
    ids: tuple[int, ...]
    prompt_len: int  # first supervised position
    features: dict | None = None  # {"path", "start", "rows", "dim", "skip"}; start is a position in ids

    @property
    def length(self) -> int:
        return len(self.ids)

    @property
    def output_len(self) -> int:
        return len(self.ids) - self.prompt_len

    def context(self, device) -> torch.Tensor:
        """[length, dim] BF16 projected context; rows never recorded are zero.

        Unrecorded rows lie before the retained prompt window or at the last
        output token, which no supervised anchor attends to.
        """
        if self.features is None:
            raise ValueError(f"request {self.request_id} has no recorded features")
        info = self.features
        rows = load_features(info)[info["skip"]:]
        out = torch.zeros((self.length, rows.shape[1]), dtype=torch.bfloat16, device=device)
        end = min(self.length, info["start"] + rows.shape[0])
        out[info["start"]:end] = rows[:end - info["start"]].to(device)
        return out


def load_samples(directory: str | Path, *, max_length: int = 8192, min_output: int = 16,
                 greedy_only: bool = False, require_features: bool = False) -> list[Sample]:
    samples = []
    for doc in iter_records(directory):
        output = doc["output_ids"]
        if len(output) < min_output:
            continue
        if greedy_only and doc["params"]["temperature"] > 0:
            continue
        info = doc.get("features")
        if require_features and not info:
            continue
        ids = list(doc["prompt_ids"]) + list(output)
        prompt_len = len(doc["prompt_ids"])
        cut = 0
        if len(ids) > max_length:
            # Keep the tail; when the prompt is cut away entirely, the first
            # kept token still serves as the unsupervised context start.
            cut = len(ids) - max_length
            ids, prompt_len = ids[cut:], max(1, prompt_len - cut)
        if len(ids) - prompt_len < min_output or prompt_len < 1:
            continue
        features = None
        if info:
            start = info["start"] - cut
            skip = max(0, -start)
            features = {"path": info["path"], "start": max(0, start), "rows": info["rows"], "dim": info["dim"], "skip": skip}
            if features["rows"] - skip <= 0 or features["start"] > prompt_len:
                if require_features:
                    continue  # the retained window does not reach the response
                features = None
        samples.append(Sample(doc["id"], float(doc.get("finish", 0.0)), tuple(ids), prompt_len, features))
    samples.sort(key=lambda s: (s.finish, s.request_id))
    return samples


def split(samples: list[Sample], holdout_fraction: float = 0.1, min_holdout: int = 1) -> tuple[list[Sample], list[Sample]]:
    """Newest requests are held out; the rest train."""
    if not 0 <= holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in [0, 1)")
    count = max(min_holdout, int(round(len(samples) * holdout_fraction))) if samples else 0
    count = min(count, max(0, len(samples) - 1))
    return samples[:len(samples) - count], samples[len(samples) - count:]


def digest(samples: list[Sample]) -> str:
    h = hashlib.sha256()
    for s in samples:
        h.update(json.dumps([s.request_id, s.prompt_len, list(s.ids)]).encode())
    return h.hexdigest()


def summary(samples: list[Sample]) -> dict:
    return {
        "requests": len(samples),
        "tokens": sum(s.length for s in samples),
        "output_tokens": sum(s.output_len for s in samples),
        "with_features": sum(s.features is not None for s in samples),
        "digest": digest(samples),
    }
