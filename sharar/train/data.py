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

from sharar.engine.traces import iter_records, load_features, load_topk


@dataclass(frozen=True)
class Sample:
    request_id: int
    finish: float
    ids: tuple[int, ...]
    prompt_len: int  # first supervised position
    features: dict | None = None  # {"path", "start", "rows", "dim", "skip"}; start is a position in ids
    topk: dict | None = None  # {"ids_path", "logits_path", "start", "rows", "k", "skip"}; row p is the target's top-k over the token at p + 1
    rounds: tuple = ()  # (anchor position, proposals y[k], target argmax rows z[k+1]) per greedy verification round
    relaxed: tuple = ()  # positions in ids emitted under relaxed acceptance (not the target's argmax): masked as labels

    @property
    def length(self) -> int:
        return len(self.ids)

    @property
    def output_len(self) -> int:
        return len(self.ids) - self.prompt_len

    def teacher(self, device, k: int = 16):
        """([length, k] int64 ids, [length, k] float32 logits, [length] bool valid) aligned to
        positions: row p describes the token at p + 1. Rows never recorded are invalid."""
        ids = torch.zeros((self.length, k), dtype=torch.int64, device=device)
        logits = torch.zeros((self.length, k), dtype=torch.float32, device=device)
        valid = torch.zeros(self.length, dtype=torch.bool, device=device)
        if self.topk is None:
            return ids, logits, valid
        info = self.topk
        rec_ids, rec_logits = load_topk(info)
        rec_ids, rec_logits = rec_ids[info["skip"]:], rec_logits[info["skip"]:]
        end = min(self.length, info["start"] + rec_ids.shape[0])
        n = end - info["start"]
        if n > 0 and rec_ids.shape[1] == k:
            ids[info["start"]:end] = rec_ids[:n].to(device=device, dtype=torch.int64)
            logits[info["start"]:end] = rec_logits[:n].to(device=device, dtype=torch.float32)
            valid[info["start"]:end] = True
        return ids, logits, valid

    def on_policy(self, device):
        """Verification rounds as (anchors [R], proposals [R, k], target rows [R, k + 1]) tensors.

        Row j of a round's target rows is the target's argmax after the block
        prefix through row j (row 0 is the anchor token), so it labels block
        position j + 1 given predecessor block token j, including positions
        after the first rejected proposal.
        """
        if not self.rounds:
            return torch.empty(0, dtype=torch.int64, device=device), None, None
        anchors = torch.tensor([r[0] for r in self.rounds], dtype=torch.int64, device=device)
        y = torch.tensor([r[1] for r in self.rounds], dtype=torch.int64, device=device)
        z = torch.tensor([r[2] for r in self.rounds], dtype=torch.int64, device=device)
        return anchors, y, z

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
        topk = None
        tinfo = doc.get("topk")
        if tinfo:
            start = tinfo["start"] - cut
            skip = max(0, -start)
            if tinfo["rows"] - skip > 0:
                topk = {"ids_path": tinfo["ids_path"], "logits_path": tinfo["logits_path"], "start": max(0, start),
                        "rows": tinfo["rows"], "k": tinfo["k"], "skip": skip}
        rounds = []
        relaxed = []
        spec = doc.get("spec")
        if spec:
            k = spec["k"]
            position = len(doc["prompt_ids"])  # the pending token of the first round is the first output token
            for r in spec["rounds"]:
                anchor = position - cut
                if r["z"] is not None and len(r["y"]) == k and len(r["z"]) == k + 1 and 0 <= anchor < len(ids) - 1:
                    rounds.append((anchor, tuple(r["y"]), tuple(r["z"])))
                rx = r.get("rx")
                if isinstance(rx, list):  # block offsets emitted under the relaxed rule -> absolute positions in ids
                    relaxed.extend(anchor + 1 + j for j in rx if 0 <= anchor + 1 + j < len(ids))
                position += r["a"] + 1
        samples.append(Sample(doc["id"], float(doc.get("finish", 0.0)), tuple(ids), prompt_len, features, topk, tuple(rounds), tuple(relaxed)))
    samples.sort(key=lambda s: (s.finish, s.request_id))
    return samples


HOLDOUT_TOKENS = 20_000  # about 4,000 verification rounds: resolves a 1% change in a few minutes


def split(samples: list[Sample], holdout_tokens: int = HOLDOUT_TOKENS, min_requests: int = 4,
          max_fraction: float = 0.2) -> tuple[list[Sample], list[Sample]]:
    """Newest requests are held out until they cover ``holdout_tokens`` output tokens.

    Tokens per round is measured over rounds, so the held-out budget is output
    tokens, not a share of requests: at least ``min_requests`` requests, never
    more than ``max_fraction`` of them (and always at least one), so small
    corpora still train and large ones do not waste data.
    """
    if holdout_tokens < 0 or min_requests < 1 or not 0 < max_fraction < 1:
        raise ValueError("invalid holdout budget")
    if not samples:
        return [], []
    cap = max(1, int(len(samples) * max_fraction))
    # Held-out requests must be exact: a relaxed request's outputs contain the
    # drafter's own picks, which would inflate its replayed acceptance. Relaxed
    # requests train (their relaxed positions are masked) but are never held out.
    held, train, tokens = [], [], 0
    for sample in reversed(samples):
        full = len(held) >= cap or (tokens >= holdout_tokens and len(held) >= min_requests)
        if not full and not sample.relaxed:
            held.append(sample)
            tokens += sample.output_len
        else:
            train.append(sample)
    if not train and held:  # a lone sample trains; run() needs two anyway
        train.append(held.pop())
    train.reverse()
    held.reverse()
    return train, held


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
        "with_topk": sum(s.topk is not None for s in samples),
        "rounds": sum(len(s.rounds) for s in samples),
        "digest": digest(samples),
    }
