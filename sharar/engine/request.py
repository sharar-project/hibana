"""Request lifecycle shared by the executor and the scheduler."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import torch

from sharar.engine.sampler import SamplingParams


class State(str, Enum):
    QUEUED = "queued"
    PREFILL = "prefill"  # admitted; prompt tokens still being prefilled in chunks
    RUNNING = "running"  # first token produced; decoding
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass
class Request:
    prompt_ids: list[int]
    params: SamplingParams
    request_id: int = 0
    state: State = State.QUEUED
    output_ids: list[int] = field(default_factory=list)
    # resources
    slot: int = -1
    pages: torch.Tensor | None = None  # all KV pages: cached prefix pages then own pages
    prefix_len: int = 0  # tokens served from the cache
    prefill_pos: int = 0  # prompt tokens whose state/KV exist (cached + chunks done)
    match: object = None
    checkpoints: dict[int, int] = field(default_factory=dict)  # prefix length -> snapshot slot
    released: bool = False
    # progress and timing
    finished: bool = False
    finish_reason: str = ""
    cancel_requested: bool = False
    t_arrival: float = field(default_factory=time.perf_counter)
    t_admit: float = 0.0
    t_first_token: float = 0.0
    t_finish: float = 0.0
    prefill_seconds: float = 0.0
    stream_cursor: int = 0

    @property
    def context_len(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def prefill_done(self) -> bool:
        return self.prefill_pos >= len(self.prompt_ids)

    @property
    def ttft(self) -> float:
        return self.t_first_token - self.t_arrival if self.t_first_token else float("nan")

    def push(self, token: int) -> None:
        if not self.output_ids:
            self.t_first_token = time.perf_counter()
            self.state = State.RUNNING
        self.output_ids.append(token)
        if not self.params.ignore_eos and token in self.params.stop_token_ids:
            self._finish("stop")
        elif len(self.output_ids) >= self.params.max_new_tokens:
            self._finish("length")

    def _finish(self, reason: str) -> None:
        self.finished = True
        self.finish_reason = reason
        self.state = State.FINISHED
        self.t_finish = time.perf_counter()

    def cancel(self) -> None:
        self.cancel_requested = True

    def new_tokens(self) -> list[int]:
        """Tokens produced since the last call (for streaming)."""
        out = self.output_ids[self.stream_cursor :]
        self.stream_cursor = len(self.output_ids)
        return out
