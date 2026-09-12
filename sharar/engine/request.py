"""Request lifecycle shared by the executor and the scheduler."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import torch

from sharar.engine.sampler import SamplingParams
from sharar.memory.request_state import RequestState


class State(str, Enum):
    QUEUED = "queued"
    PREFILL = "prefill"  # admitted; prompt tokens still being prefilled in chunks
    RUNNING = "running"  # first token produced; decoding
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass(eq=False)
class Request(RequestState):
    prompt_ids: list[int]
    params: SamplingParams
    request_id: int = 0
    state: State = State.QUEUED
    output_ids: list[int] = field(default_factory=list)
    # resources
    prefill_pos: int = 0  # prompt tokens whose state/KV exist (cached + chunks done)
    replay_len: int = 0  # emitted outputs included in prefill after retraction
    generator: torch.Generator | None = None  # request-owned, survives retraction
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
    # speculative decoding bookkeeping (engine/spec.py)
    proposer_state: object = None  # private to the selected proposer
    spec_rounds: int = 0  # verify rounds this request took part in
    spec_proposed: int = 0  # draft tokens proposed for this request
    spec_accepted: int = 0  # draft tokens accepted for this request
    spec_emitted: int = 0  # outputs actually emitted by verify rounds (excludes prefill)
    # opt-in trace sink (engine/traces.py): per-round records while recording, else None
    trace_rounds: list[dict] | None = None
    trace_features: object = None  # FeatureRecorder while recording projected draft context, else None

    @property
    def prefill_len(self) -> int:
        return len(self.prompt_ids) + self.replay_len

    @property
    def prefill_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids[:self.replay_len]

    @property
    def context_len(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def prefill_done(self) -> bool:
        return self.prefill_pos >= self.prefill_len

    @property
    def ttft(self) -> float:
        return self.t_first_token - self.t_arrival if self.t_first_token else float("nan")

    def push(self, token: int) -> None:
        if self.finished:
            return
        if not self.output_ids:
            self.t_first_token = time.perf_counter()
            self.state = State.RUNNING
        self.output_ids.append(token)
        if not self.params.ignore_eos and token in self.params.stop_token_ids:
            self._finish("stop")
        elif len(self.output_ids) >= self.params.max_new_tokens:
            self._finish("length")

    def _finish(self, reason: str) -> None:
        if self.finished:
            return
        self.finished = True
        self.finish_reason = reason
        self.state = State.CANCELLED if reason == "cancelled" else State.FINISHED
        self.t_finish = time.perf_counter()

    def cancel(self) -> None:
        self.cancel_requested = True

    def new_tokens(self) -> list[int]:
        """Tokens produced since the last call (for streaming)."""
        out = self.output_ids[self.stream_cursor :]
        self.stream_cursor = len(self.output_ids)
        return out
