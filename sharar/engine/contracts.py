"""Small contracts shared by the target, proposers and execution drivers.

These types import no GPU backend. Feature capture is opt-in; tensor views
and CUDA graph lifetimes are specified in docs/interfaces.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict

if TYPE_CHECKING:
    import torch
    from sharar.engine.request import Request
    from sharar.engine.sampler import SamplingParams
    from sharar.engine.rejection import Proposal


@dataclass(frozen=True)
class FeatureRequest:
    final_norm: bool = False
    layers: tuple[int, ...] = ()  # post-block residual stream, before next norm


class ForwardResult(TypedDict):
    norm: torch.Tensor  # bf16 [T, H], final Gemma RMSNorm output used by lm_head
    pre_norm: NotRequired[torch.Tensor]
    features: NotRequired[dict[int, torch.Tensor]]  # requested bf16 [T, H] streams
    input_ids: NotRequired[torch.Tensor]
    positions: NotRequired[torch.Tensor]
    embed: NotRequired[torch.Tensor]
    layers: NotRequired[list[tuple[torch.Tensor, torch.Tensor]]]


@dataclass(frozen=True)
class ProposerRequirements:
    name: str
    features: FeatureRequest = FeatureRequest()
    persistent_bytes: int = 0
    scratch_bytes: int = 0
    greedy_only: bool = True
    max_depth: int = 4


@dataclass(frozen=True)
class ProposerContext:
    device: torch.device
    max_batch: int
    max_context: int


@dataclass
class StepInputs:
    tokens: torch.Tensor  # [B] pending target input, int64
    positions: torch.Tensor  # [B] absolute processed-prefix length, int64
    slots: torch.Tensor  # [B] int32


@dataclass
class VerifiedFeatures:
    norm: torch.Tensor | None = None  # [B, width, H]; absent if not requested
    layers: dict[int, torch.Tensor] | None = None


class Proposer(Protocol):
    k: int
    context: ProposerContext
    requirements: ProposerRequirements

    def ready(self, req: Request) -> bool: ...
    def ingest_prefill(self, req: Request, ids: list[int], start: int, result: ForwardResult) -> None: ...
    def start(self, req: Request, token: int) -> None: ...
    def propose(self, requests: list[Request], pages: torch.Tensor, inputs: StepInputs, *, graphs: bool) -> torch.Tensor | Proposal: ...
    def accepted_batch(self, requests: list[Request], emitted: list[list[int]], features: VerifiedFeatures) -> None:
        """Consume stop-truncated features after every target row is committed."""
        ...
    def prepare_release(self, req: Request) -> None:
        """Attach optional draft metadata before target ownership transfers."""
        ...
    def release(self, req: Request) -> None: ...
    def capture_graphs(self, log=None) -> None: ...


class ExecutionBackend(Protocol):
    """Scheduler-facing operations; graph buffers and proposer state stay private."""

    max_batch: int

    def validate_request(self, prompt_ids: list[int], params: SamplingParams) -> None: ...
    def admission_reserve(self, req: Request, limit: int) -> int: ...
    def can_admit(self, prompt_ids: list[int], reserve_tokens: int = 0) -> bool: ...
    def admit(self, prompt_ids: list[int], params: SamplingParams, req: Request) -> Request: ...
    def prefill_batch(self, chunks: list[tuple[Request, int]]) -> None: ...
    def decode_step(self, reqs: list[Request]) -> list[int]: ...
    def retract(self, req: Request) -> None: ...
    def release(self, req: Request) -> None: ...
    def cache_stats(self) -> dict | None: ...
