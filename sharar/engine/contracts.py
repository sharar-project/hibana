"""Small contracts shared by the target, proposers and execution drivers.

These types import no GPU backend. Feature capture is opt-in; tensor views
and CUDA graph lifetimes are specified in docs/interfaces.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict, runtime_checkable

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
class Capabilities:
    """What a proposer offers beyond the `Proposer` lifecycle (#45). The engine reads these flags
    instead of probing for methods; `check_capabilities` verifies each promised protocol."""
    device_feedback: bool = False   # DeviceFeedback: accepted_prepare / feedback_device / accepted_publish (device acceptance)
    reloadable: bool = False        # Reloadable: reload(snapshot) -> served source, between rounds
    checkpoint_cache: bool = False  # CheckpointCache: bind_checkpoint_cache(pool) and a feature_cache with stats
    cold_payload: bool = False      # ColdPayload: auxiliary_codec() -> (identity, export, import) for parked sessions
    trace_features: bool = False    # TraceFeatures: projected_rows(req, start, count) and a context `window`
    draft_head: bool = False        # DraftHeadChoice: set_draft_head(ids, fmt, rescore) on the head


@dataclass(frozen=True)
class ProposerRequirements:
    name: str
    features: FeatureRequest = FeatureRequest()
    persistent_bytes: int = 0
    scratch_bytes: int = 0
    greedy_only: bool = True
    max_depth: int = 4
    capabilities: Capabilities = Capabilities()


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


@runtime_checkable
class DeviceFeedback(Protocol):
    def accepted_prepare(self, requests: list[Request]): ...
    def feedback_device(self, features, starts, ring_slots, counts, steps, *, norm=None, tokens=None, pages=None) -> None: ...
    def accepted_publish(self, requests: list[Request], emitted: list[list[int]]) -> None: ...


@runtime_checkable
class Reloadable(Protocol):
    def reload(self, snapshot) -> str: ...


@runtime_checkable
class CheckpointCache(Protocol):
    feature_cache: object
    def bind_checkpoint_cache(self, pool) -> None: ...


@runtime_checkable
class ColdPayload(Protocol):
    def auxiliary_codec(self): ...


@runtime_checkable
class TraceFeatures(Protocol):
    window: int
    def projected_rows(self, req: Request, start: int, count: int): ...


PROTOCOLS = {"device_feedback": DeviceFeedback, "reloadable": Reloadable, "checkpoint_cache": CheckpointCache,
             "cold_payload": ColdPayload, "trace_features": TraceFeatures}


def check_capabilities(proposer) -> None:
    """Every capability a proposer declares must come with its protocol's methods."""
    caps = proposer.requirements.capabilities
    for name, protocol in PROTOCOLS.items():
        if getattr(caps, name) and not isinstance(proposer, protocol):
            raise TypeError(f"{proposer.requirements.name} declares {name} without the {protocol.__name__} methods")
    if caps.draft_head and not callable(getattr(getattr(proposer, "head", None), "set_draft_head", None)):
        raise TypeError(f"{proposer.requirements.name} declares draft_head without head.set_draft_head")


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
