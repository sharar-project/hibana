"""Per-step forward context handed to every layer.

Prefill runs one fresh sequence (pages for every token, one state slot).
Decode runs a batch of B requests, one new token each.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sharar import config
from sharar.memory.kv_pool import KvPool
from sharar.memory.state_pool import StatePool


@dataclass
class ForwardContext:
    mode: str  # "prefill" | "decode"
    kv_pool: KvPool | None
    state_pool: StatePool | None
    positions: torch.Tensor  # [T] prefill or [B] decode, int64
    # prefill
    slot: int = 0
    pages: torch.Tensor | None = None  # [T] int32, KV write location per token
    seq_len: int = 0
    # decode
    slots: torch.Tensor | None = None  # [B] int32 committed state slot per row
    out_pages: torch.Tensor | None = None  # [B] int32 KV write location for the new token
    decode_wrapper: object = None  # planned FlashInfer BatchDecodeWithPagedKVCacheWrapper
    batch_size: int = 0
    # Parity capture keeps every layer's (hidden, residual); the residual is
    # then cloned instead of updated in place.
    capture: bool = False

    @property
    def query_start_loc(self) -> torch.Tensor:
        return torch.tensor([0, self.seq_len], dtype=torch.int32, device=self.positions.device)

    @property
    def slot_tensor(self) -> torch.Tensor:
        return torch.tensor([self.slot], dtype=torch.int32, device=self.positions.device)


def scratch_prefill_context(t: int, device: torch.device) -> ForwardContext:
    """Context for a stand-alone prefill without pools (parity tools)."""
    return ForwardContext(
        mode="prefill",
        kv_pool=None,
        state_pool=None,
        positions=torch.arange(t, dtype=torch.int64, device=device),
        seq_len=t,
    )
