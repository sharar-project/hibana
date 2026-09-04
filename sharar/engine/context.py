"""Per-step forward context handed to every layer.

Prefill runs one fresh sequence (pages for every token, one state slot).
Decode runs a batch of B requests, one new token each.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sharar import config
from sharar.memory.kv_pool import KvPool
from sharar.memory.state_pool import StatePool


@dataclass
class ForwardContext:
    mode: str  # "prefill" | "decode" | "verify"
    kv_pool: KvPool | None
    state_pool: StatePool | None
    positions: torch.Tensor  # [T] prefill or [B] decode, int64
    # prefill
    slot: int = 0
    pages: torch.Tensor | None = None  # [T] int32, KV write location for each new token
    seq_len: int = 0  # number of new tokens in this prefill
    # prefix reuse: the request's committed slot already holds the state after
    # `prefix_len` tokens and `prefix_pages` hold their KV
    prefix_len: int = 0
    prefix_pages: torch.Tensor | None = None
    paged_prefill_wrapper: object = None  # planned FlashInfer paged prefill (prefix case)
    # GDN state checkpoints requested at these absolute token counts (multiples
    # of 64, inside the new tokens); layer i writes them to checkpoint_slots
    checkpoint_positions: list[int] = field(default_factory=list)
    checkpoint_slots: list[int] = field(default_factory=list)
    # batched prefill (several requests' chunks in one forward): per-sequence
    # metadata; when set, `slot`/`prefix_len` above are unused
    cu_seqlens: torch.Tensor | None = None  # int32 [B+1] over the new tokens
    seq_slots: torch.Tensor | None = None  # int32 [B]
    seq_prefix_lens: list[int] = field(default_factory=list)  # tokens already in state/pages per seq
    seq_lens: list[int] = field(default_factory=list)  # new tokens per seq
    checkpoint_seqs: list[int] = field(default_factory=list)  # sequence index of each checkpoint

    # speculative verify: k+1 tokens per sequence; GDN runs the recurrent
    # kernel from the committed state without updating it, recording the state
    # after every token in inter_ssm[layer][scratch, step]
    verify_steps: int = 0
    verify_scratch: torch.Tensor | None = None  # int32 [B]
    inter_ssm: dict = field(default_factory=dict)  # layer -> [num_scratch, steps, HV, V, K] fp32
    verify_conv_windows: dict = field(default_factory=dict)  # layer -> [B, steps, dim, 3] bf16 (output)

    @property
    def batched(self) -> bool:
        return self.cu_seqlens is not None
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
