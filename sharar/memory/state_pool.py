"""Conv and SSM state slots for the 48 Gated DeltaNet layers.

Every request owns one committed slot. Scratch slots are reserved for
speculative verification (Phase 7). Acceptance either copies a retained
snapshot or reconstructs directly into validated private slots. The tensors are per layer so the
kernels' `initial_state_indices` address the slot directly.
"""

from __future__ import annotations

import torch

from sharar.target import DEFAULT_TARGET, Target

# The default target's convolution geometry, for tools written against the 27B.
CONV_DIM = DEFAULT_TARGET.conv_dim
CONV_STATE_LEN = DEFAULT_TARGET.conv_state_len


class StatePool:
    def __init__(self, num_slots: int, device: torch.device, num_scratch: int = 0, target: Target | None = None):
        t = self.target = target or DEFAULT_TARGET
        self.num_slots = num_slots
        self.device = device
        self.layer_ids = list(t.gdn_layer_ids)
        self.conv_dim, self.conv_state_len = t.conv_dim, t.conv_state_len
        total = num_slots + num_scratch
        n_layers = len(self.layer_ids)
        # One contiguous tensor per kind, indexed [layer_index, slot, ...], with a
        # per-layer view for the kernels; clearing or copying a slot across all
        # layers is then one strided copy instead of one launch per layer.
        self.conv_all = torch.zeros((n_layers, total, t.conv_dim, t.conv_state_len), dtype=torch.bfloat16, device=device)
        self.ssm_all = torch.zeros(
            (n_layers, total, t.linear_num_value_heads, t.linear_value_head_dim, t.linear_key_head_dim),
            dtype=torch.float32,
            device=device,
        )
        self.conv = {lid: self.conv_all[j] for j, lid in enumerate(self.layer_ids)}
        self.ssm = {lid: self.ssm_all[j] for j, lid in enumerate(self.layer_ids)}
        # Slot 0 is the dummy slot for padded batch rows; scratch slots follow the committed ones.
        self._free = list(range(num_slots - 1, 0, -1))
        self.scratch = list(range(num_slots, total))
        self.on_free = []  # optional auxiliary owners tied to a committed slot

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def bytes_per_slot(self) -> int:
        return self.target.state_bytes_per_layer * len(self.layer_ids)

    def alloc(self) -> int:
        if not self._free:
            raise MemoryError("state pool exhausted")
        slot = self._free.pop()
        self.clear(slot)
        return slot

    def is_allocated(self, slot: int) -> bool:
        """Whether a non-dummy committed slot has a current owner (CPU only)."""
        return type(slot) is int and 0 < slot < self.num_slots and slot not in self._free

    def free(self, slot: int) -> None:
        for callback in self.on_free:
            callback(slot)
        self._free.append(slot)

    def clear(self, slot: int) -> None:
        self.conv_all[:, slot].zero_()
        self.ssm_all[:, slot].zero_()

    def copy_slot(self, src: int, dst: int) -> None:
        self.conv_all[:, dst].copy_(self.conv_all[:, src])
        self.ssm_all[:, dst].copy_(self.ssm_all[:, src])
