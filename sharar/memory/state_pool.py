"""Conv and SSM state slots for the 48 Gated DeltaNet layers.

Every request owns one committed slot. Scratch slots are reserved for
speculative verification (Phase 7). Acceptance either copies a retained
snapshot or reconstructs directly into validated private slots. The tensors are per layer so the
kernels' `initial_state_indices` address the slot directly.
"""

from __future__ import annotations

import torch

from sharar import config

CONV_DIM = 2 * config.LINEAR_NUM_KEY_HEADS * config.LINEAR_KEY_HEAD_DIM + config.LINEAR_NUM_VALUE_HEADS * config.LINEAR_VALUE_HEAD_DIM
CONV_STATE_LEN = config.LINEAR_CONV_KERNEL - 1


class StatePool:
    def __init__(self, num_slots: int, device: torch.device, num_scratch: int = 0):
        self.num_slots = num_slots
        self.device = device
        self.layer_ids = [i for i in range(config.NUM_LAYERS) if (i + 1) % config.FULL_ATTENTION_INTERVAL != 0]
        total = num_slots + num_scratch
        n_layers = len(self.layer_ids)
        # One contiguous tensor per kind, indexed [layer_index, slot, ...], with a
        # per-layer view for the kernels; clearing or copying a slot across all
        # layers is then one strided copy instead of one launch per layer.
        self.conv_all = torch.zeros((n_layers, total, CONV_DIM, CONV_STATE_LEN), dtype=torch.bfloat16, device=device)
        self.ssm_all = torch.zeros(
            (n_layers, total, config.LINEAR_NUM_VALUE_HEADS, config.LINEAR_VALUE_HEAD_DIM, config.LINEAR_KEY_HEAD_DIM),
            dtype=torch.float32,
            device=device,
        )
        self.conv = {lid: self.conv_all[j] for j, lid in enumerate(self.layer_ids)}
        self.ssm = {lid: self.ssm_all[j] for j, lid in enumerate(self.layer_ids)}
        # Slot 0 is the dummy slot for padded batch rows; scratch slots follow the committed ones.
        self._free = list(range(num_slots - 1, 0, -1))
        self.scratch = list(range(num_slots, total))

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def bytes_per_slot(self) -> int:
        per_layer = CONV_DIM * CONV_STATE_LEN * 2 + config.LINEAR_NUM_VALUE_HEADS * config.LINEAR_VALUE_HEAD_DIM * config.LINEAR_KEY_HEAD_DIM * 4
        return per_layer * len(self.layer_ids)

    def alloc(self) -> int:
        if not self._free:
            raise MemoryError("state pool exhausted")
        slot = self._free.pop()
        self.clear(slot)
        return slot

    def free(self, slot: int) -> None:
        self._free.append(slot)

    def clear(self, slot: int) -> None:
        self.conv_all[:, slot].zero_()
        self.ssm_all[:, slot].zero_()

    def copy_slot(self, src: int, dst: int) -> None:
        self.conv_all[:, dst].copy_(self.conv_all[:, src])
        self.ssm_all[:, dst].copy_(self.ssm_all[:, src])
