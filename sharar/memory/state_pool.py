"""Conv and SSM state slots for the 48 Gated DeltaNet layers.

Every request owns one committed slot. Scratch slots are reserved for
speculative verification (Phase 7): a verify step advances scratch and the
accept step copies scratch into committed. The tensors are per layer so the
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
        self.conv = {
            i: torch.zeros((total, CONV_DIM, CONV_STATE_LEN), dtype=torch.bfloat16, device=device) for i in self.layer_ids
        }
        self.ssm = {
            i: torch.zeros(
                (total, config.LINEAR_NUM_VALUE_HEADS, config.LINEAR_VALUE_HEAD_DIM, config.LINEAR_KEY_HEAD_DIM),
                dtype=torch.float32,
                device=device,
            )
            for i in self.layer_ids
        }
        # Slot 0 is the dummy slot for padded batch rows; scratch slots follow the committed ones.
        self._free = list(range(num_slots - 1, 0, -1))
        self.scratch = list(range(num_slots, total))

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
        for i in self.layer_ids:
            self.conv[i][slot].zero_()
            self.ssm[i][slot].zero_()

    def copy_slot(self, src: int, dst: int) -> None:
        for i in self.layer_ids:
            self.conv[i][dst].copy_(self.conv[i][src])
            self.ssm[i][dst].copy_(self.ssm[i][src])
