"""Conv and SSM state slots for the recurrent layers (the 48 Gated DeltaNet layers on the 27B, Nemotron's 40 Mamba-2).

Every request owns one committed slot. Scratch slots are reserved for
speculative verification. Acceptance either copies a retained
snapshot or reconstructs directly into validated private slots. The tensors are per layer so the
kernels' `initial_state_indices` address the slot directly.

The recurrent state is FP32 [HV, V, K] per layer and slot (FP16 with a Mamba-2 target's `fp16_sr` format), or,
with a quantized GDN `SsmFormat` (int8, int4), one byte row per layer and slot that holds the per-window quantized
state (`cache/ssm_format.py`; typed views in `window`).
"""

from __future__ import annotations

import torch

from sharar.cache.ssm_format import FP32, SsmFormat, WindowPool
from sharar.target import DEFAULT_TARGET, Target

# The default target's convolution geometry, for tools written against the 27B.
CONV_DIM = DEFAULT_TARGET.conv_dim
CONV_STATE_LEN = DEFAULT_TARGET.conv_state_len


class StatePool:
    def __init__(self, num_slots: int, device: torch.device, num_scratch: int = 0, target: Target | None = None,
                 ssm_format: SsmFormat = FP32):
        t = self.target = target or DEFAULT_TARGET
        ssm_format.check(t)
        self.ssm_format = ssm_format
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
        layout = ssm_format.layout(t)
        if layout is None:
            self.ssm_all = torch.zeros(
                (n_layers, total, t.linear_num_value_heads, t.linear_value_head_dim, t.linear_key_head_dim),
                dtype=ssm_format.dtype,
                device=device,
            )
            self.window = None
        else:  # a zeroed row is the zero state
            self.ssm_all = torch.zeros((n_layers, total, layout.nbytes), dtype=torch.uint8, device=device)
            self.window = WindowPool(layout, self.ssm_all, self.layer_ids)
        self.conv = {lid: self.conv_all[j] for j, lid in enumerate(self.layer_ids)}
        self.ssm = {lid: self.ssm_all[j] for j, lid in enumerate(self.layer_ids)}
        # Slot 0 is the dummy slot for padded batch rows; scratch slots follow the committed ones.
        self._free = list(range(num_slots - 1, 0, -1))
        self.scratch = list(range(num_slots, total))
        self.on_free = []  # optional auxiliary owners tied to a committed slot
        # Extra per-slot state registered by a model (e.g. the PLE convolution window), [total, ...] each;
        # cleared, copied and stored on the SSD tier together with the recurrent states.
        self.aux: dict[str, torch.Tensor] = {}

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def bytes_per_slot(self) -> int:
        return self.ssm_format.state_bytes_per_slot(self.target)

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

    def register_aux(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Allocate (once) a zeroed [total_slots, *shape] tensor that follows the slots' lifecycle."""
        t = self.aux.get(name)
        if t is None:
            t = self.aux[name] = torch.zeros((self.num_slots + len(self.scratch), *shape), dtype=dtype, device=self.device)
        return t

    def clear(self, slot: int) -> None:
        self.conv_all[:, slot].zero_()
        self.ssm_all[:, slot].zero_()
        for t in getattr(self, "aux", {}).values():  # test doubles skip __init__
            t[slot].zero_()

    def copy_slot(self, src: int, dst: int) -> None:
        self.conv_all[:, dst].copy_(self.conv_all[:, src])
        self.ssm_all[:, dst].copy_(self.ssm_all[:, src])
        for t in getattr(self, "aux", {}).values():
            t[dst].copy_(t[src])
