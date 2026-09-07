"""Read tensors from the pinned snapshot's safetensors shards."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open


class SnapshotWeights:
    def __init__(self, snap: Path):
        self.snap = snap
        index = json.loads((snap / "model.safetensors.index.json").read_text())
        self._map: dict[str, str] = index["weight_map"]
        self._handles: dict[str, object] = {}

    def names(self) -> list[str]:
        return sorted(self._map)

    def __contains__(self, name: str) -> bool:
        return name in self._map

    def get(self, name: str) -> torch.Tensor:
        shard = self._map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.snap / shard), framework="pt", device="cpu")
            self._handles[shard] = handle
        # safetensors returns a view of the file mapping; copying that to the
        # device page-faults the file in through the DMA staging path at a
        # few hundred MB/s on GB10 (105 s of a 108 s load). A memcpy into RAM
        # first runs at 25 GB/s and the device copy then at 18 GB/s.
        return handle.get_tensor(name).clone()

    def close(self) -> None:
        self._handles.clear()
