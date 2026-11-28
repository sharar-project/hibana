"""Read tensors from the pinned snapshot's safetensors shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from hibana.weights import files


def snapshot_identity(snap: Path) -> str:
    """The files a model reads from its snapshot (config.json, the index and every shard it names), by size and
    modification time: the SSD store keys stored prefixes by it, so two weight sets under one revision (from separate
    model caches, say) never restore each other's. Only a stat each; a re-downloaded or copied file costs a cold store."""
    index = json.loads((snap / "model.safetensors.index.json").read_text())
    h = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", *sorted(set(index["weight_map"].values()))):
        st = (snap / name).stat()  # through the model cache's link, its blob
        h.update(f"{name}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


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

    def shards(self, names) -> list[Path]:
        """The files holding `names`, in first-use order."""
        return [self.snap / shard for shard in dict.fromkeys(self._map[name] for name in names)]

    def ahead(self, names) -> None:
        """Start reading the files holding `names` whole (`files.ahead`), for a later `get_whole`."""
        for path in self.shards(names):
            files.ahead(path)

    def get_whole(self, names, device) -> dict[str, torch.Tensor]:
        """`names` as views of one device copy of each file holding them, read whole (`weights/files.py`): for files
        that hold little else, such as one layer's routed experts. The files' other tensors are copied too."""
        out = {}
        for path in self.shards(names):
            data = files.take(path)
            entries = files.layout(data)
            raw = files.as_tensor(data).to(device)
            del data
            out.update({name: files.tensor(raw, entries[name]) for name in names if self.snap / self._map[name] == path})
        return out

    def close(self) -> None:
        self._handles.clear()
