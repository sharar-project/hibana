"""Two-rank one-shot all-reduce over RDMA (csrc/rdma_oneshot.cu), compiled on first use.

DGX Spark has no GPUDirect RDMA, so NCCL stages each step of its ring through host buffers and a proxy thread: a
160 KiB all-reduce (one verification of 16 tokens) costs it about 50 us inside a CUDA graph, against about 12 us to
move the bytes both ways. This exchanges the two partials once, through cudaHostAlloc buffers registered with both
RDMA devices, and adds them: about 19 us, bitwise NCCL's sum. The devices are the ones with an IPv4 RoCE v2 GID (the
connected ConnectX-7 halves), paired in name order with the peer's (on two Sparks, the same-named ones). Builds against the ibverbs headers:
the system's, else the copy `sharar serve --tp 2` unpacks into the Python environment (sharar/launch/tp.py).
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load

from sharar.engine.config import current

_SRC = Path(__file__).resolve().parent / "csrc" / "rdma_oneshot.cu"
_SYSFS = Path("/sys/class/infiniband")
IPV4_GID = "0000:0000:0000:0000:0000:ffff:"


@lru_cache(maxsize=1)
def _module():
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", current().cuda_arch)
    return load(name="sharar_rdma_oneshot", sources=[str(_SRC)], extra_cuda_cflags=["-O3"],
                extra_include_paths=[str(Path(sys.prefix) / "ibverbs/usr/include")],
                extra_ldflags=["-l:libibverbs.so.1"], verbose=False)


def roce_v2_ipv4_gid(device: str) -> int | None:
    """The index of `device`'s RoCE v2 GID that holds an IPv4 address, or None (an unconnected port)."""
    port = _SYSFS / device / "ports/1"
    for entry in sorted((port / "gid_attrs/types").iterdir(), key=lambda p: int(p.name)):
        try:
            kind = entry.read_text().strip()
        except OSError:  # empty table entries do not read
            continue
        if kind == "RoCE v2" and (port / "gids" / entry.name).read_text().startswith(IPV4_GID):
            return int(entry.name)
    return None


def create(max_bytes: int):
    """The transport for messages of up to `max_bytes`, unconnected: exchange `info()` with the peer, then `connect`."""
    gids = {d.name: roce_v2_ipv4_gid(d.name) for d in sorted(_SYSFS.iterdir())}
    devices = [name for name, gid in gids.items() if gid is not None]
    return _module().OneShot(devices, [gids[d] for d in devices], max_bytes)
