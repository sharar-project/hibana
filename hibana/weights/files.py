"""Whole weight files read once, on worker threads ahead of their use, and the safetensors tensors in them as views.

A load on the GB10 is bound by reading and hashing: cold NVMe reads reach about 1.8 GB/s from one thread and 3.2 GB/s
from several, SHA-256 about 2.2 GB/s a thread, and copying a file-backed mapping to the GPU page-faults it in at a few
hundred MB/s. So a file is read with `readinto` into ordinary memory, hashed there when it has a recorded checksum,
and its tensors are views of that buffer (on the CPU, or of one copy of the whole file on the GPU). `ahead` starts a
read on the pool; `take` returns it, reading on the spot when nothing started it. (Dropping a file's pages from the
page cache once read left about 4 GiB less MemAvailable after a Flash-Next load, so fewer KV pages: they stay.)
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch

WORKERS = 4
CHUNK = 64 << 20

_DTYPES = {"BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8, "U16": torch.uint16, "I16": torch.int16,
           "U32": torch.uint32, "I32": torch.int32, "U64": torch.uint64, "I64": torch.int64, "F16": torch.float16,
           "BF16": torch.bfloat16, "F32": torch.float32, "F64": torch.float64, "F8_E4M3": torch.float8_e4m3fn,
           "F8_E5M2": torch.float8_e5m2}

_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None
_pending: dict[Path, Future] = {}


def read(path: Path, sha256: str | None = None) -> bytearray:
    """The file's bytes; with `sha256`, checked against it."""
    size = os.path.getsize(path)
    data = bytearray(size)
    view = memoryview(data)
    with open(path, "rb", buffering=0) as handle:
        done = 0
        while done < size:
            got = handle.readinto(view[done:done + CHUNK])
            if not got:
                raise EOFError(f"{path}: {done} of {size} bytes")
            done += got
    if sha256 is not None and hashlib.sha256(view).hexdigest() != sha256:
        raise ValueError(f"checksum mismatch for {path}")
    return data


def ahead(path: Path, sha256: str | None = None) -> None:
    """Start reading `path` on the pool (a no-op when it is already read or being read)."""
    global _pool
    path = Path(path)
    with _lock:
        if path not in _pending:
            if _pool is None:
                _pool = ThreadPoolExecutor(WORKERS, thread_name_prefix="sharar-read")
            _pending[path] = _pool.submit(read, path, sha256)


def take(path: Path, sha256: str | None = None) -> bytearray:
    """`path`'s bytes: the read `ahead` started, else one now. A read started ahead with a checksum was checked."""
    with _lock:
        future = _pending.pop(Path(path), None)
    return future.result() if future is not None else read(Path(path), sha256)


def layout(data: bytearray) -> dict[str, tuple[torch.dtype, tuple[int, ...], int, int]]:
    """A safetensors file's tensors: name -> (dtype, shape, first byte, end byte) within the file."""
    n = int.from_bytes(data[:8], "little")
    header = json.loads(bytes(data[8:8 + n]))
    header.pop("__metadata__", None)
    base = 8 + n
    return {name: (_DTYPES[t["dtype"]], tuple(t["shape"]), base + t["data_offsets"][0], base + t["data_offsets"][1])
            for name, t in header.items()}


def tensor(raw: torch.Tensor, entry: tuple[torch.dtype, tuple[int, ...], int, int]) -> torch.Tensor:
    """One tensor of a file as a view of `raw`, the whole file as uint8 on any device."""
    dtype, shape, start, end = entry
    return raw[start:end].view(dtype).reshape(shape)


def as_tensor(data: bytearray) -> torch.Tensor:
    """The bytes as a uint8 CPU tensor sharing their memory."""
    return torch.frombuffer(data, dtype=torch.uint8) if len(data) else torch.empty(0, dtype=torch.uint8)


def discard() -> None:
    """Forget every read started ahead and not taken (after a load that failed part way)."""
    with _lock:
        futures = list(_pending.values())
        _pending.clear()
    for future in futures:
        future.cancel()
