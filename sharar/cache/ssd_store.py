"""SSD store: content-addressed KV blocks and session tails on the local NVMe.

The tiered radix tree (`radix.py`) records what is on disk; this module only
stores and returns objects. Two kinds:

- `kv`: one block of `block` tokens (default `BLOCK`), keyed by a chained hash
  of every token from the root, so a shared prefix is stored once whatever
  session wrote it. Segments: K and V of every attention layer, then the KV
  pool's page aux rows.
- `tail`: the end of a session: the KV of its last partial block, the GDN
  snapshot at its exact length, the state pool's per-slot aux and the drafter
  payload. Keyed by the last full block's hash and the partial tokens.

A file is a header (magic, the JSON's size and CRC32, the JSON, padded to
4 KiB) and the payload segments, each 256-byte aligned. The header names its
own key, its parent hash, the tokens it covers and a CRC32 of the payload. A
read checks both CRCs, recomputes the key from the parent and tokens and checks
the segment sizes, so a collision, a stale file, a flipped byte anywhere or a
torn write is a miss. A
writer writes each file to `.tmp`, fdatasyncs it and renames it into place: a
crash loses only in-flight writes. The root hash folds in the serving identity
(target, revision, decode weights, geometry), the block size and the format,
so another build never matches.

Writes run on `writers` threads, each key always on the same thread: one key's
writes and deletes run in submission order (a write queued after a delete of
its key lands after it), while other keys' fsyncs overlap. Completions come
back in submission order whichever writer finishes first, so every rank sees
the same sequence. The engine thread queues every write, drains completions
each step and is the only owner of the tree.

Restores read with O_DIRECT straight into one buffer the GPU reads in place
(`read_into`). Writes use O_DIRECT too: each writer copies the object through
its own page-aligned `WRITE_BUFFER` (the same one copy a buffered write makes
into the page cache), so backups neither crowd unified memory nor wait on
writeback. Where the filesystem refuses O_DIRECT, a writer writes through the
page cache and drops the file from it after the fdatasync (ZFS ignores the
drop: its own cache, the ARC, keeps it). O_DIRECT must reach the drive to mean
anything: under a container whose root is a ZFS pool kept in a file on the
host (LXD's default), reads and writes still go through the host page cache.

Files live in `v2/e<epoch>/<hh>/`, hh being the key's first two hex digits. A
flush starts a new epoch and deletes the old one in the background, and
startup deletes any epoch a crash left behind, so flushed data never comes
back, not even from a write that was in flight.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import functools
import hashlib
import json
import math
import mmap
import os
import queue
import shutil
import struct
import threading
import time
import zlib
from array import array
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import torch

FORMAT = 3
BLOCK = 256  # tokens per KV block; divides the 2048-token prefill chunk
WRITERS = 4  # writer threads: each object's fdatasync waits on the drive, so one thread would leave it mostly idle
WRITE_BUFFER = 16 * 2**20  # each writer's O_DIRECT buffer: an object goes out in writes this large
MAGIC = b"SHARSSD\n"
FIXED = len(MAGIC) + 12  # the magic, then the JSON header's size (u64) and CRC32 (u32)
ALIGN = 4096
SEG_ALIGN = 256
GIB = 2**30


def root_hash(identity: dict) -> bytes:
    return hashlib.sha256(json.dumps({**identity, "format": FORMAT}, sort_keys=True).encode()).digest()


def _token_bytes(tokens) -> bytes:
    return array("i", tokens).tobytes()  # int32 in native (little-endian) order: every stored key depends on it


def block_hash(parent: bytes, tokens) -> bytes:
    return hashlib.sha256(parent + _token_bytes(tokens)).digest()


def tail_hash(last_block: bytes, tokens) -> bytes:
    return hashlib.sha256(b"tail" + last_block + _token_bytes(tokens)).digest()


def _seg_pad(n: int) -> int:
    return (n + SEG_ALIGN - 1) // SEG_ALIGN * SEG_ALIGN


@dataclass
class Header:
    kind: str  # "kv" or "tail"
    key: str  # hex of its hash
    parent: str  # hex of the previous block's hash (the root hash for the first)
    start: int  # absolute position of its first token
    tokens: list[int]
    segments: list[list]  # [name, dtype, shape, nbytes], in payload order
    scalars: dict = field(default_factory=dict)  # the non-tensor values of each drafter's payload, by drafter identity
    crc: int = 0  # CRC32 of the payload
    nbytes: int = 0  # the payload's size, every segment padded to SEG_ALIGN

    @property
    def end(self) -> int:
        return self.start + len(self.tokens)

    def layout(self):
        """(name, dtype, shape, offset in the payload, nbytes) per segment."""
        cursor = 0
        for name, dtype, shape, nbytes in self.segments:
            yield name, dtype, shape, cursor, nbytes
            cursor += _seg_pad(nbytes)

    def check(self) -> None:
        """Raise ValueError unless the key matches the parent and tokens and the
        segments add up, so a corrupt header cannot file an object under the
        wrong tokens."""
        if (not isinstance(self.parent, str) or not isinstance(self.key, str)
                or len(self.parent) != 64 or len(self.key) != 64
                or type(self.start) is not int or self.start < 0
                or not isinstance(self.tokens, list)
                or any(type(t) is not int or not 0 <= t < 2**31 for t in self.tokens)
                or type(self.nbytes) is not int or self.nbytes < 0
                or type(self.crc) is not int or not 0 <= self.crc < 2**32
                or not isinstance(self.segments, list) or not isinstance(self.scalars, dict)
                or any(not isinstance(v, list) for v in self.scalars.values())):
            raise ValueError("invalid header fields")
        parent, key = bytes.fromhex(self.parent), bytes.fromhex(self.key)
        want = block_hash(parent, self.tokens) if self.kind == "kv" else tail_hash(parent, self.tokens)
        if self.kind not in ("kv", "tail") or key != want:
            raise ValueError("key does not match its tokens")
        names, total = set(), 0
        for segment in self.segments:
            if not isinstance(segment, list) or len(segment) != 4:
                raise ValueError("invalid segment")
            name, dtype, shape, nbytes = segment
            if (not isinstance(name, str) or name in names or not isinstance(dtype, str)
                    or not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                    or type(nbytes) is not int or nbytes < 0):
                raise ValueError("invalid segment fields")
            if math.prod(shape) * _itemsize(dtype) != nbytes:
                raise ValueError("segment shape does not match its size")
            names.add(name)
            total += _seg_pad(nbytes)
        if total != self.nbytes:
            raise ValueError("segments do not add up")


@functools.cache
def _itemsize(dtype: str) -> int:
    """Bytes per element of a torch dtype name; cached because restores check
    every segment of every object on the reader threads, which share the GIL
    with the engine thread."""
    value = getattr(torch, dtype, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unknown tensor dtype: {dtype}")
    return value.itemsize


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8).reshape(-1)


def memory_backed(path, mounts: str | None = None) -> str | None:
    """The filesystem type when `path` is on one kept in memory (tmpfs, ramfs: /tmp on some systems), else None."""
    if mounts is None:
        try:
            mounts = Path("/proc/mounts").read_text()
        except OSError:
            return None
    target, best, fstype = str(Path(path).resolve()), "", None
    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mount = fields[1].replace("\\040", " ")
        if (target == mount or target.startswith(mount.rstrip("/") + "/")) and len(mount) >= len(best):
            best, fstype = mount, fields[2]
    return fstype if fstype in ("tmpfs", "ramfs") else None


_LOCKED: dict[str, int] = {}  # stores this process holds: a store opened again here (a restart in tests) keeps the lock


def _lock(base: Path, directory) -> None:
    """One process per store, held until it exits: another would delete this one's files as stale."""
    key = str(base.resolve())
    if key in _LOCKED:
        return
    fd = os.open(base / "LOCK", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise OSError(f"{directory} is in use by another server (SHARAR_SSD_DIR gives this one its own)") from None
    _LOCKED[key] = fd


def _pad(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def _write_all(fd: int, view: memoryview) -> None:
    done = 0
    while done < len(view):
        k = os.write(fd, view[done:])
        if k <= 0:
            raise OSError(errno.EIO, "short write")
        done += k


class SsdStore:
    def __init__(self, directory, identity: dict, log=None, block: int = BLOCK, writers: int = WRITERS):
        if block < 1 or 2048 % block:
            raise ValueError(f"SSD block size must divide the 2048-token prefill chunk, got {block}")
        self.block = block  # tokens per KV block object
        if (fs := memory_backed(directory)) is not None:
            raise OSError(f"{directory} is on {fs}: the SSD tier would sit in memory (on unified memory, the GPU's); "
                          "set SHARAR_SSD_DIR to a directory on the NVMe")
        self.base = Path(directory) / "v2"
        self.base.mkdir(parents=True, exist_ok=True, mode=0o700)  # tails hold token ids: the user's own
        if self.base.is_symlink() or self.base.stat().st_uid != os.getuid():
            raise OSError(f"{self.base} is not this user's own directory")
        _lock(self.base, directory)
        self.root = root_hash({**identity, "block": block})
        self.epoch = self._read_epoch()
        self.dir = self.base / f"e{self.epoch}"
        self.dir.mkdir(exist_ok=True)
        stale = [p for p in self.base.iterdir() if p.is_dir() and p != self.dir]  # older epochs and trash a crash left behind
        if stale:
            threading.Thread(target=lambda: [shutil.rmtree(p, ignore_errors=True) for p in stale], daemon=True).start()
        self.queues: list[queue.Queue] = [queue.Queue() for _ in range(writers)]  # one per writer, chosen by key
        self.done: deque = deque()  # per write, in submission order: a one-item list its writer fills with the completion
        self._lock = threading.Lock()  # guards the counters the writer and reader threads update
        self.writes = self.reads = self.read_errors = self.write_errors = self.deletes = 0
        self.write_bytes = self.read_bytes = 0
        self.write_seconds = self.read_seconds = 0.0
        self.last_error = None
        self.direct = hasattr(os, "O_DIRECT")  # until the filesystem refuses it
        for i, q in enumerate(self.queues):
            threading.Thread(target=self._writer, args=(q,), name=f"sharar-ssd-writer-{i}", daemon=True).start()
        if log:
            log(f"  ssd store: {self.dir}")

    # ---- paths ----
    def _read_epoch(self) -> int:
        try:
            return int((self.base / "EPOCH").read_text())
        except (OSError, ValueError):
            return 0

    def path(self, kind: str, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.{kind}"

    # ---- write ----
    def _queue(self, key: str) -> queue.Queue:
        return self.queues[int(key[:8], 16) % len(self.queues)]

    def write(self, header: Header, tensors: list[torch.Tensor], ready=None) -> None:
        """Queue an object, from the engine thread. `tensors` must stay untouched until its completion
        (host staging copies, or device views guarded by `ready`, a CUDA event)."""
        slot = [None]
        self.done.append(slot)
        self._queue(header.key).put(("write", header, tensors, ready, slot))

    def delete(self, kind: str, key: str) -> None:
        self._queue(key).put(("delete", kind, key, None, None))

    def wait(self) -> None:
        """Block until every queued write and delete has run (a writer fills a
        write's slot before it marks the job done)."""
        for q in self.queues:
            q.join()

    def completions(self) -> list[tuple[str, str, bool, int]]:
        """(kind, key, ok, nbytes) of the finished writes, in submission order, up to the first still in flight."""
        out = []
        while self.done and self.done[0][0] is not None:
            out.append(self.done.popleft()[0])
        return out

    def _writer(self, jobs: queue.Queue) -> None:
        buf = None  # this writer's O_DIRECT buffer, mapped at its first write
        while True:
            op, a, b, c, slot = jobs.get()  # ("write", header, tensors, ready, slot) or ("delete", kind, key, None, None)
            if op == "delete":
                self._unlink(self.path(a, b))
                with self._lock:
                    self.deletes += 1
            else:
                try:
                    if buf is None and self.direct:
                        buf = mmap.mmap(-1, WRITE_BUFFER)  # anonymous, so page-aligned
                    slot[0] = self._write(a, b, c, buf)
                except Exception as exc:  # a full disk or an I/O error loses the stored copy, never the served session
                    with self._lock:
                        self.write_errors += 1
                        self.last_error = repr(exc)
                    slot[0] = (a.kind, a.key, False, 0)
            del a, b, c, slot  # do not retain the last write's staging tensors while waiting for another job
            jobs.task_done()

    def _write(self, header: Header, tensors: list[torch.Tensor], ready, buf: mmap.mmap | None) -> tuple[str, str, bool, int]:
        t0 = time.perf_counter()
        if ready is not None:
            ready.synchronize()
        chunks = []
        for t in tensors:  # each segment starts 256-byte aligned, so a read can view every dtype in place
            chunk = _bytes(t).cpu()
            chunks.append(chunk)
            if chunk.numel() % SEG_ALIGN:
                chunks.append(torch.zeros(SEG_ALIGN - chunk.numel() % SEG_ALIGN, dtype=torch.uint8))
        crc, n = 0, 0
        for chunk in chunks:
            crc = zlib.crc32(memoryview(chunk.numpy()), crc)
            n += chunk.numel()
        header.crc, header.nbytes = crc, n
        head = json.dumps(header.__dict__).encode()
        prefix = MAGIC + struct.pack("<QI", len(head), zlib.crc32(head)) + head
        path = self.path(header.kind, header.key)
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        head = ctypes.create_string_buffer(prefix, _pad(len(prefix)))  # zero-filled to a page boundary
        parts = [(ctypes.addressof(head), len(head)), *((c.data_ptr(), c.numel()) for c in chunks), (None, _pad(n) - n)]  # whole 4 KiB pages
        try:
            if not (self.direct and self._write_direct(tmp, parts, buf)):
                self._write_buffered(tmp, parts)
            os.replace(tmp, path)
        except BaseException:
            self._unlink(tmp)  # a full disk must not fill up further with partial files
            raise
        with self._lock:
            self.writes += 1
            self.write_bytes += n
            self.write_seconds += time.perf_counter() - t0
        return header.kind, header.key, True, _pad(len(prefix)) + _pad(n)

    def _write_direct(self, tmp: Path, parts, buf: mmap.mmap) -> bool:
        """Write parts ((address, nbytes), or (None, nbytes) of zeros) to tmp with O_DIRECT through buf, then fdatasync
        it. Returns False, and turns O_DIRECT off for good, where the filesystem refuses it."""
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
        except OSError as exc:
            if exc.errno != errno.EINVAL:
                raise
            self.direct = False
            return False
        base, view, fill = ctypes.addressof(ctypes.c_char.from_buffer(buf)), memoryview(buf), 0
        try:
            for src, nbytes in parts:
                done = 0
                while done < nbytes:
                    k = min(nbytes - done, len(buf) - fill)
                    if src is None:  # ctypes calls release the GIL, so the engine thread keeps running
                        ctypes.memset(base + fill, 0, k)
                    else:
                        ctypes.memmove(base + fill, src + done, k)
                    fill, done = fill + k, done + k
                    if fill == len(buf):
                        _write_all(fd, view[:fill])
                        fill = 0
            if fill:
                _write_all(fd, view[:fill])
            os.fdatasync(fd)  # O_DIRECT persists neither the file's size nor what sits in the drive's write cache
        except OSError as exc:
            if exc.errno != errno.EINVAL:
                raise
            self.direct = False  # opened with O_DIRECT, then refused it (some overlays)
            return False
        finally:
            view.release()
            os.close(fd)
        return True

    @staticmethod
    def _write_buffered(tmp: Path, parts) -> None:
        with open(tmp, "wb") as f:
            for src, nbytes in parts:
                f.write(b"\0" * nbytes if src is None else (ctypes.c_char * nbytes).from_address(src))
            f.flush()
            os.fdatasync(f.fileno())  # then drop it from the page cache: backups must not squeeze the engine's unified memory
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

    # ---- read ----
    def _read_header(self, f) -> tuple[Header, int]:
        size, crc = self._header_size(f.read(FIXED))
        return self._parse_header(f.read(size), crc), _pad(FIXED + size)

    @staticmethod
    def _header_size(fixed: bytes) -> tuple[int, int]:
        """The JSON header's size and CRC32 from the bytes before it."""
        if len(fixed) != FIXED or fixed[: len(MAGIC)] != MAGIC:
            raise ValueError("bad magic")
        size, crc = struct.unpack("<QI", fixed[len(MAGIC):])
        if size > 16 * 2**20:
            raise ValueError("bad header size")
        return size, crc

    @staticmethod
    def _parse_header(head: bytes, crc: int) -> Header:
        if zlib.crc32(head) != crc:
            raise ValueError("header crc mismatch")
        return Header(**json.loads(head))

    def size(self, kind: str, key: str) -> int | None:
        try:
            return os.stat(self.path(kind, key)).st_size
        except OSError:
            return None

    @staticmethod
    def segments(header: Header, base: torch.Tensor, offset: int) -> dict[str, torch.Tensor]:
        """An object's segments as views of `base`, whose payload starts at `offset`."""
        return {name: base[offset + at:offset + at + nbytes].view(getattr(torch, dtype)).view(shape)
                for name, dtype, shape, at, nbytes in header.layout()}

    def read_into(self, kind: str, key: str, view: memoryview) -> tuple[Header, int] | None:
        """Read a whole object with O_DIRECT (no page cache) into `view`, a
        4 KiB-aligned span at least as large as the file, and check it.
        Returns the header and the payload's offset in `view` (segments() gives
        the views): on GB10 the caller's buffer is registered host memory the
        GPU reads in place. Returns None for a missing file, and for a torn,
        mismatched or corrupt one, which it also deletes."""
        t0 = time.perf_counter()
        path = self.path(kind, key)
        try:
            size = self._read_file(path, view)
            hsize, hcrc = self._header_size(bytes(view[:FIXED]))
            header = self._parse_header(bytes(view[FIXED:FIXED + hsize]), hcrc)
            offset = _pad(FIXED + hsize)
            if header.kind != kind or header.key != key or offset + header.nbytes > size:
                raise ValueError("header does not match its name")
            header.check()
            if zlib.crc32(view[offset:offset + header.nbytes]) != header.crc:
                raise ValueError("crc mismatch")
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            with self._lock:  # restores read from several threads at once
                self.read_errors += 1
                self.last_error = repr(exc)
            self.delete(kind, key)
            return None
        with self._lock:
            self.reads += 1
            self.read_bytes += header.nbytes
            self.read_seconds += time.perf_counter() - t0
        return header, offset

    @staticmethod
    def _read_file(path: Path, view: memoryview) -> int:
        """Read the whole file into view with O_DIRECT, or through the page cache
        where the filesystem refuses it (at open, or on some overlays at read).
        Returns the file's size."""
        flags = os.O_RDONLY | getattr(os, "O_DIRECT", 0)
        while True:
            try:
                fd = os.open(path, flags)
            except OSError as exc:
                if exc.errno != errno.EINVAL or flags == os.O_RDONLY:
                    raise
                flags = os.O_RDONLY
                continue
            try:
                size, got = os.fstat(fd).st_size, 0
                if size > len(view):
                    raise ValueError("object larger than its span")
                while got < size:
                    k = os.preadv(fd, [view[got:size]], got)
                    if k <= 0:
                        raise ValueError("short read")
                    got += k
                return size
            except OSError as exc:
                if exc.errno != errno.EINVAL or flags == os.O_RDONLY:
                    raise
                flags = os.O_RDONLY
            finally:
                os.close(fd)

    def scan(self):
        """Every stored object's header and file size (payloads unread), for
        rebuilding the tree at startup. Deletes leftover `.tmp` files and any
        file it cannot use, a truncated one included."""
        for sub in sorted(self.dir.iterdir()) if self.dir.exists() else []:
            if not sub.is_dir():
                continue
            for path in sorted(sub.iterdir()):
                if path.suffix == ".tmp":
                    self._unlink(path)
                    continue
                try:
                    with open(path, "rb") as f:
                        header, offset = self._read_header(f)
                        size = os.fstat(f.fileno()).st_size
                    if path.name != f"{header.key}.{header.kind}":
                        raise ValueError("misnamed")
                    header.check()
                    if size != offset + _pad(header.nbytes):
                        raise ValueError("file size does not match its header")
                    yield header, size
                except (OSError, ValueError, TypeError, KeyError, RuntimeError):
                    self._unlink(path)

    # ---- flush ----
    def flush(self) -> None:
        """Start a new epoch: let queued work finish into the old directory,
        then rename it aside and delete it in the background."""
        self.wait()
        old = self.dir
        self.epoch += 1
        tmp = self.base / "EPOCH.tmp"
        with open(tmp, "w") as f:
            f.write(str(self.epoch))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.base / "EPOCH")
        fd = os.open(self.base, os.O_RDONLY)  # the rename itself must survive a power loss, or flushed data comes back
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self.dir = self.base / f"e{self.epoch}"
        self.dir.mkdir(exist_ok=True)
        trash = self.base / f"trash-{self.epoch - 1}-{time.monotonic_ns()}"
        try:
            os.replace(old, trash)
        except OSError:
            return
        threading.Thread(target=shutil.rmtree, args=(trash,), kwargs={"ignore_errors": True}, daemon=True).start()

    @staticmethod
    def _unlink(path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass

    def stats(self) -> dict:
        return {"dir": str(self.dir), "writes": self.writes, "reads": self.reads, "deletes": self.deletes,
                "write_gib": round(self.write_bytes / GIB, 3), "read_gib": round(self.read_bytes / GIB, 3),
                "write_errors": self.write_errors, "read_errors": self.read_errors, "last_error": self.last_error,
                "pending": sum(q.unfinished_tasks for q in self.queues), "writers": len(self.queues), "direct_writes": self.direct,
                "write_ms_avg": round(1000 * self.write_seconds / max(1, self.writes), 2),
                "read_ms_avg": round(1000 * self.read_seconds / max(1, self.reads), 2)}
