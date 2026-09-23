"""Session store on the NVMe: the cold tier's parked sessions, written through.

A session parked in the unified-memory arena (`cold_tier.py`) is copied to a
file in the background; when the arena drops it (or the server restarts), the
next request whose prompt extends its key reads it straight into fresh pool
pages and a state slot, and the radix cache resumes it like any parked leaf. Measured on this GB10 (2026-09-23): direct
reads 9.7 GB/s and writes 7.0 GB/s, so a 262K-token session (8.2 GiB of FP8 KV
plus a 144 MiB recurrent snapshot) comes back in about a second instead of a
351-second prefill.

A record is `<sha1 of the key>.bin` (segments, raw bytes) and `.json` (identity,
key length, segment table, codec scalars), renamed into place in that order, so
a record without its JSON is incomplete and ignored. The KV layout does not
depend on how the arena chunked the session: one segment per layer and K/V
with every token of the leaf. Only records whose identity (target, revision,
decode weights, geometry, format) matches the serving process are listed.
The writer thread owns no arena memory: the cold tier waits for a pending
write before it frees an entry's chunks.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

GIB = 2**30
FORMAT = 1
STAGING = 128 * 2**20


@dataclass
class Record:
    key: list[int]
    own: int
    stem: Path
    nbytes: int
    meta: dict
    last_access: float = field(default_factory=time.time)

    def aux_specs(self, codecs: dict) -> list[tuple]:
        """(registered identity, scalars, [(dtype, shape)]) for payloads this process can import."""
        by_name = {str(identity): identity for identity in codecs}
        out = []
        for item in self.meta["aux"]:
            identity = by_name.get(item["identity"])
            if identity is not None:
                out.append((identity, item["scalars"], [(getattr(torch, d), tuple(s)) for d, s in item["tensors"]]))
        return out


def _segments(entry) -> list[tuple[str, list[torch.Tensor]]]:
    """The byte segments of a parked entry, in file order; each is the
    concatenation of its tensors' bytes."""
    layers = entry.chunks[0][1].shape[0] if entry.chunks else 0  # chunks are (first token, KV view, page aux views)
    out = [(f"kv.{j}.{t}", [chunk[1][j, t] for chunk in entry.chunks]) for j in range(layers) for t in range(2)]
    for name in (entry.chunks[0][2] if entry.chunks and len(entry.chunks[0]) > 2 else {}):
        out.append((f"paux.{name}", [chunk[2][name] for chunk in entry.chunks]))
    out += [("conv", [entry.conv]), ("ssm", [entry.ssm])]
    for identity, (scalars, views) in entry.auxiliary.items():
        if identity == "__pool__":
            out += [(f"pool.{name}", [v]) for name, v in views]
        else:
            out += [(f"aux.{identity}.{i}", [v]) for i, v in enumerate(views)]
    return out


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8).reshape(-1)


class SessionDisk:
    def __init__(self, directory, budget_bytes: int, identity: dict, device: torch.device, log=None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.budget = int(budget_bytes)
        self.identity = {**identity, "format": FORMAT}
        self.device = device
        self.records: dict[str, Record] = {}
        self.lock = threading.Lock()  # records and files: the writer thread and the engine thread
        self.pending: dict[int, threading.Event] = {}
        self.queue: queue.Queue = queue.Queue()
        self.writes = self.reads = self.skipped = self.errors = self.deleted = 0
        self.write_seconds = self.read_seconds = 0.0
        self.last_error = None
        cuda = device.type == "cuda"
        self._write_staging = torch.empty(STAGING, dtype=torch.uint8, pin_memory=cuda)
        self._read_staging = torch.empty(STAGING, dtype=torch.uint8, pin_memory=cuda)
        self._stream = torch.cuda.Stream(device) if cuda else None
        self._bounce = None  # device bounce buffer for reads, allocated on first use
        self._scan()
        self.thread = threading.Thread(target=self._writer, name="sharar-session-writer", daemon=True)
        self.thread.start()
        if log:
            log(f"  session store: {self.dir} ({len(self.records)} sessions, {self.used_bytes / GIB:.1f} of {self.budget / GIB:.0f} GiB)")

    # ---- index ----
    @property
    def used_bytes(self) -> int:
        return sum(r.nbytes for r in self.records.values())

    def _scan(self) -> None:
        for meta_path in self.dir.glob("*.json"):
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get("identity") != self.identity:
                    continue
                bin_path = meta_path.with_suffix(".bin")
                key = torch.from_file(str(bin_path), size=meta["key_len"], dtype=torch.int32).tolist()
                self.records[meta_path.stem] = Record(key, meta["own"], meta_path.with_suffix(""), bin_path.stat().st_size,
                                                      meta, bin_path.stat().st_mtime)
            except (OSError, ValueError, KeyError, RuntimeError):
                continue  # incomplete or foreign: left alone

    def find(self, ids: list[int]) -> list[Record]:
        with self.lock:
            return [r for r in self.records.values() if len(r.key) <= len(ids) and ids[:len(r.key)] == r.key]

    # ---- write ----
    def submit(self, entry) -> None:
        """Queue a parked entry for writing unless a stored record already covers it."""
        if entry.on_disk:
            return
        with self.lock:
            covered = any(len(r.key) >= len(entry.key) and r.key[:len(entry.key)] == entry.key for r in self.records.values())
        if covered:
            entry.on_disk = True
            self.skipped += 1
            return
        done = threading.Event()
        self.pending[id(entry)] = done
        self.queue.put((entry, done))

    def drain(self) -> None:
        """Wait for every queued write."""
        for done in list(self.pending.values()):
            done.wait()

    def wait(self, entry) -> None:
        done = self.pending.get(id(entry))
        if done is not None:
            done.wait()

    def _writer(self) -> None:
        while True:
            entry, done = self.queue.get()
            try:
                self._write(entry)
            except Exception as exc:  # a full disk or an I/O error loses the stored copy, never the served session
                self.errors += 1
                self.last_error = repr(exc)
            finally:
                self.pending.pop(id(entry), None)
                done.set()

    def _write(self, entry) -> None:
        t0 = time.perf_counter()
        if entry.ready is not None:
            entry.ready.synchronize()
        key = torch.tensor(entry.key, dtype=torch.int32)
        name = hashlib.sha1(key.numpy().tobytes()).hexdigest()
        stem = self.dir / name
        table, aux = [], []
        for identity, (scalars, views) in entry.auxiliary.items():
            if identity == "__pool__":
                continue
            try:
                json.dumps(list(scalars))
            except TypeError:
                continue
            aux.append({"identity": str(identity), "scalars": list(scalars),
                        "tensors": [[str(v.dtype).removeprefix("torch."), list(v.shape)] for v in views]})
        tmp = stem.with_suffix(".bin.tmp")
        with open(tmp, "wb") as f:
            f.write(key.numpy().tobytes())
            offset = key.numel() * 4
            for seg, tensors in _segments(entry):
                if seg.startswith("aux.") and not any(seg.startswith(f"aux.{a['identity']}.") for a in aux):
                    continue
                n = 0
                for t in tensors:
                    n += self._write_tensor(f, t)
                table.append([seg, offset, n])
                offset += n
        os.replace(tmp, stem.with_suffix(".bin"))
        meta = {"identity": self.identity, "key_len": len(entry.key), "own": entry.own, "segments": table, "aux": aux,
                "written": time.time()}
        tmp = stem.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, stem.with_suffix(".json"))
        with self.lock:
            self.records[name] = Record(list(entry.key), entry.own, stem, offset, meta)
        entry.on_disk = True
        self.writes += 1
        self.write_seconds += time.perf_counter() - t0
        self._enforce_budget(keep=name)

    def _write_tensor(self, f, t: torch.Tensor) -> int:
        src = _bytes(t)
        if src.device.type != "cuda":
            f.write(src.numpy().tobytes())
            return src.numel()
        with torch.cuda.stream(self._stream):
            for a in range(0, src.numel(), STAGING):
                n = min(STAGING, src.numel() - a)
                self._write_staging[:n].copy_(src[a:a + n])
                self._stream.synchronize()
                f.write(memoryview(self._write_staging[:n].numpy()))
        return src.numel()

    def _enforce_budget(self, keep: str) -> None:
        with self.lock:
            while self.used_bytes > self.budget and len(self.records) > 1:
                name = min((n for n in self.records if n != keep), key=lambda n: self.records[n].last_access)
                self._delete(name)

    def _delete(self, name: str) -> None:
        record = self.records.pop(name)
        for suffix in (".json", ".bin"):
            try:
                record.stem.with_suffix(suffix).unlink()
            except OSError:
                pass
        self.deleted += 1

    def clear(self) -> None:
        """Delete every stored session of this identity (a cache flush); writes in flight finish first."""
        for done in list(self.pending.values()):
            done.wait()
        with self.lock:
            for name in list(self.records):
                self._delete(name)

    # ---- read ----
    def restore_into(self, record: Record, start: int, pages: torch.Tensor, kv: list[torch.Tensor],
                     state: dict[str, torch.Tensor], aux_specs: list[tuple], page_aux: dict | None = None) -> dict | None:
        """Read a stored session straight into pool pages and a state slot:
        tokens [start:] of every KV segment into `pages` of the matching
        `kv` pool tensor (one per layer and K/V, in segment order) and of
        every page aux segment into its `page_aux` pool tensor, each
        `state` segment ("conv", "ssm", "pool.<name>") into its destination
        view, and the codec payloads into new device tensors. Returns
        {identity: (scalars, tensors)} or None on any mismatch or I/O error.
        Skipping the managed arena matters: first writes into fresh managed
        slabs run at 1.5 GB/s against 8.6 GB/s from the file (2026-09-23)."""
        t0 = time.perf_counter()
        stored = {seg: (offset, n) for seg, offset, n in record.meta["segments"]}
        payloads = {}
        try:
            with self.lock, open(record.stem.with_suffix(".bin"), "rb", buffering=0) as f:
                idx = pages.to(torch.int64)
                per_page = [(f"kv.{i // 2}.{i % 2}", pool) for i, pool in enumerate(kv)]
                per_page += [(f"paux.{name}", pool) for name, pool in (page_aux or {}).items()]
                for seg, pool in per_page:
                    rows = pool.view(torch.uint8).reshape(pool.shape[0], -1)
                    row = rows.shape[1]
                    if seg not in stored or stored[seg][1] != record.own * row:
                        raise ValueError(f"segment {seg} missing or resized")
                    f.seek(stored[seg][0] + start * row)
                    per = max(1, STAGING // row)
                    for a in range(0, record.own - start, per):
                        n = min(per, record.own - start - a)
                        chunk = self._read_chunk(f, n * row)
                        rows.index_copy_(0, idx[a:a + n], chunk.view(n, row))
                for seg, dst in state.items():
                    n = dst.numel() * dst.element_size()
                    if seg not in stored or stored[seg][1] != n:
                        raise ValueError(f"segment {seg} missing or resized")
                    f.seek(stored[seg][0])
                    whole = torch.empty(dst.shape, dtype=dst.dtype, device=dst.device)  # dst may be strided (conv_all[:, slot])
                    self._read_into(f, whole)
                    dst.copy_(whole)
                for identity, scalars, specs in aux_specs:
                    tensors = []
                    for i, (dtype, shape) in enumerate(specs):
                        seg = f"aux.{identity}.{i}"
                        t = torch.empty(shape, dtype=dtype, device=self.device)
                        n = t.numel() * t.element_size()
                        if seg not in stored or stored[seg][1] != n:
                            raise ValueError(f"segment {seg} missing or resized")
                        f.seek(stored[seg][0])
                        self._read_into(f, t)
                        tensors.append(t)
                    payloads[identity] = (tuple(scalars), tensors)
                record.last_access = time.time()
                os.utime(record.stem.with_suffix(".bin"))
        except (OSError, ValueError) as exc:
            self.errors += 1
            self.last_error = repr(exc)
            return None
        self.reads += 1
        self.read_seconds += time.perf_counter() - t0
        return payloads

    def _read_into(self, f, t: torch.Tensor) -> None:
        """Fill a contiguous tensor from the file position, STAGING bytes at a time."""
        flat = t.view(torch.uint8).reshape(-1)
        for a in range(0, flat.numel(), STAGING):
            n = min(STAGING, flat.numel() - a)
            flat[a:a + n].copy_(self._read_chunk(f, n))

    def _read_chunk(self, f, n: int) -> torch.Tensor:
        """n <= STAGING bytes from the file onto the device (a bounce through pinned staging)."""
        view = self._read_staging[:n]
        buf, got = memoryview(view.numpy()), 0
        while got < n:  # an unbuffered read may return less than asked
            k = f.readinto(buf[got:])
            if not k:
                raise ValueError("short read")
            got += k
        if self.device.type != "cuda":
            return view.clone()
        if self._bounce is None:
            self._bounce = torch.empty(STAGING, dtype=torch.uint8, device=self.device)
        self._bounce[:n].copy_(view)
        return self._bounce[:n]

    def stats(self) -> dict:
        return {"records": len(self.records), "gib": round(self.used_bytes / GIB, 2), "budget_gib": round(self.budget / GIB, 1),
                "writes": self.writes, "reads": self.reads, "skipped": self.skipped, "deleted": self.deleted,
                "pending": len(self.pending), "errors": self.errors, "last_error": self.last_error,
                "write_ms_avg": round(1000 * self.write_seconds / max(1, self.writes), 1),
                "read_ms_avg": round(1000 * self.read_seconds / max(1, self.reads), 1)}
