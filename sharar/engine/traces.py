"""Opt-in request traces for draft adaptation.

A trace is the complete record of one served request: prompt and output token
IDs, sampling parameters, every verification round (accepted count, the
draft's proposals and the target's argmax rows) and, for DFlash2 requests, the
projected context features the drafter actually attended to: the last window
of the prompt and every committed response position, copied from the
proposer's ring right after it was written. Those features are what training
needs, and they can only be taken here: the target's prefill numerics depend
on sequence length and chunking, so recomputing them later never reproduces
what the drafter saw while serving.

The sink never blocks the engine thread. Token records are a few lists handed
to one writer thread; feature rows are copied asynchronously into pinned
staging chunks and the writer waits on their CUDA events before writing raw
BF16 files next to the trace file that references them. Files rotate, and the
oldest trace file is pruned together with its feature directory above the
byte cap. Traces are the user's prompts; the directory is only written when
the server was started with one, and any request can opt out with
``trace: false``.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

import torch

FORMAT_VERSION = 2
FILE_PREFIX = "traces-"
FILE_SUFFIX = ".jsonl"
FEATURE_DIM = 5120
CHUNK_ROWS = 512  # 5 MiB of BF16 per pinned staging chunk
DEFAULT_MAX_BYTES = 8192 * 2**20
DEFAULT_ROTATE_BYTES = 64 * 2**20
QUEUE_CAPACITY = 4096


class PinnedPool:
    """Reusable pinned host chunks for device-to-host feature copies."""

    def __init__(self, rows: int = CHUNK_ROWS, dim: int = FEATURE_DIM):
        self.rows, self.dim = rows, dim
        self._free: list[torch.Tensor] = []
        self._lock = threading.Lock()
        self.allocated = 0

    def acquire(self) -> torch.Tensor:
        with self._lock:
            if self._free:
                return self._free.pop()
            self.allocated += 1
        return torch.empty((self.rows, self.dim), dtype=torch.bfloat16, pin_memory=torch.cuda.is_available())

    def release(self, chunk: torch.Tensor) -> None:
        with self._lock:
            self._free.append(chunk)


class FeatureRecorder:
    """Contiguous projected feature rows of one request, staged on the host.

    ``append`` copies device rows for positions ``[position, position + n)``
    without synchronizing. Rows already recorded are skipped; a gap invalidates
    the recording (the request then stores no features).
    """

    def __init__(self, pool: PinnedPool):
        self.pool = pool
        self.start: int | None = None
        self.end: int | None = None
        self.chunks: list[torch.Tensor] = []
        self.used = 0  # rows used in the last chunk
        self.events: list[torch.cuda.Event] = []
        self.valid = True

    @property
    def rows(self) -> int:
        return 0 if self.start is None else self.end - self.start

    def append(self, rows: torch.Tensor, position: int) -> None:
        if not self.valid or rows.shape[0] == 0:
            return
        if self.start is None:
            self.start = self.end = position
        if position < self.end:  # overlap with rows already recorded
            rows = rows[self.end - position:]
            position = self.end
        if position != self.end:
            self.valid = False
            self._release()
            return
        offset = 0
        while offset < rows.shape[0]:
            if not self.chunks or self.used == self.pool.rows:
                self.chunks.append(self.pool.acquire())
                self.used = 0
            n = min(rows.shape[0] - offset, self.pool.rows - self.used)
            self.chunks[-1][self.used:self.used + n].copy_(rows[offset:offset + n], non_blocking=True)
            self.used += n
            offset += n
        self.end += rows.shape[0]
        if rows.is_cuda:
            event = torch.cuda.Event()
            event.record()
            self.events.append(event)

    def _release(self) -> None:
        for chunk in self.chunks:
            self.pool.release(chunk)
        self.chunks, self.used = [], 0

    def write(self, path: Path) -> dict | None:
        """Writer thread: wait for the copies, write raw BF16 rows, free the chunks."""
        try:
            if not self.valid or self.start is None:
                return None
            for event in self.events:
                event.synchronize()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as handle:
                for i, chunk in enumerate(self.chunks):
                    n = self.used if i == len(self.chunks) - 1 else self.pool.rows
                    handle.write(chunk[:n].contiguous().view(torch.uint8).numpy().tobytes())
            return {"start": self.start, "rows": self.rows, "dim": self.pool.dim, "dtype": "bfloat16"}
        finally:
            self._release()


def _record(req) -> dict:
    """Build the trace document for a finished request on the engine thread."""
    now = time.time()
    finish = req.t_finish or time.perf_counter()
    params = req.params
    doc = {
        "v": FORMAT_VERSION,
        "id": req.request_id,
        "arrival": now - (finish - req.t_arrival),
        "finish": now,
        "reason": req.finish_reason,
        "prompt_ids": list(req.prompt_ids),
        "output_ids": list(req.output_ids),
        "cached": int(req.prefix_len),
        "params": {
            "max_new_tokens": params.max_new_tokens, "temperature": params.temperature,
            "top_p": params.top_p, "top_k": params.top_k, "seed": params.seed, "ignore_eos": params.ignore_eos,
        },
        "spec": None,
        "features": None,
    }
    rounds = req.trace_rounds
    if rounds:
        doc["spec"] = {"k": len(rounds[0]["y"]), "rounds": list(rounds)}
    return doc


class TraceSink:
    def __init__(self, directory: str | Path, max_bytes: int = DEFAULT_MAX_BYTES,
                 rotate_bytes: int = DEFAULT_ROTATE_BYTES, features: bool = True):
        if type(max_bytes) is not int or max_bytes < 1 or type(rotate_bytes) is not int or rotate_bytes < 1:
            raise ValueError("trace byte limits must be positive integers")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.rotate_bytes = min(rotate_bytes, max_bytes)
        self.features = features
        self.pool = PinnedPool()
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_CAPACITY)
        self._current: Path | None = None
        self._current_size = 0
        self._sequence = 0
        self.records = 0
        self.dropped = 0
        self.bytes_written = 0
        self.feature_bytes_written = 0
        self.closed = False
        self._thread = threading.Thread(target=self._run, name="sharar-traces", daemon=True)
        self._thread.start()

    # ---- engine thread ----
    def recorder(self) -> FeatureRecorder | None:
        return FeatureRecorder(self.pool) if self.features else None

    def record(self, req) -> bool:
        """Queue one finished request; returns False when the queue is full."""
        recorder = getattr(req, "trace_features", None)
        if self.closed:
            if recorder is not None:
                recorder._release()
            return False
        try:
            self._queue.put_nowait((_record(req), recorder))
        except queue.Full:
            self.dropped += 1
            if recorder is not None:
                recorder._release()
            return False
        return True

    def stats(self) -> dict:
        files = self._files()
        return {
            "dir": str(self.directory), "records": self.records, "dropped": self.dropped,
            "bytes_written": self.bytes_written, "feature_bytes_written": self.feature_bytes_written,
            "files": len(files), "bytes": self._total_bytes(files), "max_bytes": self.max_bytes,
            "features": self.features, "pinned_chunks": self.pool.allocated,
        }

    def close(self, timeout: float = 5.0) -> None:
        if self.closed:
            return
        self.closed = True
        self._queue.put(None)
        self._thread.join(timeout)

    # ---- writer thread ----
    def _files(self) -> list[Path]:
        return sorted(p for p in self.directory.glob(f"{FILE_PREFIX}*{FILE_SUFFIX}") if p.is_file())

    @staticmethod
    def _feature_dir(trace_file: Path) -> Path:
        return trace_file.parent / "features" / trace_file.stem

    def _size(self, trace_file: Path) -> int:
        total = trace_file.stat().st_size if trace_file.exists() else 0
        directory = self._feature_dir(trace_file)
        if directory.is_dir():
            total += sum(f.stat().st_size for f in directory.iterdir() if f.is_file())
        return total

    def _total_bytes(self, files: list[Path]) -> int:
        return sum(self._size(f) for f in files)

    def _open(self) -> Path:
        self._sequence += 1
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        path = self.directory / f"{FILE_PREFIX}{stamp}-{self._sequence:04d}{FILE_SUFFIX}"
        while path.exists():
            self._sequence += 1
            path = self.directory / f"{FILE_PREFIX}{stamp}-{self._sequence:04d}{FILE_SUFFIX}"
        path.touch()
        self._current, self._current_size = path, 0
        return path

    def _prune(self) -> None:
        files = self._files()
        total = self._total_bytes(files)
        for old in files:
            if total <= self.max_bytes or old == self._current:
                break
            total -= self._size(old)
            directory = self._feature_dir(old)
            if directory.is_dir():
                for f in directory.iterdir():
                    f.unlink(missing_ok=True)
                directory.rmdir()
            old.unlink(missing_ok=True)

    def _write(self, doc: dict, recorder: FeatureRecorder | None) -> None:
        if self._current is None or self._current_size >= self.rotate_bytes:
            self._open()
        if recorder is not None:
            path = self._feature_dir(self._current) / f"{doc['id']}.bin"
            info = recorder.write(path)
            if info is not None:
                info["file"] = str(path.relative_to(self.directory))
                doc["features"] = info
                self.feature_bytes_written += info["rows"] * info["dim"] * 2
        line = json.dumps(doc, separators=(",", ":")) + "\n"
        data = line.encode()
        with open(self._current, "ab") as handle:
            handle.write(data)
        self._current_size += len(data)
        self.bytes_written += len(data)
        self.records += 1
        self._prune()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            doc, recorder = item
            try:
                self._write(doc, recorder)
            except Exception:  # noqa: BLE001 - a full disk must not stop serving
                self.dropped += 1
                if recorder is not None:
                    recorder._release()
                logging.exception("trace write failed")


def iter_records(directory: str | Path):
    """Yield trace documents in file order; a trailing partial line is skipped."""
    directory = Path(directory)
    for path in sorted(directory.glob(f"{FILE_PREFIX}*{FILE_SUFFIX}")):
        with open(path, "rb") as handle:
            for raw in handle:
                if not raw.endswith(b"\n"):
                    break
                try:
                    doc = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(doc, dict) and doc.get("v") in (1, FORMAT_VERSION):
                    doc.setdefault("features", None)
                    if doc["features"] is not None:
                        doc["features"]["path"] = str(directory / doc["features"]["file"])
                    yield doc


def load_features(info: dict) -> torch.Tensor:
    """[rows, dim] BF16 CPU tensor from a trace's feature file."""
    data = Path(info["path"]).read_bytes()
    rows, dim = info["rows"], info["dim"]
    if len(data) != rows * dim * 2:
        raise ValueError(f"feature file {info['path']} has {len(data)} bytes, expected {rows * dim * 2}")
    return torch.frombuffer(bytearray(data), dtype=torch.bfloat16).view(rows, dim)


def summarize(directory: str | Path) -> dict:
    """Counts a trainer or operator wants before reading anything."""
    requests = prompt_tokens = output_tokens = rounds = emitted = accepted = with_features = feature_rows = 0
    reasons: dict[str, int] = {}
    for doc in iter_records(directory):
        requests += 1
        prompt_tokens += len(doc["prompt_ids"])
        output_tokens += len(doc["output_ids"])
        reasons[doc["reason"]] = reasons.get(doc["reason"], 0) + 1
        spec = doc.get("spec")
        if spec:
            for r in spec["rounds"]:
                rounds += 1
                accepted += r["a"]
                emitted += r["a"] + 1
        if doc.get("features"):
            with_features += 1
            feature_rows += doc["features"]["rows"]
    return {
        "requests": requests, "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
        "reasons": reasons, "spec_rounds": rounds, "accepted_tokens": accepted,
        "tokens_per_round": emitted / rounds if rounds else None,
        "requests_with_features": with_features, "feature_rows": feature_rows,
    }
