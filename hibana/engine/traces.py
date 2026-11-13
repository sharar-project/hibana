"""Opt-in request traces.

A trace is the token-level record of one served request: prompt and output
token IDs, sampling parameters and every verification round (accepted count,
the draft's proposals and the target's argmax rows). They are for measuring
speculation on real traffic (acceptance per depth, the input of adaptive depth
calibration); `hibana traces` summarizes a directory.

The sink never blocks the engine thread: records are a few lists handed to one
writer thread. Files rotate, and the oldest file is pruned above the byte cap.
Traces are the user's prompts; the directory is only written when the server
was started with one, and any request can opt out with ``trace: false``.

With ``HIBANA_TRACE_CONFIDENCE=1`` (sequential MTP drafts), each round record
also carries the draft's top-1 probability ``p1`` and top-1 minus top-2
margin ``m`` per drafted position (the input for adaptive depth).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

FORMAT_VERSION = 4  # 3: params carry the acceptance rule; rounds may carry the draft confidence ("p1", "m"); 4: no feature files
FILE_PREFIX = "traces-"
FILE_SUFFIX = ".jsonl"
DEFAULT_MAX_BYTES = 32768 * 2**20
DEFAULT_ROTATE_BYTES = 64 * 2**20
QUEUE_CAPACITY = 4096


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
            "accept_top_k": params.accept_top_k, "accept_margin": params.accept_margin,
        },
        "spec": None,
    }
    if params.accept_soft > 0:  # a sampled request's relaxed rule; absent: exact
        doc["params"]["accept_soft"] = params.accept_soft
    rounds = req.trace_rounds
    if rounds:
        doc["spec"] = {"k": len(rounds[0]["y"]), "rounds": list(rounds)}
    return doc


class TraceSink:
    def __init__(self, directory: str | Path, max_bytes: int = DEFAULT_MAX_BYTES,
                 rotate_bytes: int = DEFAULT_ROTATE_BYTES):
        if type(max_bytes) is not int or max_bytes < 1 or type(rotate_bytes) is not int or rotate_bytes < 1:
            raise ValueError("trace byte limits must be positive integers")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.rotate_bytes = min(rotate_bytes, max_bytes)
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_CAPACITY)
        self._current: Path | None = None
        self._current_size = 0
        self._sequence = 0
        self.records = 0
        self.dropped = 0
        self.bytes_written = 0
        self.closed = False
        # Scan once before serving starts. Only the writer mutates the inventory;
        # the engine reads an immutable snapshot without filesystem operations.
        self._sizes = {path: self._size(path) for path in self._files()}
        self._stored_bytes = sum(self._sizes.values())
        self._disk_usage = (len(self._sizes), self._stored_bytes)
        self._thread = threading.Thread(target=self._run, name="hibana-traces", daemon=True)
        self._thread.start()

    # ---- engine thread ----
    def record(self, req) -> bool:
        """Queue one finished request; returns False when the queue is full."""
        if self.closed:
            return False
        try:
            self._queue.put_nowait(_record(req))
        except queue.Full:
            self.dropped += 1
            return False
        return True

    def stats(self) -> dict:
        files, nbytes = self._disk_usage
        return {
            "dir": str(self.directory), "records": self.records, "dropped": self.dropped,
            "bytes_written": self.bytes_written, "files": files, "bytes": nbytes, "max_bytes": self.max_bytes,
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
        """Where traces recorded before format 4 kept their feature files: counted and pruned with their file."""
        return trace_file.parent / "features" / trace_file.stem

    def _size(self, trace_file: Path) -> int:
        total = trace_file.stat().st_size if trace_file.exists() else 0
        directory = self._feature_dir(trace_file)
        if directory.is_dir():
            total += sum(f.stat().st_size for f in directory.iterdir() if f.is_file())
        return total

    def _open(self) -> Path:
        self._sequence += 1
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        path = self.directory / f"{FILE_PREFIX}{stamp}-{self._sequence:04d}{FILE_SUFFIX}"
        while path.exists():
            self._sequence += 1
            path = self.directory / f"{FILE_PREFIX}{stamp}-{self._sequence:04d}{FILE_SUFFIX}"
        path.touch()
        self._sizes[path] = 0
        self._current, self._current_size = path, 0
        return path

    def _prune(self) -> None:
        for old in list(self._sizes):
            if self._stored_bytes <= self.max_bytes or old == self._current:
                break
            directory = self._feature_dir(old)
            if directory.is_dir():
                for f in directory.iterdir():
                    f.unlink(missing_ok=True)
                directory.rmdir()
            old.unlink(missing_ok=True)
            self._stored_bytes -= self._sizes.pop(old)
        self._disk_usage = (len(self._sizes), self._stored_bytes)

    def _write(self, doc: dict) -> None:
        if self._current is None or self._current_size >= self.rotate_bytes:
            self._open()
        data = (json.dumps(doc, separators=(",", ":")) + "\n").encode()
        with open(self._current, "ab") as handle:
            handle.write(data)
        self._current_size += len(data)
        self.bytes_written += len(data)
        self._sizes[self._current] += len(data)
        self._stored_bytes += len(data)
        self.records += 1
        self._prune()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._write(item)
            except Exception:  # noqa: BLE001 - a full disk must not stop serving
                self.dropped += 1
                logging.exception("trace write failed")
                # A failed write can leave a partial line on disk. Reconcile that file on the writer thread.
                try:
                    if self._current is not None:
                        size = self._size(self._current)
                        self._current_size = size
                        self._stored_bytes += size - self._sizes[self._current]
                        self._sizes[self._current] = size
                    self._prune()
                except OSError:
                    logging.exception("trace storage accounting failed")


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
                if isinstance(doc, dict) and doc.get("v") in (1, 2, 3, FORMAT_VERSION):
                    yield doc


def summarize(directory: str | Path) -> dict:
    """Counts an operator wants before reading anything."""
    requests = prompt_tokens = output_tokens = rounds = emitted = accepted = 0
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
    return {
        "requests": requests, "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
        "reasons": reasons, "spec_rounds": rounds, "accepted_tokens": accepted,
        "tokens_per_round": emitted / rounds if rounds else None,
    }
