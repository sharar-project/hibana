"""Stop a GPU workload before unified memory runs out.

On the GB10 the GPU allocates from host memory: when MemAvailable stays under the floor (SHARAR_MIN_AVAILABLE_KIB,
default 16 GiB) the weights page and the whole machine stalls. The watch samples every two seconds and acts after
three low samples in a row: it asks the workload to stop (SIGTERM: the server drains, the trainer evaluates) and
kills it 15 seconds later if it is still there. The scheduler also holds new admissions under the floor, so this is
the backstop.

    `sharar serve` and `sharar train` run it as a thread beside themselves (`start_thread`).
    python -m sharar.launch.guard --pid PID   watches another process and its process group (the CI's GPU suite).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import threading
import time
from pathlib import Path

FLOOR_KIB = int(os.environ.get("SHARAR_MIN_AVAILABLE_KIB", str(16 * 2**20)))


def available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable not found")


def watch(stop, alive=lambda: True, log=print, floor_kib: int = FLOOR_KIB, interval: float = 2.0) -> bool:
    """Call stop() after three samples under the floor; returns True if it did, False once alive() turns false."""
    low = 0
    while alive():
        low = low + 1 if available_kib() < floor_kib else 0
        if low >= 3:
            log(f"{dt.datetime.now().isoformat(timespec='seconds')} memory guard: MemAvailable under "
                f"{floor_kib // 2**20} GiB for 6 s; stopping")
            stop()
            return True
        time.sleep(interval)
    return False


def _stop_self() -> None:
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(15)
    os._exit(1)


def start_thread(log=print, stop=_stop_self) -> threading.Thread:
    """Guard this process from a daemon thread."""
    thread = threading.Thread(target=watch, args=(stop,), kwargs={"log": log}, name="sharar-memory-guard", daemon=True)
    thread.start()
    return thread


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sharar.launch.guard", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", type=int, required=True)
    args = ap.parse_args(argv)

    def alive() -> bool:
        try:
            os.kill(args.pid, 0)
            return True
        except ProcessLookupError:
            return False

    def stop() -> None:
        try:
            os.killpg(os.getpgid(args.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        time.sleep(15)
        if alive():
            os.kill(args.pid, signal.SIGKILL)

    return 1 if watch(stop, alive) else 0


if __name__ == "__main__":
    raise SystemExit(main())
