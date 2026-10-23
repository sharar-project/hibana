"""Stop a GPU workload before unified memory runs out.

On the GB10 the GPU allocates from host memory: when MemAvailable runs out the weights page and the whole machine
stalls. The pools are planned to leave the host floor (`floor_kib`) for the OS and other programs; the guard acts at
half of it (`guard_kib`), when those have used more than the plan left them. The watch samples every two seconds and
acts after three low samples in a row: it asks the workload to stop (SIGTERM: the server drains) and kills it 15
seconds later if it is still there. The scheduler also holds new admissions there, so this is the backstop.

    `sharar serve` runs it as a thread beside itself (`start_thread`).
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

from sharar.hardware.gb10 import HOST_FLOOR_KIB


def floor_kib(values=os.environ) -> int:
    """The host floor in KiB from resolved settings (or the environment): SHARAR_HOST_FLOOR_GIB, else the GB10's."""
    return int(float(values.get("SHARAR_HOST_FLOOR_GIB") or HOST_FLOOR_KIB / 2**20) * 2**20)


def guard_kib(values=os.environ) -> int:
    """Where the guard acts, in KiB: SHARAR_MIN_AVAILABLE_KIB, else half the host floor."""
    return int(values.get("SHARAR_MIN_AVAILABLE_KIB") or floor_kib(values) // 2)


def available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable not found")


def watch(stop, alive=lambda: True, log=print, floor: int | None = None, interval: float = 2.0) -> bool:
    """Call stop() after three samples under `floor` KiB (default `guard_kib()`); returns True if it did, False once
    alive() turns false."""
    floor = guard_kib() if floor is None else floor
    low = 0
    while alive():
        low = low + 1 if available_kib() < floor else 0
        if low >= 3:
            log(f"{dt.datetime.now().isoformat(timespec='seconds')} memory guard: MemAvailable under "
                f"{floor // 2**20} GiB for 6 s; stopping")
            stop()
            return True
        time.sleep(interval)
    return False


def _stop_self() -> None:
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(15)
    os._exit(1)


def start_thread(log=print, stop=_stop_self, floor: int | None = None) -> threading.Thread:
    """Guard this process from a daemon thread, under `floor` KiB (default `guard_kib()`)."""
    thread = threading.Thread(target=watch, args=(stop,), kwargs={"log": log, "floor": floor}, name="sharar-memory-guard",
                              daemon=True)
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
