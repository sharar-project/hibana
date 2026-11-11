"""`sharar serve --detach`: the same command as a user systemd service.

The service runs this checkout's `sharar` with the caller's HIBANA_* settings and config file; its output goes to
$HIBANA_RESULTS/logs/<service>.log (the previous run's is .log.1); `systemctl --user stop` sends SIGTERM, so the server
drains (and with the SSD tier on, stores its cached prefixes). SIGTERM reaches the main process only (KillMode=mixed):
on two Sparks that is the supervisor, which passes it to rank 0 alone, and rank 1's ssh must stay up until rank 0's
stop has ended rank 1 (launch/tp.py); whatever is left when the main process exits is killed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from hibana.launch.settings import config_file

MILESTONES = ("loading", "  weights loaded", "  w3: ", "  captured verify graph for batch", "  warmup done", "serving on",
              "downloading", "fetching", "[rank 1]   weights loaded")


def systemd_available() -> bool:
    return shutil.which("systemd-run") is not None and subprocess.run(
        ["systemctl", "--user", "show-environment"], capture_output=True).returncode == 0


def running(unit: str) -> bool:
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit], capture_output=True).returncode == 0


def start(unit: str, argv: list[str], log: Path, description: str) -> None:
    """Run `python -m hibana ARGV` as the user systemd service UNIT."""
    if not systemd_available():
        raise RuntimeError(f"--detach needs a user systemd session: log in with a full session, or sudo loginctl enable-linger {os.environ.get('USER', '$USER')}")
    if running(unit):
        raise RuntimeError(f"{unit} is already running (sharar stop first)")
    subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.is_file():
        log.replace(log.with_name(log.name + ".1"))
    env = {k: v for k, v in os.environ.items() if k.startswith(("HIBANA_", "HF_", "CUDA_VISIBLE", "NCCL_"))}
    conf = config_file()
    if conf is not None:
        env["HIBANA_CONFIG_FILE"] = str(conf)
    env.update(PATH=os.environ.get("PATH", "/usr/bin:/bin"), PYTHONUNBUFFERED="1")
    setenv = [a for k, v in env.items() for a in ("--setenv", f"{k}={v}")]
    cmd = ["systemd-run", "--user", "--quiet", "--collect", "--unit", unit, "--description", description,
           "--working-directory", os.getcwd(), "--property", "KillSignal=SIGTERM", "--property", "KillMode=mixed",
           "--property", "TimeoutStopSec=120",
           "--property", f"StandardOutput=append:{log}", "--property", f"StandardError=append:{log}", *setenv,
           sys.executable, "-m", "hibana", *argv]
    subprocess.run(cmd, check=True)


def stop(unit: str) -> bool:
    if not running(unit):
        return False
    subprocess.run(["systemctl", "--user", "stop", unit])
    subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True)
    return True


def follow_startup(unit: str, log: Path, url: str) -> int:
    """Print the server's startup milestones from its log until it serves, fails or exits."""
    shown = 0
    while True:
        lines = log.read_text(errors="replace").splitlines() if log.is_file() else []
        for line in lines[shown:]:
            if line.startswith(MILESTONES) or line.startswith("Sharar ") or "WARNING" in line or line.startswith("note:"):
                print(line)
            if line.startswith("serving on"):
                print(f"ready: {url} (sharar chat, sharar launch claude, sharar stop; logs: sharar logs -f)")
                return 0
        shown = len(lines)
        if not running(unit) or any(line.startswith(("Traceback", "[rank 1] Traceback")) for line in lines):
            print("\n".join(lines[-30:]))
            print(f"{unit} failed to start; full log: {log}")
            return 1
        time.sleep(2)
