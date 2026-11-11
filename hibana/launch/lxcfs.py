"""PyTorch inside an LXD/Incus container whose CPU set has holes.

lxcfs numbers /proc/cpuinfo's processors 0..N-1 while /sys/devices/system/cpu keeps the host's numbering (e.g. online
0-11,14-19 of possible 0-19); PyTorch's cpuinfo indexes one by the other and corrupts the heap on `import torch`
("malloc(): corrupted top size"). `wrap` runs a command with CPU lists that agree with /proc/cpuinfo, in a private user
and mount namespace; nothing outside the command changes. Elsewhere it returns the command unchanged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_SCRIPT = ('list=$(mktemp); echo "0-$(($(grep -c ^processor /proc/cpuinfo) - 1))" > "$list"; '
           'for f in possible present online; do mount --bind "$list" /sys/devices/system/cpu/$f; done; rm -f "$list"; exec "$@"')


def needed() -> bool:
    try:
        return " /sys/devices/system/cpu fuse.lxcfs " in Path("/proc/mounts").read_text()
    except OSError:
        return False


def wrap(command: list[str]) -> list[str]:
    if not needed():
        return command
    return ["unshare", "--user", "--map-root-user", "--mount", "bash", "-c", _SCRIPT, "lxcfs-wrap", *command]


def reexec_if_needed() -> None:
    """Restart this process inside the namespace before anything imports torch (once)."""
    if needed() and not os.environ.get("HIBANA_LXCFS_WRAPPED"):
        os.environ["HIBANA_LXCFS_WRAPPED"] = "1"
        command = wrap([sys.executable, *sys.orig_argv[1:]])
        os.execvp(command[0], command)
