"""`sharar serve <target> --tp 2`: one target tensor-parallel on two DGX Sparks (docs/tensor-parallel.md).

Rank 0 runs here and serves HTTP; rank 1 runs on the peer: the host at the other end of the RoCE link that answers
ssh (or SHARAR_TP_PEER). Before each serve the checkout is mirrored to the same path on the peer (`rsync --delete`,
.git included: it replaces whatever is at that path there), the peer's Python environment is synced (`uv sync
--frozen`), and the model, draft and 3-bit checkpoint are copied to it (rsync copies only what changed). Both ranks
run `sharar engine ... --tp 2 --tp-rank R`; rank 0 broadcasts its EngineConfig, so rank 1 serves with rank 0's
settings and the paths in them must exist on both Sparks.

A rank that ends, however it ends, ends the other within a second (the ranks' TCP link to the primary), and the
pair restarts, up to 5 times in 10 minutes. SIGTERM or Ctrl-C reaches rank 0 alone: it drains its requests, and the
stop it broadcasts ends rank 1; with the SSD tier on, both store their sessions first. SHARAR_TP_PREFIX is a command
both ranks run under, e.g. "nsys profile -o /path/tp2_%h ..." (%h: each host's name).
"""
from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from sharar.launch import guard
from sharar.launch.preflight import pinned_snapshots, snapshot_dir
from sharar.launch.settings import CHECKOUT, Settings

TP_PORT = 29500  # where the ranks meet on rank 0's link address
RESTARTS, RESTART_WINDOW = 5, 600


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kw)


def ssh(peer: str, command: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", peer, command], **kw)


def peer(s: Settings) -> str:
    """SHARAR_TP_PEER, else the first host at the other end of a RoCE link that answers ssh."""
    if s.get("SHARAR_TP_PEER"):
        return s.get("SHARAR_TP_PEER")
    for net in sorted(Path("/sys/class/infiniband").glob("*/device/net/*")):
        out = subprocess.run(["ip", "-4", "neigh", "show", "dev", net.name], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "FAILED" in line or not line.split():
                continue
            ip = line.split()[0]
            if subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", ip, "true"], capture_output=True).returncode == 0:
                return ip
    raise RuntimeError("no peer found: set SHARAR_TP_PEER to the other Spark's address on the RoCE link")


def link(peer_host: str) -> tuple[str, str]:
    """(where the ranks meet, the network device NCCL uses) for the route to the peer."""
    route = subprocess.run(["ip", "-o", "route", "get", peer_host], capture_output=True, text=True, check=True).stdout.split()
    return f"{route[route.index('src') + 1]}:{TP_PORT}", route[route.index("dev") + 1]


def mirror(peer_host: str) -> None:
    """This checkout at the same path on the peer."""
    run(["rsync", "-a", "--delete", "--exclude", ".venv", "--exclude", "__pycache__", "--exclude", "*.egg-info",
         "--exclude", "results", f"{CHECKOUT}/", f"{peer_host}:{CHECKOUT}/"])


# The RDMA all-reduce (kernels/rdma_oneshot.py) builds against the ibverbs headers: the system's, else unpacked into the
# Python environment from the distribution's package (no root needed).
IBVERBS = ('test -e /usr/include/infiniband/verbs.h || test -d ibverbs || '
           '{ apt-get download libibverbs-dev >/dev/null 2>&1 && dpkg -x libibverbs-dev_*.deb ibverbs && rm -f libibverbs-dev_*.deb; }')


def setup(s: Settings, peer_host: str, w3: Path | None, log=print, models: bool = True) -> None:
    """The peer's Python environment and copies of the model, draft and 3-bit checkpoint (only what changed; `models`
    False: the environment only, for the diagnostics)."""
    venv = Path(sys.prefix)
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if ssh(peer_host, "test -x ~/.local/bin/uv || command -v uv", capture_output=True).returncode:
        ssh(peer_host, "mkdir -p ~/.local/bin", check=True)
        run(["scp", "-q", uv, f"{peer_host}:.local/bin/uv"])
    log(f"syncing the Python environment on {peer_host}")
    ssh(peer_host, f"cd {shlex.quote(str(CHECKOUT))} && PATH=$HOME/.local/bin:$PATH UV_PROJECT_ENVIRONMENT={shlex.quote(str(venv))} "
        "uv sync --frozen --quiet", check=True)
    for where in (None, peer_host):
        command = f"cd {shlex.quote(str(venv))} && {{ {IBVERBS}; }}"
        r = ssh(where, command) if where else subprocess.run(["bash", "-c", command])
        if r.returncode:
            log(f"note: no ibverbs headers on {where or 'this Spark'}; the RDMA all-reduce cannot build (SHARAR_TP_ALLREDUCE=nccl works without)")
    if not models:
        return
    copies: list[Path] = []
    for cache, repo, revision in pinned_snapshots(s):
        snap = snapshot_dir(cache, repo, revision)
        if snap is not None:
            copies.append(snap.parents[1])  # models--<repo>, into its cache root
    if s.draft_dir is not None:
        copies.append(s.draft_dir)
    if w3 is not None:  # the served checkpoint only: w3-checkpoints/ also holds other targets' and older calibrations
        copies.append(w3)
    copy(peer_host, copies, log)
    if s.traces is not None:
        ssh(peer_host, f"mkdir -p {shlex.quote(str(s.traces))}", check=True)


def copy(peer_host: str, paths: list[Path], log=print) -> None:
    """Each directory at the same path on the peer (rsync copies only what changed)."""
    for src in paths:
        ssh(peer_host, f"mkdir -p {shlex.quote(str(src.parent))}", check=True)
        log(f"copying {src} to {peer_host}")
        run(["rsync", "-a", str(src), f"{peer_host}:{src.parent}/"])


def rank_env(s: Settings, w3: Path | None, device: str) -> dict[str, str]:
    venv_bin = str(Path(sys.prefix) / "bin")
    env = {**s.engine_env(w3), "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CUDA_HOME": "/usr/local/cuda", "NCCL_SOCKET_IFNAME": device,
           "PATH": f"{venv_bin}:/usr/local/cuda/bin:/usr/bin:/bin", "PYTHONUNBUFFERED": "1"}
    env.setdefault("SHARAR_TP_ALLREDUCE", "rdma")
    return env


class Pair:
    """Rank 0 here and rank 1 on the peer, each `python -m <module> ARGS --tp-rank R`."""

    def __init__(self, peer_host: str, env: dict[str, str], module: str, args: list[str], show_rank1: bool = True):
        self.peer, self.env, self.module, self.args, self.show_rank1 = peer_host, env, module, args, show_rank1
        self.rank0: subprocess.Popen | None = None
        self.rank1: subprocess.Popen | None = None

    def command(self, rank: int) -> list[str]:
        prefix = shlex.split(os.environ.get("SHARAR_TP_PREFIX", ""))
        return [*prefix, str(Path(sys.prefix) / "bin" / "python"), "-u", "-m", self.module, *self.args, "--tp-rank", str(rank)]

    def start(self) -> None:
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
        remote = f"cd {shlex.quote(str(CHECKOUT))} && env {exports} {shlex.join(self.command(1))}"
        # Ctrl-C signals the terminal's whole process group. Both ranks start with SIGINT ignored (an ignored signal
        # stays ignored across exec, and Python and ssh keep it so), so only this process acts on it and passes SIGTERM
        # to rank 0 alone: rank 1's ssh must stay up until rank 0's stop has ended rank 1 (after both stored their sessions).
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            # A terminal for rank 1 (-tt): it gets SIGHUP and exits when its ssh ends, which ends with this process.
            self.rank1 = subprocess.Popen(["ssh", "-tt", self.peer, remote], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True, errors="replace")
            # The engine's settings are exactly the resolved ones (`rank_env`), as on one Spark (`run.install_env`): none of
            # the caller's SHARAR_* settings, e.g. SHARAR_TRACES=off, which the engine would read as a trace directory.
            base = {k: v for k, v in os.environ.items() if not k.startswith("SHARAR_")}
            self.rank0 = subprocess.Popen(self.command(0), env={**base, **self.env}, cwd=CHECKOUT)
        finally:
            signal.signal(signal.SIGINT, previous)
        threading.Thread(target=self._relay, args=(self.rank1.stdout, self.show_rank1), daemon=True).start()

    @staticmethod
    def _relay(stream, show: bool) -> None:
        for line in stream:
            if show:
                print(f"[rank 1] {line.rstrip()}", flush=True)

    def wait(self) -> int:
        """Rank 0's exit status, once rank 1 is gone too: rank 1 ends by itself when rank 0's stop reaches it, else
        wait() terminates it after 30 s. With the SSD tier on, both ranks store their sessions in lockstep first."""
        code = self.rank0.wait()
        if self.rank1.poll() is None:
            try:
                self.rank1.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.rank1.terminate()
                self.rank1.wait()
        return code

    def signal_rank0(self, sig=signal.SIGTERM) -> None:
        if self.rank0 is not None and self.rank0.poll() is None:
            self.rank0.send_signal(sig)


def serve(s: Settings, w3: Path | None, log=print) -> int:
    """Both ranks in the foreground, restarted when either fails; returns rank 0's last exit status."""
    sys.stdout.reconfigure(line_buffering=True)  # this process's lines in order with the ranks'
    peer_host = peer(s)
    log(f"two Sparks: this one and {peer_host}")
    mirror(peer_host)
    setup(s, peer_host, w3, log)
    address, device = link(peer_host)
    env = rank_env(s, w3, device)
    args = ["engine", *s.engine_args(tp=2), "--tp-address", address]
    stopping = False
    pair: Pair | None = None

    def on_signal(signum, frame):  # noqa: ARG001
        nonlocal stopping
        stopping = True
        if pair is not None:
            pair.signal_rank0()  # rank 0 drains; the stop it broadcasts ends rank 1

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    guard.start_thread(log=log, stop=lambda: on_signal(signal.SIGTERM, None))
    starts: list[float] = []
    while True:
        mirror(peer_host)
        pair = Pair(peer_host, env, "sharar", args)
        pair.start()
        code = pair.wait()
        if code == 0 or stopping:
            return code
        now = time.monotonic()
        starts = [t for t in starts if now - t < RESTART_WINDOW] + [now]
        if len(starts) >= RESTARTS:
            log(f"the pair failed {len(starts)} times in 10 minutes (exit {code}); not restarting")
            return code
        log(f"the pair exited (status {code}); restarting in 5 s")
        time.sleep(5)


def diagnostic(module: str, args: list[str], s: Settings, log=print) -> int:
    """A two-rank profiler (profiling.tp_comm, profiling.tp_draft_head) on both Sparks; rank 1's output is dropped."""
    sys.stdout.reconfigure(line_buffering=True)
    peer_host = peer(s)
    mirror(peer_host)
    setup(s, peer_host, None, log, models=False)
    address, device = link(peer_host)
    env = rank_env(s, None, device)
    pair = Pair(peer_host, env, module, [*args, "--tp-address", address], show_rank1=False)
    pair.start()
    return pair.wait()
