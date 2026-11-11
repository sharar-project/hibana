"""Tensor parallelism: this process's rank, and the collectives between ranks.

SGLang's layout, named after mini-sglang's `minisgl.distributed`: every rank runs the whole engine on its own GPU
(one per Spark) and holds 1/size of each sharded projection (`weights/sharding.py`). Row-parallel outputs are
partial sums joined with an NCCL all-reduce, and the vocabulary-parallel lm_head's logits are all-gathered, so every
rank holds bitwise-identical logits and samples the same tokens. Between two ranks, decode-sized all-reduces take
the RDMA one-shot (`kernels/rdma_oneshot.py`: one exchange of the partials, 19 us at 160 KiB against NCCL's 50).
The primary rank (0) serves HTTP and broadcasts
each command it applies (`engine/loop.py`), so every rank's scheduler takes the same steps; the few decisions that
read local memory take the minimum over ranks (`all_reduce_min`). Host values travel over NCCL too: on these links
a gloo round trip costs about 5 ms, an NCCL one about 0.2 ms.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
from dataclasses import dataclass

import torch
import torch.distributed as dist

from hibana.kernels import rdma_oneshot


@dataclass(frozen=True)
class DistributedInfo:
    rank: int
    size: int

    def is_primary(self) -> bool:
        return self.rank == 0


_TP_INFO = DistributedInfo(0, 1)
_ONE_SHOT = None  # the RDMA one-shot all-reduce between two ranks
# Above 4 MiB NCCL's ring keeps up (a tie at 8 MiB, ahead at a 20 MiB prefill chunk); below, the one-shot is 2.5-5x
# faster (profiling/tp_comm.py): decode and verification at up to 25 x 16 rows, prefill chunks of up to 409 tokens.
ONE_SHOT_MAX_BYTES = 4 << 20


def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    return _TP_INFO


def init_distributed(rank: int, size: int, address: str, allreduce: str = "rdma") -> None:
    """Join the tensor-parallel group whose primary rank listens on `address` (host:port). `allreduce` "nccl" runs
    every all-reduce on NCCL, as vLLM does; "rdma" takes the one-shot for decode-sized ones between two ranks."""
    global _ONE_SHOT
    set_tp_info(rank, size)
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=f"tcp://{address}", rank=rank, world_size=size,
                            device_id=torch.device("cuda", 0))
    if size == 2 and allreduce == "rdma":
        _ONE_SHOT = rdma_oneshot.create(ONE_SHOT_MAX_BYTES)
        infos = [None, None]
        dist.all_gather_object(infos, _ONE_SHOT.info())
        _ONE_SHOT.connect(infos[1 - rank])
    if size > 1:
        _watch_peers(rank, size, address)


# A rank that dies leaves the others inside a collective until NCCL's timeout (10 minutes). Each rank above 0 holds
# a TCP connection to the primary; the kernel closes it the moment a process ends, however it ends, and keepalive
# probes notice a host that vanished. Either side then ends its own process, so the launcher restarts the pair.
_PEER_EXIT_EXPECTED = threading.Event()
_KEEPALIVE = ((socket.TCP_KEEPIDLE, 2), (socket.TCP_KEEPINTVL, 1), (socket.TCP_KEEPCNT, 3))  # a lost host: ~5 s


def _watch_peers(rank: int, size: int, address: str) -> None:
    host, port = address.rsplit(":", 1)
    if rank == 0:
        server = socket.create_server((host, int(port) + 1))
        dist.barrier()  # listening before the others connect
        peers = [server.accept()[0] for _ in range(size - 1)]
        server.close()
    else:
        dist.barrier()
        peers = [socket.create_connection((host, int(port) + 1), timeout=60)]
    for sock in peers:
        sock.settimeout(None)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for opt, value in _KEEPALIVE:
            sock.setsockopt(socket.IPPROTO_TCP, opt, value)
        threading.Thread(target=_watch, args=(sock,), name="tp-peer-watch", daemon=True).start()


def _watch(sock: socket.socket) -> None:
    try:
        sock.recv(1)  # nothing is ever sent: this returns or raises only when the peer is gone
    except OSError:
        pass
    if not _PEER_EXIT_EXPECTED.is_set():
        print(f"tensor-parallel rank {_TP_INFO.rank}: a peer rank exited; ending this rank so the pair restarts",
              file=sys.stderr, flush=True)
        os._exit(1)


def expect_peer_exit() -> None:
    """The pair is stopping on purpose (the primary broadcast its stop): a peer that exits now is not a failure."""
    _PEER_EXIT_EXPECTED.set()


def close_distributed() -> None:
    """End this rank once its engine has stopped and its cache persisted. NCCL's destroy_process_group can wait
    forever on two Sparks (measured: both ranks blocked in it after a clean stop), and nothing is left to tear down
    that the process exit does not release, so the rank exits here."""
    expect_peer_exit()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def div_even(n: int, size: int) -> int:
    if n % size:
        raise ValueError(f"{n} does not split evenly across {size} tensor-parallel ranks")
    return n // size


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    """The sum of every rank's partial `x`, in place: the one-shot for decode-sized BF16 tensors between two ranks,
    NCCL's ring otherwise (prefill chunks, where bandwidth rather than latency sets the time)."""
    if _ONE_SHOT is not None and x.dtype == torch.bfloat16 and x.numel() * x.element_size() <= ONE_SHOT_MAX_BYTES:
        _ONE_SHOT.all_reduce(x)
    elif _TP_INFO.size > 1:
        dist.all_reduce(x)
    return x


def all_gather(x: torch.Tensor) -> torch.Tensor:
    """Every rank's `x` [..., N] concatenated in rank order along the last dimension: [..., size * N]."""
    if _TP_INFO.size == 1:
        return x
    out = x.new_empty((_TP_INFO.size, *x.shape))
    dist.all_gather_single(out, x.contiguous())
    return out.movedim(0, -2).reshape(*x.shape[:-1], _TP_INFO.size * x.shape[-1])


def broadcast_pyobj(obj):
    """The primary rank's `obj` (any picklable value) on every rank."""
    if _TP_INFO.size == 1:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def all_reduce_min(*values: int) -> list[int]:
    """The element-wise minimum over ranks of host-side integers: one answer to a question every rank answers
    from its own memory (pool sizes, the memory floor)."""
    if _TP_INFO.size == 1:
        return list(values)
    t = torch.tensor(values, dtype=torch.int64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return t.tolist()


def all_reduce_max(values: list[float]) -> list[float]:
    """The element-wise maximum over ranks of host-side floats: one value every rank measured on its own GPU (the
    depth policy's costs and round times; a tensor-parallel round takes the slower rank's time)."""
    if _TP_INFO.size == 1:
        return list(values)
    t = torch.tensor(values, dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


def common_values(values: list[int]) -> list[int]:
    """The values every rank holds, in this rank's order: the SSD tier's stored prefixes after a restart (no rank
    restores a prefix another rank lacks)."""
    if _TP_INFO.size == 1:
        return values
    held = [None] * _TP_INFO.size
    dist.all_gather_object(held, list(values))
    common = set.intersection(*(set(h) for h in held))
    return [v for v in values if v in common]
