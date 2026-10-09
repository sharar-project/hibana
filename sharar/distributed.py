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

from dataclasses import dataclass

import torch
import torch.distributed as dist

from sharar.kernels import rdma_oneshot


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


def close_distributed() -> None:
    """Leave the group once this rank's engine has stopped."""
    dist.destroy_process_group()


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
