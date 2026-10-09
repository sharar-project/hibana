"""Tensor-parallel communication between two Sparks: the RDMA one-shot all-reduce against NCCL's ring.

Run on both Sparks at once (`./sharar-tp comm` starts both ranks). For each message size it checks that
`sharar.distributed.all_reduce` (the RDMA one-shot up to `ONE_SHOT_MAX_BYTES`, NCCL above) gives NCCL's sum bit for
bit on both ranks, then times both: inside a CUDA graph after about 100 us of GPU work per collective, as serving
runs them, and back to back. Rank 0 prints us per collective and the effective rate (message bytes / time).

`--vllm` adds vLLM's NCCL path (`vllm/distributed/device_communicators/pynccl.py`): its own communicator from a bare
`ncclCommInitRank` through ctypes, out-of-place `ncclAllReduce` on the current stream, the library torch loaded.
`--nccl-env vllm-batch-invariant` sets, before any communicator, the NCCL variables `VLLM_BATCH_INVARIANT` pins
(`vllm/model_executor/determinism/batch_invariant.py`); vLLM sets no NCCL variable otherwise.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import statistics

import torch
import torch.distributed as dist

from sharar import distributed

SIZES_KIB = (10, 40, 160, 640, 1024, 2560, 4096, 8192, 20480)  # 1, 4, 16, 64, 256 rows of 5120 BF16; prefill chunks
COLLECTIVES = 32  # per graph
WORK_CYCLES = 200_000  # torch.cuda._sleep before each collective: about 100 us on the GB10
NCCL_ENVS = {
    "default": {},
    "vllm-batch-invariant": {  # NCCL_LAUNCH_MODE plus pin_nccl_env's pins
        "NCCL_LAUNCH_MODE": "GROUP", "NCCL_COLLNET_ENABLE": "0", "NCCL_NVLS_ENABLE": "0", "NCCL_P2P_NET_DISABLE": "1",
        "NCCL_MIN_NCHANNELS": "1", "NCCL_MAX_NCHANNELS": "1", "NCCL_PROTO": "Simple",
        "NCCL_ALGO": "ring,tree;allreduce:tree", "NCCL_NTHREADS": "1", "NCCL_SOCKET_NTHREADS": "1",
    },
}


class VllmNccl:
    """vLLM's PyNcclCommunicator reduced to its all-reduce: the same NCCL calls with the same arguments."""

    def __init__(self, rank: int):
        path = next(line.split()[-1] for line in open("/proc/self/maps") if "libnccl.so" in line)  # torch's
        self.lib = ctypes.CDLL(path)
        uid = ctypes.create_string_buffer(128)
        if rank == 0:
            self._check(self.lib.ncclGetUniqueId(uid))
        ids = [uid.raw]
        dist.broadcast_object_list(ids, src=0)
        uid = (ctypes.c_byte * 128).from_buffer_copy(ids[0])
        self.comm = ctypes.c_void_p()
        self._check(self.lib.ncclCommInitRank(ctypes.byref(self.comm), 2, uid, rank))

    def _check(self, result: int) -> None:
        if result != 0:
            raise RuntimeError(f"NCCL error {result}")

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        self._check(self.lib.ncclAllReduce(  # ncclBfloat16 = 9, ncclSum = 0
            ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(out.data_ptr()), ctypes.c_size_t(x.numel()),
            ctypes.c_int(9), ctypes.c_int(0), self.comm, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)))
        return out


def graph_us(body, replays: int) -> float:
    """Median time of one replay of a graph capturing `body`."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        body()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / replays)
    graph.reset()
    return statistics.median(times)


def per_collective(reduce, xs, with_work: bool) -> float:
    def body():
        for x in xs:
            if with_work:
                torch.cuda._sleep(WORK_CYCLES)
            reduce(x)
    us = graph_us(body, replays=3)
    if with_work:
        us -= graph_us(lambda: [torch.cuda._sleep(WORK_CYCLES) for _ in xs], replays=3)
    return us / len(xs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tp-address", required=True, help="HOST:PORT on rank 0's host where the ranks meet")
    parser.add_argument("--tp-rank", type=int, required=True)
    parser.add_argument("--vllm", action="store_true", help="also time vLLM's NCCL path")
    parser.add_argument("--nccl-env", choices=NCCL_ENVS, default="default", help="NCCL variables for every communicator")
    args = parser.parse_args()
    os.environ.update(NCCL_ENVS[args.nccl_env])
    distributed.init_distributed(args.tp_rank, 2, args.tp_address)
    rank = args.tp_rank
    paths = [("ring", dist.all_reduce), ("ours", distributed.all_reduce)]
    if args.vllm:
        vllm = VllmNccl(rank)
        paths.append(("vllm", vllm.all_reduce))
    rows = []
    for kib in SIZES_KIB:
        n = kib * 512  # BF16 elements
        partial = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=torch.Generator("cuda").manual_seed(rank + kib))
        ours, nccl = partial.clone(), partial.clone()
        distributed.all_reduce(ours)
        dist.all_reduce(nccl)
        both = torch.empty(2, n, device="cuda", dtype=torch.bfloat16)
        dist.all_gather_single(both, ours)
        exact = torch.equal(ours, nccl) and torch.equal(both[0], both[1])
        if args.vllm:
            exact = exact and torch.equal(vllm.all_reduce(partial), nccl)
        xs = [torch.zeros(n, device="cuda", dtype=torch.bfloat16) for _ in range(COLLECTIVES)]
        times = {name: (per_collective(fn, xs, True), per_collective(fn, xs, False))
                 for name, fn in paths}
        rows.append((kib, n * 2 <= distributed.ONE_SHOT_MAX_BYTES, exact, times))
    if rank == 0:
        print(f"NCCL {'.'.join(map(str, torch.cuda.nccl.version()))}, variables: {args.nccl_env}")
        rate = lambda kib, us: f"{kib * 1024 / us / 1e3:5.1f} GB/s"  # noqa: E731
        print(f"{'message':>9} {'path':>14} {'exact':>6} | {'in a graph, after GPU work':^40} | {'back to back':^20}")
        print(f"{'':>9} {'':>14} {'':>6} | {'NCCL ring':^19} {'sharar':^20} | {'NCCL':>9} {'sharar':>9}")
        for kib, one_shot, exact, t in rows:
            (ring, ring_b2b), (ours, ours_b2b) = t["ring"], t["ours"]
            print(f"{kib:>6} KiB {'RDMA one-shot' if one_shot else 'NCCL ring':>14} {str(exact):>6} | "
                  f"{ring:6.1f} us {rate(kib, ring)}  {ours:6.1f} us {rate(kib, ours)} | {ring_b2b:6.1f} us {ours_b2b:6.1f} us")
        if args.vllm:
            print(f"\n{'message':>9} | vLLM's NCCL path: in a graph after GPU work, back to back")
            for kib, _, _, t in rows:
                print(f"{kib:>6} KiB | {t['vllm'][0]:6.1f} us {rate(kib, t['vllm'][0])}  {t['vllm'][1]:6.1f} us")
    dist.barrier(device_ids=[0])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
