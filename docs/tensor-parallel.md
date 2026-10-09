# Tensor parallel: the 27B on two Sparks

```bash
./sharar-ctl serve 27b --tp 2     # rank 0 here (http://127.0.0.1:30001), rank 1 on the peer; --exact, --balanced, --throughput
./sharar-ctl status               # stop, logs [-f], chat and launch claude work as on one Spark
./sharar-tp comm                  # the RDMA one-shot all-reduce against NCCL between the two Sparks
```

`sharar-ctl` hands two Sparks to `./sharar-tp`, as it hands one to `run-sharar.sh`. The peer is the host at the other
end of the RoCE link (or `SHARAR_TP_PEER`). The first serve sets both Sparks up (`./sharar-tp setup`: the uv venv on
each; the model, draft and 3-bit checkpoint here, copied to the peer); every serve mirrors the checkout to the same
path on the peer (`rsync --delete`, `.git` included: it replaces whatever is at that path there) and starts
`sharar serve --tp 2` on both with sharar-ctl's serving profiles (`fast` by default; `SHARAR_W3_DECODE=0` serves the
native NVFP4/FP8 weights). Rank 1 serves with rank 0's configuration: when the ranks meet, rank 0 broadcasts the
`EngineConfig` it read from its environment, so the paths in it (draft, 3-bit checkpoint) must exist on both Sparks.
`stop` signals rank 0 alone; it drains its requests, and the stop command it broadcasts ends rank 1. In LXD/Incus containers
whose CPU set has holes, `tools/lxcfs-run.sh` (used by the launcher) keeps `import torch` from corrupting its heap.

## Design

SGLang's layout, with mini-sglang's names (`sharar/distributed.py`, `sharar/weights/sharding.py`):

| Projection | Role | Per rank |
|---|---|---|
| attention q\|k\|v | `ColumnParallel((2q, kv, kv))` | 12 of 24 heads, 2 of 4 KV heads |
| attention o_proj, GDN out_proj, MLP down | `ROW_PARALLEL` + all-reduce (RDMA one-shot) | half the input columns |
| GDN in_proj q\|k\|v\|z, b\|a, conv, A_log, dt_bias | `ColumnParallel` by head | 8 of 16 key heads, 24 of 48 value heads |
| MLP gate\|up | `ColumnParallel((I, I))` | 8704 of 17408 |
| lm_head | `ColumnParallel((V,))` + all-gather | half the vocabulary |
| embedding, norms, the DFlash2 draft | replicated | |

The loaders cut each checkpoint tensor by its role (`Shard.rows` / `Shard.cols`): FP8 and NVFP4 weights and block
scales, and the calibrated 3-bit copies' packed fragments (32-row and 128-column units), so every rank serves its part
of the same calibrated weights. `Target.shard` gives the pools, kernels and graphs each rank's geometry.

Every rank runs the whole engine (scheduler, speculation, CUDA graphs). The primary rank owns HTTP; at every point
where it applies commands (`EngineLoop._publish`) it broadcasts them, and the other ranks apply the same ones there, so
all schedulers take identical steps and the logits (all-reduced, then all-gathered) are identical on every rank.
Decisions that read local memory take the minimum over ranks. Request constraints travel as their grammar source and
compile again on each rank; sampled requests get one seed for all ranks. An engine failure ends the process; the
other rank waits in its next collective until it times out (10 minutes, NCCL's default and the one-shot's), so
restart the pair (`./sharar-ctl stop`, then `serve`).

## Results (2026-10-07, two GB10 containers linked by RoCE)

Single-stream greedy decode over seven prompts (chat, code, math, multilingual, prose, structured, a 12K-character
code prompt), median tokens per second:

| Settings | TP=1 | TP=2, NCCL all-reduce | TP=2, RDMA one-shot |
|---|---:|---:|---:|
| exact acceptance, NVFP4/FP8 weights | 42.4 | 72.9 | 80.2 |
| relaxed acceptance, NVFP4/FP8 weights | 52.6 | 71.6 | |
| `fast` profile (relaxed acceptance, 3-bit decode copies) | 77.1 | 105.9 | **119.1** |

A speculative round of the `fast` profile takes 41 ms on two Sparks against 67 ms on one (46 ms with NCCL's
all-reduce); every prompt is faster (x1.39 to x1.79). With exact acceptance the one-shot's outputs are bitwise those of
NCCL's all-reduce. GSM8K (first 300, zero-shot, greedy): 95.3% on one Spark and 95.0% on two with the `fast` profile
(the one-shot and NCCL alike), 95.0% and 94.3% with exact acceptance; the final answers agree on 282-284 of 300.

Without the 3-bit copies (native NVFP4/FP8 weights, relaxed acceptance) two Sparks decode at 76.9 tok/s greedy and
69.8 sampled (temperature 0.7) against 53.6 and 39.6 on one; four concurrent requests of 384 tokens reach 143.2 tok/s
in aggregate against 92.8 (the `fast` profile: 199.1 against 135.1). Relaxed acceptance accepts the same on both
(5.64 and 5.56 tokens per round greedy, 4.27 with exact acceptance), and GSM8K stays at 94.7-95.0% greedy and sampled.

Cold prefill (a fresh prompt each time, `max_tokens` 1, median of three): x1.38 at 1K tokens (2304 -> 3188 tok/s),
x1.64 at 8.6K, x1.70 at 28K (1751 -> 2979 tok/s; time to first token 16.1 -> 9.5 s). A 2048-token chunk's 128
all-reduces of 20 MB run on NCCL's ring near the link's 23.4 GB/s, about 0.12 s of a 0.64 s chunk, not overlapped
with compute.

TP=2 is not bitwise TP=1: summing two partial products in another order moves the logits of this NVFP4/FP8 model as
much as any other reassociation does. Teacher-forced over 6K tokens, TP=2 agrees with TP=1 on 88-95% of argmax tokens
(mean KL 0.02-0.04); so does TP=1 with only its row-parallel FP8 sums split in two on one GPU, and TP=1 itself
changes 6 of the 7 greedy outputs above when the same prompts are decoded in a batch of four. The flips are near ties
(a median top-two margin of 0.25 nats, one or two BF16 steps of the logits). TP=1 is unchanged by this code: its
logits are bitwise those of the commit before it.

## Profiling

```bash
SHARAR_TP_PREFIX="nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  --delay=420 --duration=20 -o /path/tp2_%h" ./sharar-ctl serve 27b --tp 2   # then keep requests flowing
```

Both ranks run under the prefix; `%h` names each host's report, and nsys ends the session after the window. In 20 s
of single-stream `fast` traffic with NCCL's all-reduce (385 rounds), 96% of kernel time ran from CUDA graphs and the GPU was busy 95% of the
time. A round replays a verify graph (1388 kernels with its 128 all-reduces and the logits all-gather, 38 ms under
node tracing), a DFlash2 draft graph (210 kernels, one all-gather, 7-8 ms) and an accept graph (1.7 ms). Kernel time:
3-bit GEMMs 58%, NCCL 18% (about 65 us per collective including skew), NVFP4/FP8 GEMMs 13%, the rest under 5% each.

## The link

Each Spark's ConnectX-7 port reaches the GB10 through two PCIe Gen5 x4 halves, the two RDMA devices NCCL uses
together: about 109 Gb/s each in raw RDMA, 21.8 GB/s (174 Gb/s) for a large NCCL all-reduce. DGX Spark has no
GPUDirect RDMA (the driver reports neither GPUDirect RDMA nor dma-buf support: device memory of the integrated GPU
is not coherently accessible to PCIe devices), so NCCL stages through registered host buffers and every collective
pays a fixed 20-45 us however small it is. A decode or verification all-reduce moves 160 KiB, so its time is that
fixed cost: about 4.5 GB/s effective. Moving 160 KiB each way across both devices at once takes about 12 us.

Two ranks need one exchange, not a ring's two steps, and the staging can be NVIDIA's own pattern for the GB10:
`kernels/rdma_oneshot.py` registers cudaHostAlloc buffers with both RDMA devices; a kernel copies the partial in and
rings a doorbell, a busy-polling CPU thread posts the RDMA writes (half per device, each followed by a flag), and the
peer's kernel waits on the flags and adds. Per collective inside a CUDA graph, after ~100 us of GPU work as in
serving:

| Message | NCCL ring | NCCL one-shot (send/recv) | RDMA one-shot |
|---|---:|---:|---:|
| 10 KiB | 31.5 us | 29.1 us | **8.1 us** |
| 160 KiB (verification, one request) | 50.7 us | 42.8 us | **18.7 us** |
| 640 KiB (four requests) | 99.0 us | 74.6 us | **50.3 us** |

No NCCL setting (protocol, channels, graph mixing) went below 45 us at 160 KiB. The one detail that mattered: only one
GPU thread polls the flags in host memory; with every block polling, the NIC's flag writes contended with them and
the wait grew from 4 to 17 us. The one-shot takes messages of up to 4 MiB (verification of up to 25 requests, prefill
chunks of up to 409 tokens): 263 us at 4 MiB against NCCL's 721 (307 with `NCCL_PROTO=Simple`); at 8 MiB they tie and
a 20 MiB prefill chunk keeps NCCL's ring (1,067 us against 1,420), where bandwidth sets the time. Fusing the two kernels or
pipelining copy, wire and add in chunks gained nothing at 160 KiB and at most 5% above it, so the transport stays one
copy, one exchange and one add. Raising the threshold from 1 to 4 MiB took the throughput profile at 16 concurrent
requests (verification of 2.5 MiB) from 268.0 to 298.8 tok/s in aggregate, its rounds from 146.9 to 126.6 ms. NCCL's own
defaults serve what remains on NCCL as well as any channel setting. The transport costs one busy CPU core per rank.

## Limits

- The 27B with the parallel drafts (DFlash2, DSpark) or n-gram speculation; not Flash-Next or MTP.
- No cold tier or session store: the ranks would park and restore sessions independently.
- `reload_draft` needs the draft at the same path on every rank; a rank that cannot load it ends the pair.
- Pool sizes, admission's memory floor and every scheduler step are collectives across the ranks: call the executor
  and scheduler only from the engine loop, which keeps them in lockstep.
- The one-shot pairs each RDMA device with the peer's in name order (on two Sparks, the same-named ones).
