# Tensor parallel: the 27B and Flash-Next on two Sparks

```bash
./sharar-ctl serve 27b --tp 2       # rank 0 here (http://127.0.0.1:30001), rank 1 on the peer; --exact, --balanced, --throughput
./sharar-ctl serve flashnext --tp 2 # Flash-Next the same way (MTP draft, its calibrated 3-bit trunk)
./sharar-ctl status                 # stop, logs [-f], chat and launch claude work as on one Spark
./sharar-tp comm                    # the RDMA one-shot all-reduce against NCCL between the two Sparks
./sharar-tp draft-head              # the vocabulary-parallel draft head against the whole-vocabulary one
```

`sharar-ctl` hands two Sparks to `./sharar-tp`, as it hands one to `run-sharar.sh`. The peer is the host at the other
end of the RoCE link (or `SHARAR_TP_PEER`). The first serve sets both Sparks up (`./sharar-tp setup`: the uv venv on
each; the model, draft and 3-bit checkpoint here, copied to the peer); every serve mirrors the checkout to the same
path on the peer (`rsync --delete`, `.git` included: it replaces whatever is at that path there) and starts
`sharar serve --tp 2` on both with sharar-ctl's serving profiles (`fast` by default; `SHARAR_W3_DECODE=0` serves the
native NVFP4/FP8 weights). Rank 1 serves with rank 0's configuration: when the ranks meet, rank 0 broadcasts the
`EngineConfig` it read from its environment, so the paths in it (draft, 3-bit checkpoint) must exist on both Sparks.
`stop` signals rank 0 alone; it drains its requests, and the stop command it broadcasts ends rank 1; both persist
their cached sessions first (below). In LXD/Incus containers
whose CPU set has holes, `tools/lxcfs-run.sh` (used by the launcher) keeps `import torch` from corrupting its heap.
The target's settings are the single-Spark launcher's (`config/targets/<target>.env` through
`tools/launcher-config.sh`; the environment wins), passed to rank 0 and broadcast from there.

Both Sparks need the same user and paths (the checkout, `~/.cache/huggingface`, `$SHARAR_RESULTS`) and the
interpreter's C headers (`sudo apt install python3.12-dev`): the RDMA all-reduce is a torch extension each rank
builds on first use, with the venv's ninja. `./sharar-tp setup` checks both. A rank killed while it builds a torch
extension leaves `~/.cache/torch_extensions/<python>_<cuda>/<extension>/lock` behind; the next start of that rank
waits on it forever while the other rank's collectives time out, so remove a stale `lock` (no build running).

**A failed rank.** Every rank above 0 holds a TCP connection to the primary (keepalive about 5 s for a host that
vanishes). When a rank ends, however it ends, the others end within a second instead of waiting in their next
collective for NCCL's 10-minute timeout, and `./sharar-tp` restarts the pair (up to 5 times in 10 minutes; then it
stops and says so). Measured on the 27B: rank 1 or rank 0 SIGKILLed, serving again 26 s later.

**The cold tier and the session store** run on every rank, each parking and storing its own heads of the same
sessions. Every decision is the same on every rank: the arena budget and each new slab take the minimum over ranks
of what the hosts allow, a stored record is found only when every rank's background writer has landed it and
restored only when every rank read it, and a record names its rank. The store is
`$SHARAR_RESULTS/sessions-tp` on each Spark (`SHARAR_SESSION_DIR`; `off` disables). Measured on the 27B: a
21K-token session persisted at a stop answered its follow-up after a restart in 0.19 s (57 ms disk load per rank)
against 6.5 s cold.

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
compile again on each rank; sampled requests get one seed for all ranks. An engine failure ends the process, and
the other ranks follow within a second (see "A failed rank" above).

## What it buys

Measured on two GB10s linked by RoCE, `fast` profile (27B 2026-10-07, Flash-Next 2026-10-09):

| | 27B, one Spark | 27B, two | Flash-Next, one Spark | Flash-Next, two |
|---|---:|---:|---:|---:|
| Single-stream greedy decode, median of seven prompts | 77.1 tok/s | 119.1 | 73.5 | 99.0 |
| Four concurrent requests, aggregate | 135.1 tok/s | 199.1 | 147.6 | 198.7 |

Cold prefill is x1.4 to x1.7 faster on two Sparks (28K tokens: time to first token 16.1 to 9.5 s on the 27B).
Flash-Next holds 36.3 GiB of weights per rank against 70.3 on one Spark. GSM8K is unchanged (27B 95.3% on one, 95.0% on two). TP=2 is not bitwise TP=1: summing two partial products in
another order moves the logits as much as any other reassociation (88-95% teacher-forced argmax agreement, flips
at near ties), and TP=1 is unchanged by the tensor-parallel code. Flash-Next's verification scales less than its
bytes: the fused MoE has a fixed cost per call, and the hyper-connection mixers, router and elementwise kernels
are replicated.

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
together (about 22 GB/s for a large all-reduce). DGX Spark has no GPUDirect RDMA, so NCCL stages through host
buffers and every collective pays a fixed 20-45 us however small it is; a decode or verification all-reduce moves
only 160 KiB, so that fixed cost is its whole time.

Two ranks need one exchange, not a ring's two steps. `kernels/rdma_oneshot.py` registers cudaHostAlloc buffers
with both RDMA devices; a kernel copies the partial in and rings a doorbell, a busy-polling CPU thread posts the
RDMA writes (half per device, each followed by a flag), and the peer's kernel waits on the flags and adds. Only
one GPU thread polls the flags (with every block polling, the NIC's flag writes contended with them). Inside a
CUDA graph a 160 KiB all-reduce takes 19 us against NCCL's 51. The one-shot takes messages up to 4 MiB
(verification of up to 25 requests, prefill chunks of up to 409 tokens); larger prefill chunks stay on NCCL's
ring, where bandwidth sets the time. The transport costs one busy CPU core per rank. `./sharar-tp comm` measures
both against each other.

## Flash-Next

The same roles on Flash-Next's layers (`model/flashnext/`): attention q|k|v by head (12 of 24, 1 of 2 KV heads),
GDN by head (8 of 16 key, 24 of 48 value heads), o_proj and out_proj row-parallel, and every expert's
intermediate split, routed and shared alike (320 of 640 columns: gate|up column-parallel, down row-parallel),
so each rank streams half of every selected expert and the decoder layer all-reduces one partial sum per block.
FlashInfer's fused NVFP4 MoE runs the 320-column experts as they are (two halves sum to the whole within BF16
rounding; one MoE call of one row takes 97 us against 145 us, sixteen rows 944 against 1,645 us). FP8 trunk parts
are quantized with the whole matrix's scale, so a rank's part is exactly its rows or columns of the single-GPU
copy, and the calibrated 3-bit copies load the same parts; the shared expert's down projection (320 input columns
per rank) cannot be held in the 3-bit format's 128-column chunks and stays FP8. The hyper-connection mixers, the
router, the indexer and the PLE read the whole (replicated) stream and stay whole on every rank; the lm_head is
vocabulary-parallel.

The MTP draft layer is replicated and drafts through the target's vocabulary-parallel lm_head
(`proposers/draft_head.py:VocabParallelHead`): each rank scores its half of the vocabulary with its 3-bit copy,
the ranks exchange their best eight, and the rank holding each candidate's FP8 row rescores it, so the pick is the
whole-vocabulary shortlist head's (`./sharar-tp draft-head`: identical picks at 1 to 20 rows, graphed too) at
0.72 ms per draft step instead of 1.17.

## Limits

- Flash-Next keeps its NVFP4 routed experts on two Sparks (the 3-bit experts serve one Spark only).
- The 27B with the parallel drafts (DFlash2, DSpark) or n-gram speculation, not MTP; Flash-Next with MTP. A reduced
  draft vocabulary (`SHARAR_DRAFT_VOCAB`) is not served tensor-parallel.
- `reload_draft` needs the draft at the same path on every rank; a rank that cannot load it ends the pair.
- Pool sizes, admission's memory floor and every scheduler step are collectives across the ranks: call the executor
  and scheduler only from the engine loop, which keeps them in lockstep.
- The one-shot pairs each RDMA device with the peer's in name order (on two Sparks, the same-named ones).
