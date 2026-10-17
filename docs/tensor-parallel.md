# Tensor parallel: the 27B and Flash-Next on two Sparks

```bash
uv run sharar serve 27b --tp 2         # rank 0 here (http://127.0.0.1:30001), rank 1 on the peer; --exact, --throughput
uv run sharar serve flashnext --tp 2   # Flash-Next the same way (http://127.0.0.1:30004; MTP draft, its calibrated 3-bit trunk)
uv run sharar status                   # chat, launch and verify work as on one Spark; --detach and stop too
uv run sharar dev tp-comm              # the RDMA one-shot all-reduce against NCCL between the two Sparks
uv run sharar dev tp-draft-head        # the vocabulary-parallel draft head against the whole-vocabulary one
```

`--tp 2` is the single-Spark command with the peer added (`sharar/launch/tp.py`): the same settings, preflight,
downloads and 3-bit checkpoint, then rank 0 here and rank 1 on the peer, the host at the other end of the RoCE link
that answers ssh (or `SHARAR_TP_PEER`). Before each serve it mirrors the checkout to the same path on the peer
(`rsync --delete`, `.git` included: it replaces whatever is at that path there), syncs the peer's Python environment
(`uv sync --frozen`; uv is copied there if missing), and copies the model, draft and 3-bit checkpoint (rsync copies
only what changed). Both ranks run `sharar engine ... --tp 2 --tp-rank R`; when they meet, rank 0 broadcasts the
`EngineConfig` it resolved, so rank 1 serves with rank 0's settings and the paths in them must exist on both Sparks.
SIGTERM or Ctrl-C reaches rank 0 alone: it drains its requests, and the stop it broadcasts ends rank 1; with the SSD
tier on, both store their cached prefixes first (below). In LXD/Incus containers whose CPU set has holes, each rank restarts
itself in a namespace whose CPU lists agree with `/proc/cpuinfo` (`sharar/launch/lxcfs.py`), which keeps
`import torch` from corrupting its heap.

Both Sparks need the same user and paths (the checkout, the model cache, `$SHARAR_RESULTS`). The RDMA all-reduce is a
torch extension each rank builds on first use, against the system's ibverbs headers or a copy the launcher unpacks
into the environment (`SHARAR_TP_ALLREDUCE=nccl` needs neither). A rank killed while it builds a torch extension leaves
`~/.cache/torch_extensions/<python>_<cuda>/<extension>/lock` behind; the next start of that rank waits on it forever
while the other rank's collectives time out, so remove a stale `lock` (no build running).

**A failed rank.** Every rank above 0 holds a TCP connection to the primary (keepalive about 5 s for a host that
vanishes). When a rank ends, however it ends, the others end within a second instead of waiting in their next
collective for NCCL's 10-minute timeout, and `sharar serve` restarts the pair (up to 5 times in 10 minutes; then it
stops and says so). Measured on the 27B: rank 1 or rank 0 SIGKILLed, serving again 26 s later.

**Cached prefixes.** The SSD tier ([operations](operations.md)) runs on both ranks: each stores its
own heads of every stored prefix under `$SHARAR_RESULTS/ssd/<target>` on its own Spark (`SHARAR_SSD_GIB` is per Spark). Both ranks
decide alike, as their schedulers do: a write counts once it completed on both ranks (failed if it failed on
either), a restore goes ahead only if both read every object intact, and a restart lists only the prefixes both
stored. The tier runs on both ranks or on neither (a store that cannot open turns it off on both).

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
  --delay=420 --duration=20 -o /path/tp2_%h" uv run sharar serve 27b --tp 2   # then keep requests flowing
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
ring, where bandwidth sets the time. The transport costs one busy CPU core per rank. `sharar dev tp-comm` measures
both against each other.

## Flash-Next

The same roles on Flash-Next's layers (`model/flashnext/`): attention q|k|v by head (12 of 24, 1 of 2 KV heads),
GDN by head (8 of 16 key, 24 of 48 value heads), o_proj and out_proj row-parallel, and every expert's
intermediate split, routed and shared alike (320 of 640 columns: gate|up column-parallel, down row-parallel),
so each rank streams half of every selected expert and the decoder layer all-reduces one partial sum per block.
The 3-bit routed experts (#73, the default when the W3 checkpoint carries them) are split by expert instead: rank r
holds experts 256r to 256r + 255 whole (`sharding.ExpertParallel`), since a down projection's 320 columns per rank
would not fill the 3-bit format's 128-wide chunks. `kernels/w3_moe.py` skips the pairs routed to the other rank's
experts, so each rank returns the partial sum over its own, all-reduced as before; a rank holds 23.0 GiB of experts
(27.6 GiB in all, against 36.3 with the split NVFP4 experts). A token's ten experts land unevenly (five each on
average), which costs less than the bytes saved: on 2026-10-11 single-stream decode went from 93.4 to 102.8 tok/s and
four requests from 192.9 to 237.7, with GSM8K (first 300) and HumanEval as on one Spark with the 3-bit experts
(95.7% and 93.9% both; the split NVFP4 experts 94.0% and 93.9%).
FlashInfer's fused NVFP4 MoE runs the 320-column experts as they are (two halves sum to the whole within BF16
rounding; one MoE call of one row takes 97 us against 145 us, sixteen rows 944 against 1,645 us). FP8 trunk parts
are quantized with the whole matrix's scale, so a rank's part is exactly its rows or columns of the single-GPU
copy, and the calibrated 3-bit copies load the same parts; the shared expert's down projection (320 input columns
per rank) cannot be held in the 3-bit format's 128-column chunks and stays FP8. The hyper-connection mixers, the
router, the indexer and the PLE read the whole (replicated) stream and stay whole on every rank; the lm_head is
vocabulary-parallel.

The PLE table (47.7 GiB: 16 heads of 20M rows of 160 bytes) is split by head instead (#178,
`model/flashnext/ple.py:ResidentRows`). Each head's rows are one contiguous region, so rank r keeps heads 8r to 8r + 7
in device memory, 23.8 GiB read from the checkpoint at load in about 15 s. A forward's rows are gathered on the device
and all-gathered (1,280 bytes per token from each rank), so every rank holds the bytes one Spark reads from its memory
maps, and the outputs are the same. Nothing waits on the NVMe: one Spark decodes text it reads for the first time 11-14%
slower than text whose rows are cached, and two Sparks now decode both at the cached speed. Measured 2026-10-14 against
the host gather (#169), page cache dropped before each start, outputs identical:

| | host gather | resident heads |
|---|---:|---:|
| `fast`, single-stream decode, median of seven prompts | 107.8 tok/s | 123.8 |
| `fast`, four concurrent requests | 234.3 tok/s | 272.0 |
| `fast`, time to first token, three 10K-token prompts | 3.86 / 3.75 / 3.68 s | 3.83 / 3.20 / 3.21 |
| `throughput`, decode at 16 / 32 / 64 requests | 481 / 658 / 954 tok/s | 591 / 874 / 1,203 |

- **Memory.** The KV pool sizes around the resident half: `throughput` keeps 590K tokens per rank against 982K, and
  257 state slots. MemAvailable stayed at least 33 GiB under 64 requests, against the 16 GiB floor.
- **Turning it off.** `SHARAR_PLE_RESIDENT=0` keeps the host gather on two Sparks.

The MTP draft layer is replicated and drafts through the target's vocabulary-parallel lm_head
(`proposers/draft_head.py:VocabParallelHead`): each rank scores its half of the vocabulary with its 3-bit copy,
the ranks exchange their best eight, and the rank holding each candidate's FP8 row rescores it, so the pick is the
whole-vocabulary shortlist head's (`sharar dev tp-draft-head`: identical picks at 1 to 20 rows, graphed too) at
0.72 ms per draft step instead of 1.17.

## Limits

- The 27B with DFlash2, not MTP; Flash-Next with MTP. A reduced
  draft vocabulary (`SHARAR_DRAFT_VOCAB`) is not served tensor-parallel.
- A `--draft DIR` checkpoint is copied to the peer at the same path before the pair starts.
- Pool sizes, admission's memory floor and every scheduler step are collectives across the ranks: call the executor
  and scheduler only from the engine loop, which keeps them in lockstep.
- The one-shot pairs each RDMA device with the peer's in name order (on two Sparks, the same-named ones).
