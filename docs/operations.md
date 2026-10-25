# Operate Sharar

`sharar serve bolt` serves the packed Nemotron 3.5 Bolt text model on port 30006. Its profiles currently share
native weights, full top-9 routing, FP32 Mamba state, FP8 KV and SSD caching. The pinned companion LiLiCorr
checkpoint drafts seven tokens through the parallel DFlash proposer, with exact target verification;
`--spec 0` selects ordinary decode. See [Bolt](bolt.md) for formats, limits and validation.

## Prerequisites

One NVIDIA GB10 (SM 12.1, a DGX Spark; two for [tensor parallel](tensor-parallel.md)), `aarch64`, about 128 GB
unified memory, NVIDIA driver 580 or newer, the CUDA 13 toolkit in `/usr/local/cuda` (DGX OS ships it; FlashInfer
and Sharar compile their GB10 kernels on first use) and [uv](https://docs.astral.sh/uv/). `--detach` also needs a
user systemd session (`loginctl enable-linger` keeps it after logout).

Two targets, one per process, chosen by name: `27b` (`RadixArk/Qwen3.8-27B-NVFP4` at `319f741`, with the DFlash2
drafter `incoai/Qwen3.8-27B-DFlash2` at `dedf8df`) and `flashnext` (`RadixArk/Qwen3.8-Flash-Next-NVFP4` at
`7b71922`, its own MTP head, or the DFlash2 drafter `incoai/Qwen3.8-Flash-Next-DFlash2` at `a22a174` on one Spark).
`sharar serve` downloads whatever the cache lacks (about 26 GB for the 27B, 135 GB for
Flash-Next) after checking free disk; serving itself runs offline.

## Install and serve

```bash
uv sync                           # .venv from uv.lock: CUDA 13 PyTorch, FlashInfer, the serving packages, sharar (editable)
uv run sharar serve 27b           # or flashnext; in the foreground, Ctrl-C drains and stops
uv run sharar serve 27b --detach  # or as a user systemd service: sharar status | logs -f | stop
uv run sharar verify              # health, the pinned model, a short exact answer
```

`uv.lock` pins every package; the GB10 group of `pyproject.toml` takes torch from the cu130 index and FlashInfer's JIT
cache from FlashInfer's, constrained to the tested versions in `constraints/`. `uv run` keeps `.venv` in step with
the lock, so after `git pull` the next `uv run sharar serve` uses the new code and packages (a running server keeps
the code it started with).

`sharar serve` resolves the settings (below), checks the host (one GB10, driver, 24 GiB available above the host floor), downloads
missing snapshots, keeps the 3-bit checkpoint on its pinned revision, prints what it will serve, and runs the server
in its own process with the memory guard as a thread: when `MemAvailable` stays under half the host floor for six seconds it
stops the server (SIGTERM, then SIGKILL after 15 s), before unified memory pages the weights. The scheduler already holds
new admissions under that threshold. `--detach` runs the same command as the user systemd service `sharar-qwen38` or
`sharar-flashnext`, logging to `$SHARAR_RESULTS/logs/<service>.log` (the previous run's is `.log.1`); `sharar stop`
sends it SIGTERM, so it drains and, with the SSD tier on, stores its cached prefixes. Automatic KV capacity depends
on available memory.

## Settings

Every setting is a `SHARAR_*` name. Order, first wins ([sharar/launch/settings.py](../sharar/launch/settings.py)):
the command line (`sharar serve --help`), the environment, the config file (`SHARAR_CONFIG_FILE`, else
`config/gb10.env` in the checkout, else `~/.config/sharar/sharar.env`; [example](../config/gb10.env.example)), the
target's settings (port, service, drafter), the profile's defaults, then the built-in defaults: models in
`SHARAR_HF_CACHE` (default `$HF_HOME`, else `~/.cache/huggingface`; Flash-Next in `SHARAR_FLASHNEXT_CACHE`, default the
same) and results, traces, 3-bit checkpoints and logs in `SHARAR_RESULTS` (default
`~/.local/share/sharar`). The server reads the resolved settings once into `EngineConfig`
([engine/config.py](../sharar/engine/config.py)); `/get_server_info` reports them under `config`.

| Setting | Launcher default | Override |
|---|---|---|
| Target | the one named on `serve` (`27b` or `flashnext`) | `SHARAR_TARGET`; Flash-Next reads its model from `SHARAR_FLASHNEXT_CACHE` and takes `SHARAR_FLASHNEXT_TRUNK=fp8\|bf16` |
| Serving profile | `fast` | `sharar serve --exact\|--fast\|--throughput` or `SHARAR_PROFILE` ([below](#serving-profiles)) |
| Proposer / draft proposals | 27B: DFlash2, 15 (16 verify rows); Flash-Next: MTP, 4 (`fast`: up to 8, adaptive, below), or with `SHARAR_PROPOSER=dflash2` (one Spark) DFlash2, 9 (`exact`: 7, `throughput`: 4) | `SHARAR_PROPOSER`, `SHARAR_SPEC` (0 = plain decode; with adaptive depth, the cap) |
| Draft precision | 27B: FP8; Flash-Next: BF16 (MTP), FP8 (DFlash2) | `SHARAR_DRAFT_PRECISION=bf16` for the 27B publisher's precision |
| Draft checkpoint | the publisher's pinned draft | `sharar serve --draft DIR` or `SHARAR_DRAFT=/dir`: a DFlash2 checkpoint of the same architecture, trained on the served target's features |
| Accepted state | reconstruction | `SHARAR_VERIFY_STATE` |
| Context per request (prompt plus output) | 262,144 | `SHARAR_MAX_CONTEXT` / `sharar serve --context-length`. The pool guarantees one request at that length (8 GiB of FP8 KV at 262K; 4.5 GiB as NVFP4) or four at 32K, whichever is more; concurrent requests share the rest and a request that does not fit waits |
| Verify depth per batch size | the proposer's depth at every batch; Flash-Next `fast`: `2:4,4:4`, `throughput`: `16:3,32:3,64:3,128:3` (3-bit experts on one Spark), else `16:3`; Flash-Next DFlash2 `fast`: `2:7,4:7`, `throughput`: MTP's | `SHARAR_SPEC_DEPTHS=BUCKET:DEPTH,...` (0 = plain decode; a profile's buckets stay within `SHARAR_SPEC` and `SHARAR_MAX_BATCH`, a limit between buckets verifying what the bucket it pads to does). The 3-bit copies serve calls of at most 128 rows on the 27B and 64 on Flash-Next, so wider verifications run the original weights |
| Adaptive draft depth (MTP and DFlash2) | Flash-Next `fast` with MTP: on, 0.14 (0.2 on two Sparks); DFlash2: off | `SHARAR_SPEC_STEP_COST` (0 = the fixed depth): each request drafts as deep as its own recent acceptance pays for, up to the batch bucket's depth ([MTP](flashnext.md#the-mtp-draft), [DFlash2](flashnext.md#the-dflash2-draft)) |
| Port / service (with `--detach`) | 27B 30001 / `sharar-qwen38`, Flash-Next 30004 / `sharar-flashnext` | `sharar serve --port`, `SHARAR_PORT` / `SHARAR_SERVICE` |
| Request traces | off | `sharar serve --traces` (in `$SHARAR_RESULTS/traces`) or `SHARAR_TRACES=/dir` (cap `SHARAR_TRACES_MAX_MIB`, 32768) |
| Calibrated 3-bit checkpoint | `$SHARAR_RESULTS/w3-checkpoints/<target>-current`, kept on the pinned published revision at every serve ([how](w3-calibration.md#where-the-served-copies-come-from)) | `SHARAR_W3_CHECKPOINT=/dir`. Without one the 27B rounds to nearest and Flash-Next serves its FP8 trunk; the preflight and `/get_server_info` (`w3_warning`) say so ([calibration](w3-calibration.md)) |
| Flash-Next routed experts | the W3 checkpoint's 3-bit experts when it carries them (`fast`, `throughput`) | `SHARAR_FLASHNEXT_EXPERTS=nvfp4` keeps the checkpoint's NVFP4 experts |
| Flash-Next PLE table on two Sparks | each rank's half of the heads in device memory (23.8 GiB per rank; [tensor parallel](tensor-parallel.md)) | `SHARAR_PLE_RESIDENT=0` reads it from the memory maps on the host, as one Spark does (which prefetches a prompt's later chunks into the page cache) |
| Shape-stable prefill numerics | off | `SHARAR_STABLE_NUMERICS=1` (M-invariant kernels for the three row-count-dependent ops; see [limits](#numerical-limits)) |

For MTP on the 27B, use `SHARAR_PROPOSER=mtp SHARAR_SPEC=3 SHARAR_DRAFT_PRECISION=bf16`. `SHARAR_PROPOSER` is `mtp`
or `dflash2`.

### NVFP4 KV

`SHARAR_KV_FORMAT=nvfp4` (`fast` and `throughput` on one Spark, both targets) stores K and V as NVFP4: two E2M1 codes
a byte and an E4M3 scale per 16 values, 4.5 bits a value where FP8 takes 8. The same memory holds 16/9 the 27B's tokens
and about 1.4x Flash-Next's (its BF16 index keys keep their size): longer contexts or more requests at once, more
cached prefixes, and fewer bytes read back from the SSD tier.
FlashInfer writes them (`nvfp4_quantize_append_paged_kv_cache_with_slot_mapping`, global scale 1) and its `fa2`
attention reads them; the scales are per-page rows (`KvPool.page_aux`), so the prefix cache and the SSD tier carry them
with each page, and an SSD store keeps its KV format in its identity.

Verification of more than 8 rows a request (the 27B's graphs) runs a GB10 kernel instead,
[kernels/csrc/verify_attn.cu](../sharar/kernels/csrc/verify_attn.cu): one block a request, KV head and chunk of the
context, the codes converted to FP16 once a tile in shared memory, both products on the FP16 tensor cores, the chunks
merged by their log-sum-exps. At 16 rows it takes 1.3-1.55x less time than `fa2` over the same cache (1.9x with eight
requests), at `fa2`'s precision or better against exact attention; up to 8 rows `fa2` is faster and serves. Two warps
of each block load and convert the tiles while six compute. A 27B prefill against a cached prefix of at least 1,024
tokens runs the same tiles without the causal mask, 128 (token, head) rows a block, and merges the chunk's own causal
attention (FlashInfer's ragged prefill) into its output: 1.6-1.7x less time than `fa2` and `merge_state` over 8-20K
prefixes, so a 32K prompt prefills in 11.8 s instead of 14.0. `SHARAR_VERIFY_ATTENTION=fa2` turns both off.

- **Speed.** Steady-state rounds of the 27B `fast` against `fa2` over NVFP4: 64K 77.6 against 83.0 ms with one request,
  137.7 against 155.6 ms with four (FP8 KV's 146.0); within 2% at 10K.
- **Accuracy.** Teacher-forced 256 tokens after 32K, 64K and 96K-token prompts against BF16 KV: mean KL 0.0063 where
  FP8 KV's is 0.0039, flat along the generation; argmax agreement 95.7-99.6% (FP8 96.1-99.6%). The lossy gate's
  in-process layers on its short prompts: KL 0.0055 against a floor of 0.0047, tokens a round unchanged.
- **Flash-Next.** Its sparse attention (`kernels/qsa/attention.py`) reads the codes with the hardware's E2M1
  conversion (two codes an instruction), one multiply by the scale and FP16 operands: on a 64K prompt's own page
  selections at or under FP8's time from one row to an 8K prefill chunk. Teacher-forced 256 tokens after 32K, 64K and
  96K-token prompts against BF16 KV: mean KL 0.0084 where FP8's is 0.0059, flat along the generation; argmax agreement
  93.8-98.4% (FP8 96.1-99.6%).
  Steady-state rounds of `fast` within 1% of FP8 or faster (10K and 64K, one and four requests), the first token of a
  64K prompt 18.9 s with either. The lossy gate's in-process layers against FP8 on its 28 prompts: KL 0.0054 against
  a floor of 0.0043, argmax agreement 97.5% (the floor's), tokens a round x0.998 [0.968, 1.027].
- **Screen.** `sim-fp4` (and `sim-k8v4`, `sim-bf16`) keep a BF16 pool and round every write through NVFP4 the way
  FlashInfer's writer does (`kernels/kv_fp4.nvfp4_round`, bitwise its arithmetic), to measure a format without its
  kernels; either target, without the SSD tier.

## Serving profiles

A profile sets the default of every speed/fidelity knob (`profile_defaults` in
[sharar/launch/settings.py](../sharar/launch/settings.py)); a knob set on the command line, in the environment or in
the config file wins. `/get_server_info` reports the profile under `config`.

| Knob | exact | fast (default) | throughput |
|---|---|---|---|
| Relaxed acceptance, greedy requests (`SHARAR_ACCEPT_DEFAULT`) | off | top-3 within 2.0 logits; Flash-Next DFlash2 top-2 within 1.0 | as fast |
| Relaxed acceptance, sampled requests (`SHARAR_ACCEPT_DEFAULT_SAMPLED`, `SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD`) | off | Flash-Next top-3 within 2.0 (DFlash2 top-2 within 1.0); the 27B top-2 within 1.0 (at 3:2.0 its sampled thinking looped in Claude Code sessions) | as fast |
| Calibrated 3-bit decode copies (`SHARAR_W3_DECODE`; on Flash-Next and Nemotron also the routed experts) | off | on | on |
| Device-resident acceptance (`SHARAR_DEVICE_ACCEPT`) | off | Flash-Next and Nemotron | as fast |
| NVFP4 KV cache (`SHARAR_KV_FORMAT`, [below](#nvfp4-kv)) | FP8 | one Spark: NVFP4 (both targets); two Sparks FP8 | as fast |
| Budgeted expert selection (`SHARAR_MOE_KEEP`, `SHARAR_MOE_BUDGET`) | off | Flash-Next: keep 4, budget 12:64; Nemotron: keep 12, budget 48:128 | Flash-Next as fast; Nemotron off |
| Optimized QSA prefill kernels (`SHARAR_QSA_PREFILL_BACKEND`) | off (original) | Flash-Next: auto (where eligible: SM121, one or two Sparks, through 32K) | as fast |
| Requests decoded together (`SHARAR_MAX_BATCH`) | 4 | 4 | 27B 16; Flash-Next 128 (64 with NVFP4 experts) |
| Reduced recurrent state (`SHARAR_SSM_STATE`) | off (FP32) | Nemotron: FP16 with stochastic rounding; else FP32 | 8-bit GDN state; Nemotron FP32 |

Relaxed acceptance accepts a drafted token the target ranks within its top-k and within a logit margin of its
argmax, so outputs are no longer the target's exact greedy continuation; a request can pin exact acceptance with
`accept_top_k: 1` ([API](api.md)). The 3-bit copies (3.25 bits per weight, tensor-core kernels for calls of at
most 128 rows on the 27B, 64 on Flash-Next) cover the 27B's MLPs, FP8 projections and FP8 draft (+9.8 GiB resident) and Flash-Next's trunk and FP8 DFlash2 draft; Flash-Next's
3-bit routed experts replace the NVFP4 ones at every row count (17 GiB less resident). The 27B's lm_head keeps its
original format; Flash-Next's has a 3-bit copy, read by calls of at most 64 rows (decode, verify and DFlash2's draft
projection). Speculative decoding itself, the Flash-Next 2-bit draft head, FP8/NVFP4
weights and CUDA graphs are lossless and on in every profile. Benchmarks and gates use `exact`.

Four of the knobs take more than the table shows:

- **`SHARAR_MAX_BATCH`** (`sharar serve --max-batch`, 1-128; graph buckets 1, 2, 4, ... 128). Above four, the pool
  guarantees the batch-four total (4 x 32K tokens) split between the running requests, or one request at the full
  context.
- **`SHARAR_MOE_KEEP=k`, `SHARAR_MOE_BUDGET=[P:]B`** (Flash-Next): a decode or verify forward reads at most B distinct
  experts, or min(B, max(16, P x its requests)): every token's guaranteed top experts, then those with the most router
  probability summed over the forward's tokens. Each token takes up to 10 of them, its guaranteed ones first. The
  guarantee shrinks as the forward grows, 2 x cap / tokens within [2, k], so the cap keeps binding: at 4 and 12:64, up
  to 4 tokens a request keep 4 (one request alone, up to 8), a depth-8 draft of one request or eight requests of 5
  tokens 3, 64 tokens and more 2. The floor of 16 is for a request alone: at 12 its GSM8K fell 4.7 points. Flash-Next's
  DFlash2 depths keep 3 in `fast` ([how](flashnext.md#the-dflash2-draft)).
  `SHARAR_MOE_KEEP` alone gives each token its top k;
  `SHARAR_MOE_KEEP=0` routes in full. A token's experts depend on the rest of its forward, so a greedy reply can change
  with what else the server runs. Nemotron's selection (22 of 512 experts a token) keeps k as each row's floor, lets a
  row's other top-22 experts ride along when the forward reads them anyway (`SHARAR_MOE_FREE_RIDERS=0`: off), and fills
  the cap by summed routing weight ([how](nemotron.md#serving)).
- **`SHARAR_SSM_STATE=fp32|int8|int4`**: the integer formats quantize each GDN layer's recurrent state per window as
  in LeapQuant ([arXiv 2609.38166](https://arxiv.org/abs/2609.38166); `sharar/cache/ssm_format.py`), a fraction of
  FP32's memory per slot, and need `SHARAR_VERIFY_STATE=reconstruct` (with `snapshots` the state stays FP32). The
  saving pays with many requests (on one Spark the 27B decodes 30% faster at 16) but not at one, where it also
  delays a long prompt's first token, so `fast` keeps FP32. Nemotron's Mamba-2 state takes `fp16` (stochastic
  rounding keyed on token position and layer) or `fp16-rn` (nearest) instead: half of FP32's state traffic, and in
  `fast` with expert sharing.
- **`SHARAR_QSA_PREFILL_BACKEND=original|auto|optimized`** (Flash-Next): `auto` runs the optimized SM121 sparse
  eager prefill through 32K where eligible and the original elsewhere; `optimized` requires it. Its score
  arithmetic and tie order differ, so `exact` keeps `original`; it makes 8K and 32K prefill 3-4% faster with
  RULER v2 at 32K unchanged. `SHARAR_STABLE_NUMERICS=1` makes `auto` use the original.

A new lossy option enters `fast` (and so `throughput`) only through the lossy-change gate
([CONTRIBUTING](../CONTRIBUTING.md#judge-a-lossy-option)).

## Request traces

Traces are off unless asked for (`sharar serve --traces`, or `SHARAR_TRACES` naming a directory). Each finished
request is then appended as one JSON line: prompt and output token IDs, sampling parameters, the cached prefix length
and, for speculative requests, every verification round with the accepted count, the draft's proposals, the target's
argmax rows (sampled rows keep only the correction token) and the offsets accepted under the relaxed rule (`rx`).
They measure speculation on real traffic, such as acceptance by draft depth. No text or client metadata is stored.
Files rotate at 64 MiB, and the oldest file is deleted above the cap (32 GiB by default). Writing happens on a
separate thread and never blocks the engine; a full queue or disk counts as `dropped` in `/get_server_info` under
`traces`. Any request can opt out with `"trace": false`; the benchmark and accuracy clients do, so the directory
holds the user's own traffic. `sharar traces --dir DIR` summarizes a directory. The traces are the user's prompts and
stay on this machine unless the user copies them elsewhere.

**Draft confidence (MTP).** With `SHARAR_TRACE_CONFIDENCE=1` beside `SHARAR_TRACES`, each round record also carries
the MTP draft's top-1 probability `p1` and top-1/top-2 margin `m` per drafted position, the calibration data for
adaptive draft depth. It is read when the proposer is built; off, nothing changes.

Inspect `sharar status`, `sharar logs` (detached servers), `/health` and `/get_server_info` for status.
The latter reports draft precision, scheduler settings, KV/state geometry and
cache statistics: locks by owner, snapshots (spare and not), evictions by reason, pruned dead-KV nodes, full
re-prefills, resumes from a branching point, the evictable totals, and the drafters' resume hits and fallbacks
(`cache.draft_resumes`; [cache design](cache_design.md#protection-and-eviction)); `cache.draft_features` reports
retained DFlash2 features.
At 96 slots, the 27B's DFlash2 feature retention reserves up to 1,900 MiB before planning
KV (Flash-Next's, half as wide, half that). The pools take what is available at startup beyond the host floor and the
allocations counted after them ([memory_plan.md](memory_plan.md)). No host-memory tier sits below them: on unified
memory it would hold the same prefixes in the same DRAM.

**SSD tier.** On in every profile (`SHARAR_SSD=0` turns it off), it is the only tier below the GPU pools: an
evicted prefix of 4,096 tokens or more goes to `$SHARAR_RESULTS/ssd/<target>` (`SHARAR_SSD_DIR` moves it, e.g. to
another drive; capacity `SHARAR_SSD_GIB`, default 200, never filling the drive) instead of being dropped,
and its next request reads it back instead of prefilling again. At shutdown the server writes every cached prefix
there, so stored prefixes survive restarts and reboots. `cache.ssd` in `/get_server_info` reports usage and
restores; `sharar cache` prints the GPU cache and the SSD tier, or with nothing serving, what the store holds.
Measure it on a real NVMe: a filesystem on a loop device without direct I/O caps reads near 1 GB/s and serves
fresh data from the host page cache. How it works: [cache design](cache_design.md).

While host `MemAvailable` is under half the host floor the scheduler holds new
admissions (`floor_holds` in `/get_server_info`) so running requests can
finish and release memory before the memory guard has to act.

**Host floor.** The MemAvailable Sharar leaves for the OS and other programs (a client, an agent, the page cache):
6 GiB, `SHARAR_HOST_FLOOR_GIB` to change it. The pools get everything else, once Sharar's own allocations after them
are counted: the drafter, the SSD tier's staging, FlashInfer's workspaces, the CUDA graphs, the GPU memory the first
prefill takes outside PyTorch, and the activation blocks PyTorch keeps cached (the server log's "counted after the
pools" line; `/get_server_info`'s `after_plan_gib`). Admission holds, and the memory guard stops the server, only under
half the floor, when other programs have used more than it left them (`SHARAR_HOST_FLOOR_GIB` moves both; main's
`SHARAR_MIN_AVAILABLE_KIB` is retired). Preflight wants the 24 GiB load peak above the floor. `/get_server_info` also reports `memory_gib`:
PyTorch's allocated and reserved bytes, now and at their peak, and the process's resident memory. Terms and measurements: [memory_plan.md](memory_plan.md).

## Development

`uv run sharar ...` runs this checkout's code (the editable install); the checks are in
[CONTRIBUTING](../CONTRIBUTING.md#run-checks). Reserve an exclusive GPU window (`flock $SHARAR_RESULTS/.gpu.lock ...`)
before loading a model; `uv run sharar dev --help` lists the in-process diagnostics.

## Numerical limits

The target's arithmetic depends on execution shape even with unchanged weights: prefill features of a fixed
prefix change with the total prefill length and chunking, and greedy continuations can change with batch shape,
verification width or the drafter, because those move the round boundaries a token is computed in. Each
configuration is deterministic run to run. The cause is three kernels whose arithmetic depends on the row count
(cuBLAS's BF16 GEMM for the GDN b/a gates, cuBLASLt's FP8 GEMM for the 5120x6144 output projections at 27 to 32
rows, the gated norm's rows-per-block choice), with FlashInfer's prefill schedule a rare fourth.
`SHARAR_STABLE_NUMERICS=1` swaps in M-invariant replacements for the first three (a prefix is then bit-exact
across prefill lengths, for under 1% prefill cost); chunked and cached-prefix prefills still take different
attention paths. This is why changes are judged on paired quality and speed, not on identical outputs
([evaluation](evaluation.md)).
