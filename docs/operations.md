# Build and operate Sharar

## Profile and prerequisites

Use one NVIDIA GB10 (SM 12.1), `aarch64`, about 128 GB unified memory, NVIDIA
driver 580 or newer, Docker and a working user systemd manager for the guard.
The pinned target is `RadixArk/Qwen3.8-27B-NVFP4` at
`319f741cce68d7914884900c138a1fbb70a42f30`. Model, tokenizer and chat-template
files must already be cached locally; serving runs offline.

Copy [the environment example](../config/gb10.env.example) to the ignored
`config/gb10.env` and set `SHARAR_HF_CACHE` and `SHARAR_RESULTS`.
The DFlash2 snapshot is `incoai/Qwen3.8-27B-DFlash2` at
`dedf8df68adfb1afeaf7b7480c0a0243108177b4`.

## Build and launch

```bash
docker/build.sh
./run-sharar.sh --check
./run-sharar.sh
tools/verify-sharar.sh
```

The launcher binds `127.0.0.1:30001`, uses container `sharar-qwen38`, and starts
`sharar-memory-guard.service`. The guard protects a 16 GiB `MemAvailable` floor.
Startup needs at least 40 GiB available and checks the prospective pool/draft
budget separately. Automatic KV capacity depends on available memory.

| Setting | Launcher default | Override |
|---|---|---|
| Target | `qwen38` | `SHARAR_TARGET=flashnext` / `sharar serve --target flashnext` serves Qwen3.8-Flash-Next; `config/flashnext.env` is a complete launcher configuration for it (`SHARAR_CONFIG_FILE=config/flashnext.env ./run-sharar.sh`: MTP k=3, port 30004). It needs `SHARAR_FLASHNEXT_CACHE` (the RadixArk NVFP4 model cache), `SHARAR_DRAFT=pinned` and `SHARAR_DRAFT_PRECISION=bf16` (no parallel draft), and takes `SHARAR_FLASHNEXT_TRUNK=fp8|bf16` |
| Proposer | DFlash2 | `SHARAR_PROPOSER` |
| Draft proposals | 15 (16 target verification rows) | `SHARAR_SPEC` |
| Draft precision | FP8 (the measured serving profile; the 3-bit draft copy applies to it) | `SHARAR_DRAFT_PRECISION=bf16` for the publisher's precision |
| Accepted state | Reconstruction | `SHARAR_VERIFY_STATE` |
| Context per request (prompt plus output) | 262,144 (the model's native length) | `SHARAR_MAX_CONTEXT` / `sharar serve --context-length` (the engine's own default is 32,768). The pool guarantees one request at that length (8 GiB of FP8 KV at 262K) or four at 32K, whichever is more; concurrent requests share the rest and a request that does not fit waits |
| Requests decoded together | 4 (16 in the throughput profile) | `SHARAR_MAX_BATCH` / `sharar serve --max-batch`, 1-16. Above four, the pool guarantees the batch-four total (4 x 32K tokens) split between the running requests, or one request at the full context: at 262K, Flash-Next holds 16 requests of about 17K tokens each, the 27B 16 of 32K; a request that does not fit waits. `SHARAR_PROFILE=throughput` (`sharar-ctl serve --throughput`) sets 16 for evaluations and several agents at once ([measurements](optimizations.md#throughput-profile-59-2026-09-28)) |
| Verify depth per batch size | The proposer's depth at every batch; the throughput profile: Flash-Next `16:3` | `SHARAR_SPEC_DEPTHS=BUCKET:DEPTH,...` / `sharar serve --spec-depths` (buckets 1, 2, 4, 8, 16; 0 = plain decode). The 3-bit copies serve at most 64 rows: Flash-Next's `16:3` keeps a full batch on the fast profile's weights; the 27B's full depth (GSM8K x1.74 against batch four) verifies batches of five or more on the original NVFP4/FP8 weights, and `8:7,16:3` keeps fast's weights (x1.29) for paired evaluations of 27B weight candidates |
| Image | `ghcr.io/abdelfattah-lab/sharar:latest` ([tools/image.sh](../tools/image.sh)), pulled when missing; the nightly GB10 check publishes it from main (`main-<commit>` and `latest`) | `SHARAR_IMAGE=sharar:<tag>` serves a local `docker/build.sh` build |
| Container / port | `sharar-qwen38` / 30001 | `SHARAR_CONTAINER` / `SHARAR_PORT` |
| Request traces | Off | `SHARAR_TRACES=/dir` (cap `SHARAR_TRACES_MAX_MIB`, 32768) |
| Calibrated 3-bit checkpoint | `auto`: `$SHARAR_RESULTS/w3-checkpoints/<target>-current`, fetched on first serve for a target with a published checkpoint (Flash-Next) | `SHARAR_W3_CHECKPOINT=/dir` serves another completed, checksummed calibration. Without one the 27B rounds to nearest and Flash-Next serves its FP8 trunk; the preflight and `/get_server_info` (`w3_warning`) say so. See [calibration](w3-calibration.md). |
| Relaxed acceptance default | top-2 within 1.0 logit | `SHARAR_ACCEPT_DEFAULT=K:M` or `off` (greedy requests that do not set `accept_top_k`/`accept_margin`; the [study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/relaxed-acceptance.md)) |
| Three-bit decode weights | Set by the serving profile: on in `balanced` (27B), `fast` and `throughput` | `SHARAR_W3_DECODE` (1 or 0) overrides the profile. 3.25-bit copies serve every decode call of at most 64 rows through a tensor-core kernel at the memory ceiling: the 27B's MLPs, FP8 projections and FP8 draft (+9.8 GiB resident), Flash-Next's trunk. The lm_head keeps NVFP4 (a 3-bit copy cost about two GSM8K points). The 27B's earlier profile gave decode x1.49 at batch one ([study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-cuda.md)); calibrated copies improve quality and speed over nearest rounding ([calibration results](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-calibration-adaptation.md)). `SHARAR_PROFILE=exact` (or `SHARAR_W3_DECODE=0`) serves the original NVFP4/FP8 weights |
| Device-resident acceptance | On for Flash-Next in the `fast` profile, otherwise off | `SHARAR_DEVICE_ACCEPT=1` runs acceptance, reconstruction and draft feedback in one captured graph per batch bucket ([study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/device-accept.md)); 27B: exact, +2% at batch one, no serving gain at batch four yet; Flash-Next MTP: served x1.057 batch one, x1.088 at four streams, outputs not identical to the host path ([Flash-Next](flashnext.md#round-idle-gaps-2026-09-28)) |
| Shape-stable prefill numerics | Off | `SHARAR_STABLE_NUMERICS=1` (research baseline for row-count experiments; the [numerics study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/target-numerics.md)) |
| Draft checkpoint | `draft-runs/current` when promoted, else pinned | `SHARAR_DRAFT=/dir` or `SHARAR_DRAFT=pinned` |

The launcher's defaults are the measured profile (FP8 DFlash2 with 15 proposals, relaxed acceptance, 3-bit layer and draft copies); the publisher's BF16 draft is

```bash
SHARAR_DRAFT_PRECISION=bf16 ./run-sharar.sh
```

For MTP without the external draft, use
`SHARAR_PROPOSER=mtp SHARAR_SPEC=3 ./run-sharar.sh`. Set `SHARAR_SPEC=0` for
ordinary decode. DSpark (`SHARAR_PROPOSER=dspark SHARAR_SPEC=7 SHARAR_MAX_CONTEXT=32768`, pinned
`RadixArk/Qwen3.8-27B-DSpark@b9a5dbd`) keeps draft features for the whole context rather than DFlash2's
2,048-token window, so it serves at up to 32K. On 2026-10-03, exact profile, 10K code workload: 30.7 decode
TPS against DFlash2's 37.0. `sharar serve --help` lists CLI choices; the launcher also
owns container and guard setup.

Before replacing an existing container, use `./run-sharar.sh --stop` and
launch the intended configuration. Updating Git or rebuilding a tag does not
replace a running container. Validate the chosen image/profile before a
production rollout; source benchmark qualification is separate from deployment.

### Settings

Every `SHARAR_*` variable the launcher forwards is read once into one
`EngineConfig` (`sharar/engine/config.py`) when the server starts; the command
line overrides it. `GET /get_server_info` returns the effective value under
`config`, so a run records exactly what it served.

## Training the drafter

`./sharar-ctl train` stops serving and trains the DFlash2 drafter on the
recorded traces until `./sharar-ctl train stop`, which evaluates once more,
promotes on improvement and relaunches serving on the promoted checkpoint
(or swaps it into a server that kept running through `POST /reload_draft`).
`./sharar-ctl status` shows serving, training, the draft in use and the trace
directory; `./sharar-ctl draft list|use|pinned` manages checkpoints. The
serving launcher follows `$SHARAR_RESULTS/draft-runs/current` whenever it
resolves to a checkpoint; `SHARAR_DRAFT=pinned` forces the publisher draft and
`SHARAR_DRAFT=/dir` serves any checkpoint saved by `sharar train`.
`/get_server_info` reports the served directory under `speculative.draft`.
See [draft adaptation](draft-adaptation.md) for how training works and its
launcher variables.

## Request traces

Setting `SHARAR_TRACES` mounts that host directory at `/traces` and starts the
server with `--traces`. Each finished request is then appended as one JSON line:
prompt and output token IDs, sampling parameters, the cached prefix length and,
for speculative requests, every verification round with the accepted count, the
draft's proposals and the target's argmax rows (sampled rows keep only the
correction token). DFlash2 requests also store the projected context features
the drafter attended to, copied from the proposer's ring as they are written:
the last 2,048 prompt positions and every committed output position, 10 KiB
per row as raw BF16 under `features/<trace file>/<id>.bin`. No text or client
metadata is stored. Files rotate at 64 MiB of JSON and feature data, and the oldest trace file is
deleted together with its feature directory above the cap. The default cap
is 32 GiB, about 1,300 requests of 2K-token traffic, and it is also the
training window.
Writing happens on a separate thread and never blocks the engine: feature rows
go through pinned staging chunks with CUDA events; a full queue or disk counts
as `dropped` in `/get_server_info` under `traces`. Any request can opt out with
`"trace": false`; the benchmark, draft A/B, accuracy and corpus clients do, so
the directory holds the user's own traffic. `sharar traces --dir DIR` summarizes a directory. The traces are the
input for [draft adaptation](draft-adaptation.md); they are the user's prompts
and stay on this machine unless the user copies them elsewhere.

**Draft confidence (MTP, #32).** With `SHARAR_TRACE_CONFIDENCE=1` beside
`SHARAR_TRACES` (the launcher forwards both), each round record also carries the
MTP draft's top-1 probability `p1` and top-1/top-2 margin `m` per drafted
position; `params` records the request's acceptance rule. The setting is read
when the proposer is built (the draft graphs capture the extra output); off,
nothing changes. It is the calibration data for adaptive draft depth (#43).

Inspect `docker logs sharar-qwen38`, `/health` and `/get_server_info` for status.
The latter reports draft precision, scheduler settings, KV/state geometry and
cache statistics. `cache.draft_features` reports retained features and fallbacks.
At 96 slots, DFlash2 feature retention reserves up to 1,900 MiB before planning
KV. The cold tier uses the same physical DRAM; it adds no physical memory.
`SHARAR_COLD_TIER_GIB` caps its managed-memory budget (the launcher forwards
it; 0 disables the tier; unset sizes it from spare memory, up to 24 GiB).
A parked session spans as many 2 GiB slabs as it needs, so long contexts park
too (before 2026-09-23 anything over about 60K tokens was rejected).

**Session store.** Parked sessions are also written, in the background, to
`$SHARAR_RESULTS/sessions` (`SHARAR_SESSION_DIR`; `off` disables it; budget
`SHARAR_SESSION_GIB`, default 200, least recently used records go first;
sessions shorter than `SHARAR_SESSION_MIN_TOKENS`, default 4,096, are not
written: every record carries the 144 MiB recurrent snapshot, and a short
session re-prefills in about a second).
When the arena drops a session, or after a restart, the next request whose
prompt extends it reads it back (NVMe at 9.7 GB/s: about a second for a
262K-token session, against a 351-second prefill) and resumes exactly where
it was. Records are listed only by a process with the same target revision,
decode weights, geometry and prefill numerics; `/flush_cache` deletes them.
`cache.cold_tier.sessions` in `/get_server_info` reports records, reads,
writes and errors. A new prompt still pays its first prefill; the store only
removes the repeats.
While host `MemAvailable` is under the 16 GiB floor the scheduler holds new
admissions (`floor_holds` in `/get_server_info`) so running requests can
finish and release memory before the external guard has to act.

## Development and checks

CPU development needs Python 3.12:

```bash
bash tools/check.sh setup-cpu
.venv-cpu/bin/python tools/check_distribution.py
python3 tools/check_evidence.py
```

The runtime image is built from pinned [constraints](../constraints/gb10-cu130.txt).
It still includes the compiler needed for SM121 JIT kernels. CPU distributions
include the CUDA source; executing those kernels still requires the GPU stack.
The source-only `SHARAR_FUSED_NORM_NVFP4` switch remains an unpromoted numerical
experiment, off by default. Its reference kernel/tests are retained; it is not
part of the measured default configuration or a launcher optimization option.

`tools/dev-shell.sh` mounts current source and local caches into a development
container of the Sharar runtime image (`SHARAR_DEV_IMAGE`, else the config's `SHARAR_IMAGE`).
Reserve an exclusive GPU window and arrange an active memory guard before
loading a model or running `bash tools/check.sh gpu`. The development helper
does not stop production or start the guard automatically.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for numerical/lifecycle checks and
[historical environment details](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/env.md) for old reference runs.
