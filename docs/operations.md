# Build and operate Sharar

## Profile and prerequisites

Use one NVIDIA GB10 (SM 12.1), `aarch64`, about 128 GB unified memory, NVIDIA
driver 580 or newer, the CUDA 13 toolkit in `/usr/local/cuda` (DGX OS ships it; FlashInfer and Sharar compile
their GB10 kernels on first use), [uv](https://docs.astral.sh/uv/) and a working user systemd manager
(`loginctl enable-linger` keeps it after logout). Two Sparks: [tensor parallel](tensor-parallel.md).
Two targets, one per process, chosen by name: `27b` (`RadixArk/Qwen3.8-27B-NVFP4` at `319f741`, with the
DFlash2 drafter `incoai/Qwen3.8-27B-DFlash2` at `dedf8df`) and `flashnext`
(`RadixArk/Qwen3.8-Flash-Next-NVFP4` at `7b71922`, its own MTP head). The first serve downloads whatever
the cache lacks (about 26 GB for the 27B, 135 GB for Flash-Next) after checking free disk
([tools/fetch-model.py](../tools/fetch-model.py)); serving itself runs offline.

No configuration file is needed. Settings resolve in this order, first wins
([tools/launcher-config.sh](../tools/launcher-config.sh)): the environment and `config/gb10.env` (optional;
copy [the example](../config/gb10.env.example)), the target's settings in `config/targets/<target>.env`
(port, service, drafter), then defaults: models in `SHARAR_HF_CACHE` (default `$HF_HOME`, else
`~/.cache/huggingface`; Flash-Next in `SHARAR_FLASHNEXT_CACHE`, default the same) and results, traces,
sessions and drafter runs in `SHARAR_RESULTS` (default `~/.local/share/sharar`).

## Install and launch

```bash
./sharar-ctl setup                # the Python environment; serve runs it when .venv is missing or stale
./sharar-ctl serve 27b            # or flashnext; ./run-sharar.sh --check previews the launch
tools/verify-sharar.sh
./sharar-ctl logs -f              # $SHARAR_RESULTS/logs/<service>.log
```

`tools/setup-venv.sh` builds `.venv` (or `SHARAR_VENV`) with uv from [the pinned constraints](../constraints/gb10-cu130.txt):
CUDA 13 PyTorch, FlashInfer with its JIT cache, the serving packages, and `sharar` installed editable. It
records a stamp of the constraints, `pyproject.toml` and itself, and the launchers rebuild the environment when
the stamp no longer matches. The launcher runs `python -m sharar serve` from the checkout as a user systemd service
(`systemd-run --user`, unit `sharar-qwen38` or `sharar-flashnext`): `stop` sends SIGTERM, and the server drains its
requests and stores its sessions first. Its log is `$SHARAR_RESULTS/logs/<service>.log` (the previous run's is
`.log.1`). The launcher also starts `sharar-memory-guard.service`. The guard protects a 16 GiB `MemAvailable` floor.
Startup needs at least 40 GiB available and checks the prospective pool/draft
budget separately. Automatic KV capacity depends on available memory.

| Setting | Launcher default | Override |
|---|---|---|
| Target | the one named on `serve` (`27b` or `flashnext`) | `SHARAR_TARGET`; Flash-Next reads its model from `SHARAR_FLASHNEXT_CACHE` and takes `SHARAR_FLASHNEXT_TRUNK=fp8\|bf16` |
| Serving profile | `fast` | `sharar-ctl serve --exact\|--balanced\|--fast\|--throughput` or `SHARAR_PROFILE` ([below](#serving-profiles)) |
| Proposer / draft proposals | 27B: DFlash2, 15 (16 verify rows); Flash-Next: MTP, 4 | `SHARAR_PROPOSER`, `SHARAR_SPEC` (0 = plain decode) |
| Draft precision | FP8 | `SHARAR_DRAFT_PRECISION=bf16` for the publisher's precision |
| Draft checkpoint | `draft-runs/current` when promoted, else pinned | `SHARAR_DRAFT=/dir` or `SHARAR_DRAFT=pinned` |
| Accepted state | reconstruction | `SHARAR_VERIFY_STATE` |
| Context per request (prompt plus output) | 262,144 | `SHARAR_MAX_CONTEXT` / `sharar serve --context-length` (the engine's own default is 32,768). The pool guarantees one request at that length (8 GiB of FP8 KV at 262K) or four at 32K, whichever is more; concurrent requests share the rest and a request that does not fit waits |
| Requests decoded together | 4; `throughput`: 16 on the 27B, 128 on Flash-Next (64 with NVFP4 experts) | `SHARAR_MAX_BATCH` / `sharar serve --max-batch`, 1-128 (graph buckets 1, 2, 4, ... 128). Above four, the pool guarantees the batch-four total (4 x 32K tokens) split between the running requests, or one request at the full context |
| Verify depth per batch size | the proposer's depth at every batch; `throughput` on Flash-Next: `16:3` | `SHARAR_SPEC_DEPTHS=BUCKET:DEPTH,...` / `sharar serve --spec-depths` (0 = plain decode). The 3-bit copies serve calls of at most 64 rows, so wider verifications run the original weights |
| Python environment | `.venv` in the checkout | `SHARAR_VENV=/dir` (built there by `tools/setup-venv.sh`) |
| Service / port | 27B `sharar-qwen38` / 30001, Flash-Next `sharar-flashnext` / 30004 | `SHARAR_SERVICE` / `SHARAR_PORT` |
| Request traces | on under `sharar-ctl` (`$SHARAR_RESULTS/traces`), off under `run-sharar.sh` | `sharar-ctl serve --no-traces`; `SHARAR_TRACES=/dir` (cap `SHARAR_TRACES_MAX_MIB`, 32768) |
| Calibrated 3-bit checkpoint | `$SHARAR_RESULTS/w3-checkpoints/<target>-current`, fetched on first serve | `SHARAR_W3_CHECKPOINT=/dir`. Without one the 27B rounds to nearest and Flash-Next serves its FP8 trunk; the preflight and `/get_server_info` (`w3_warning`) say so ([calibration](w3-calibration.md)) |
| Flash-Next routed experts | the W3 checkpoint's 3-bit experts when it carries them (`fast`, `throughput`) | `SHARAR_FLASHNEXT_EXPERTS=nvfp4` keeps the checkpoint's NVFP4 experts |
| Shape-stable prefill numerics | off | `SHARAR_STABLE_NUMERICS=1` (M-invariant kernels for the three row-count-dependent ops; see [limits](#numerical-limits)) |

For MTP on the 27B, use `SHARAR_PROPOSER=mtp SHARAR_SPEC=3`. DSpark (`SHARAR_PROPOSER=dspark SHARAR_SPEC=7
SHARAR_MAX_CONTEXT=32768`, pinned `RadixArk/Qwen3.8-27B-DSpark@b9a5dbd`) keeps draft features for the whole
context rather than DFlash2's 2,048-token window, so it serves at up to 32K. `sharar serve --help` lists the
CLI choices; the launcher also owns the service and guard setup.

The running server keeps the code it started with: after `git pull`, run `./sharar-ctl stop` and `serve`.

## Serving profiles

A profile sets the default of every speed/fidelity knob ([tools/serving-profile.sh](../tools/serving-profile.sh));
a knob set in the environment or the config file wins. `/get_server_info` reports the profile under `config`.

| Knob | exact | balanced | fast (default) | throughput |
|---|---|---|---|---|
| Relaxed acceptance, greedy requests (`SHARAR_ACCEPT_DEFAULT`) | off | top-2 within 1.0 logit | top-3 within 2.0 logits | as fast |
| Relaxed acceptance, sampled requests (`SHARAR_ACCEPT_DEFAULT_SAMPLED`, `SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD`) | off | off | Flash-Next top-3 within 2.0; the 27B top-2 within 1.0 (at 3:2.0 its sampled thinking looped in Claude Code sessions) | as fast |
| Calibrated 3-bit decode copies (`SHARAR_W3_DECODE`; on Flash-Next also its routed experts) | off | the 27B only | on | on |
| Device-resident acceptance (`SHARAR_DEVICE_ACCEPT`) | off | off | Flash-Next only | as fast |
| Requests decoded together (`SHARAR_MAX_BATCH`) | 4 | 4 | 4 | 27B 16; Flash-Next 128 (64 with NVFP4 experts) |

Relaxed acceptance accepts a drafted token the target ranks within its top-k and within a logit margin of its
argmax, so outputs are no longer the target's exact greedy continuation; a request can pin exact acceptance with
`accept_top_k: 1` ([API](api.md)). The 3-bit copies (3.25 bits per weight, a tensor-core kernel for calls of at
most 64 rows) cover the 27B's MLPs, FP8 projections and FP8 draft (+9.8 GiB resident) and Flash-Next's trunk; Flash-Next's
3-bit routed experts replace the NVFP4 ones at every row count (17 GiB less resident). The lm_head keeps its
original format. Speculative decoding itself, the Flash-Next 3-bit draft head, FP8/NVFP4
weights and CUDA graphs are lossless and on in every profile. Benchmarks, training traces and gates use `exact`.
A new lossy option enters `balanced` or `fast` only through the lossy-change gate
([CONTRIBUTING](../CONTRIBUTING.md#judge-a-lossy-option)).

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

**Draft confidence (MTP).** With `SHARAR_TRACE_CONFIDENCE=1` beside
`SHARAR_TRACES` (the launcher forwards both), each round record also carries the
MTP draft's top-1 probability `p1` and top-1/top-2 margin `m` per drafted
position; `params` records the request's acceptance rule. The setting is read
when the proposer is built (the draft graphs capture the extra output); off,
nothing changes. It is the calibration data for adaptive draft depth.

Inspect `./sharar-ctl logs`, `/health` and `/get_server_info` for status.
The latter reports draft precision, scheduler settings, KV/state geometry and
cache statistics. `cache.draft_features` reports retained features and fallbacks.
At 96 slots, DFlash2 feature retention reserves up to 1,900 MiB before planning
KV. The cold tier uses the same physical DRAM; it adds no physical memory.
`SHARAR_COLD_TIER_GIB` caps its managed-memory budget (the launcher forwards
it; 0 disables the tier; unset sizes it from spare memory, up to 24 GiB).
A parked session spans as many 2 GiB slabs as it needs, so long contexts park too.

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
python3 tools/check_links.py
```

On the GB10, `.venv` is the development environment too: `.venv/bin/python -m sharar ...` runs this checkout's
code (the editable install), and `tools/gpu-tests.sh` runs the GPU suite from it. Reserve an exclusive GPU window
(`flock $SHARAR_RESULTS/.gpu.lock ...`) before loading a model. CPU distributions include the CUDA source;
executing those kernels still requires the GPU stack.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for numerical and lifecycle checks.

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
