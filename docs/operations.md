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
| Proposer | DFlash2 | `SHARAR_PROPOSER` |
| Draft proposals | 15 (16 target verification rows) | `SHARAR_SPEC` |
| Draft precision | BF16 | `SHARAR_DRAFT_PRECISION=fp8` |
| Accepted state | Reconstruction | `SHARAR_VERIFY_STATE` |
| Image | `sharar:qwen38-27b` | `SHARAR_IMAGE` |
| Container / port | `sharar-qwen38` / 30001 | `SHARAR_CONTAINER` / `SHARAR_PORT` |
| Request traces | Off | `SHARAR_TRACES=/dir` (cap `SHARAR_TRACES_MAX_MIB`, 8192) |
| Draft checkpoint | `draft-runs/current` when promoted, else pinned | `SHARAR_DRAFT=/dir` or `SHARAR_DRAFT=pinned` |

The optimized measured profile uses FP8 DFlash2 with 15 proposals:

```bash
SHARAR_DRAFT_PRECISION=fp8 ./run-sharar.sh
```

For MTP without the external draft, use
`SHARAR_PROPOSER=mtp SHARAR_SPEC=3 ./run-sharar.sh`. Set `SHARAR_SPEC=0` for
ordinary decode. `sharar serve --help` lists CLI choices; the launcher also
owns container and guard setup.

Before replacing an existing container, use `./run-sharar.sh --stop` and
launch the intended configuration. Updating Git or rebuilding a tag does not
replace a running container. Validate the chosen image/profile before a
production rollout; source benchmark qualification is separate from deployment.
No production rollout is part of the September 11 maintenance cleanup.

## Training the drafter on this machine's traffic

Serve with `SHARAR_TRACES=/dir` so requests are recorded (below). When the
machine is free, `./run-sharar-train.sh start` stops serving and trains the
DFlash2 drafter on every recorded request, newest tenth held out, continuing
from the current promoted checkpoint if there is one. It evaluates held-out
tokens per round every 100 steps and after `./run-sharar-train.sh stop`, and
promotes a checkpoint only when that improves; promotion updates
`$SHARAR_RESULTS/draft-runs/current`. `stop` waits for the final evaluation
(up to 20 minutes) and relaunches serving. `status` shows both. The serving
launcher follows `draft-runs/current` whenever it resolves to a checkpoint;
`SHARAR_DRAFT=pinned` forces the publisher draft and `SHARAR_DRAFT=/dir`
serves any checkpoint saved by `sharar train`. `/get_server_info` reports the
directory under `speculative.draft`. Training and serving never share the
GPU: the trainer refuses to start beside a running server on this machine
(`--keep-serving` is for a second machine with a synced trace directory).
Trained weights change speed only; the target's own shape-dependent numerics
can still change greedy continuations when round boundaries move, as they do
for any change of verification width or batch shape.

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
metadata is stored. Files rotate at 64 MiB and the oldest trace file is
deleted together with its feature directory above the cap (default 8 GiB).
Writing happens on a separate thread and never blocks the engine: feature rows
go through pinned staging chunks with CUDA events; a full queue or disk counts
as `dropped` in `/get_server_info` under `traces`. Any request can opt out with
`"trace": false`. `sharar traces --dir DIR` summarizes a directory. The traces
are the input for [draft adaptation](draft-adaptation.md); they are the user's
prompts and stay on this machine unless the user copies them elsewhere.

Inspect `docker logs sharar-qwen38`, `/health` and `/get_server_info` for status.
The latter reports draft precision, scheduler settings, KV/state geometry and
cache statistics. `cache.draft_features` reports retained features and fallbacks.
At 96 slots, DFlash2 feature retention reserves up to 1,900 MiB before planning
KV. The cold tier uses the same physical DRAM; it adds no physical memory.

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
container. Its historical default is the frozen SGLang reference image. Set
`SHARAR_DEV_IMAGE=sharar:qwen38-27b` explicitly for the current Sharar stack.
Reserve an exclusive GPU window and arrange an active memory guard before
loading a model or running `bash tools/check.sh gpu`. The development helper
does not stop production or start the guard automatically. Use
`--with-reference` only for the isolated reference tools.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for numerical/lifecycle checks and
[historical environment details](archive/studies/env.md) for old reference runs.
