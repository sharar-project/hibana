# Calibrated three-bit decode weights

For an introduction to bits, codebooks, scales and what is specific to this
implementation, start with [3-bit quantization explained](3bit-quantization-explained.md).

`sharar.weights.w3_calibrate` fits the target's optional W3 copies using activation
covariances and the block-128 FP8 release. It uses the
[GPTQ column-error update](https://arxiv.org/abs/2210.17323) with the existing
eight-level E4M3 codebook, 32-weight groups and E4M3 group scales. There is no
activation permutation or new serving kernel. The pinned NVFP4/FP8 prefill
weights and vocabulary head retain their existing arithmetic.

Freeze a JSON object containing `samples`, each with `input_ids`
(`python3 -m sharar.weights.w3_calibration_inputs --tokenizer SNAPSHOT/tokenizer.json --commit C --output FILE`
builds the coding-proxy set from this repository's source files at commit C, reproducibly). Use
representative prompts, at most 2,048 tokens per sample, excluding task
accuracy benchmarks and held-out drafter requests. Source-file separation is
useful for a repository coding proxy. The calibration driver collects one
layer's covariances at a time; it materializes the BF16 SiLU product to observe
the down-projection inputs. It does not train on answers or logits.

In an exclusive, memory-guarded GB10 container, with the source checkpoint and
output directory mounted:

```bash
SHARAR_W3_DECODE=0 python3 -m sharar.weights.w3_calibrate \
  --inputs /proof/calibration-inputs.json \
  --source /source-cache/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --output /proof/calibrated-w3
```

`--layers 1` runs an incomplete development screen. `--resume` verifies input
identities and completed layer checksums before resuming an interrupted run.
Only a complete run publishes `manifest.json`; `progress.json` cannot be
loaded by serving. The manifest records the target/source revisions, frozen
input hash, quantizer settings, projection diagnostics and per-file hashes.
Projection diagnostics reuse calibration rows and are fitting diagnostics,
not held-out quality evidence.

The eight-level codebook is MSE-optimal, so each quantized row comes out shorter than the exact row by
about its relative error squared. `quantize(..., unbias="weight")` (`sharar/weights/flashnext_w3_calibrate.py
--unbias weight`) scales each row back through its row scale; codes, format and bytes do not change. On the
Flash-Next trunk the shrinkage adds coherently through the residual layers and the correction removes most of
the decode-path KL it causes ([Flash-Next](flashnext.md#3-bit-trunk-sensitivity-2026-09-28-61)).

## Where the served copies come from

The profiles that use 3-bit copies (`balanced` for the 27B, `fast`, `throughput`)
serve `$SHARAR_RESULTS/w3-checkpoints/<target>-current` (`qwen38-current`,
`flashnext-current`). When that link is missing, `run-sharar.sh` fetches the
target's published checkpoint, pinned by revision in the target's config
(`W3_REPO`, `W3_REVISION`), into `w3-checkpoints/<target>-<revision>` and links it
(`python3 -m sharar.weights.w3_checkpoint fetch`, run in the image; `HF_TOKEN` is
passed through for a private repository). Both targets have one: Flash-Next's
(`yemara/Qwen3.8-Flash-Next-NVFP4-sharar-w3`) and the 27B's
(`yemara/Qwen3.8-27B-NVFP4-sharar-w3`, GPTQ from the FP8 release on the
`bc7984f` inputs, 2026-10-01). To serve a new calibration, point the link at its
directory or set `SHARAR_W3_CHECKPOINT=/host/calibrated-w3`.

The loader checks the target revision, projection coverage, dimensions, dtypes
and hashes before graph capture. `/get_server_info` exposes `w3_target_identity`,
the manifest's SHA256 (nearest rounding reads `nearest:lloyd`). Without a
checkpoint the 27B packs nearest-rounded copies and Flash-Next serves its FP8
trunk (its copies are calibrated only); the preflight prints a warning and
`/get_server_info` reports it as `w3_warning`.

The loader stages each mapped tensor in ordinary CPU storage before copying
it to CUDA. On GB10 this reduced warm-cache target construction from 63.2 to
12.0 seconds while preserving all 304 matrices byte for byte. It changes
loading, not decode arithmetic or throughput; see the
[runtime profile](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-runtime-profile.md).

A fitted artifact is a candidate until it passes the lossy-change gate
(`tools/lossy-gate.py`) against the intended serving profile. Collect fresh
drafter traces after selecting the target profile. Keep validation and final
test files out of training, and check any trained draft through the actual
FP8/W3 serving path: the trainer's BF16 replay is a checkpoint selection proxy.

## Routed experts (Flash-Next, #73)

`sharar/weights/flashnext_w3_experts.py` writes a new Flash-Next checkpoint directory: the source checkpoint's trunk entries
(copied and verified) plus every layer's routed experts as `layers.N.experts.w13` (fc1, rows in
`w3_moe.fc1_order`) and `layers.N.experts.w2`, each one W3 matrix stacked over the 512 experts, with the same
manifest format and per-file SHA256; `manifest.json` gains an `experts` block (method, codebook, scale grid,
relative weight error per layer). There is no BF16 expert source: it fits the checkpoint's NVFP4 experts,
dequantized, with the trunk's Lloyd codebook, group scales on a 0.5-1.0 clipping grid minimizing the weight error
(`w3_calibration.fit_scales` with unit importance) and the nearest level. No activations are needed (GPTQ on
routed activations is the next step if the gate asks for it). About 30 s a layer on the GB10:

```bash
SHARAR_TARGET=flashnext python3 -m sharar.weights.flashnext_w3_experts \
  --source $SHARAR_RESULTS/w3-checkpoints/flashnext-current --output $SHARAR_RESULTS/w3-checkpoints/<date>-flashnext-experts3
```

Mean relative weight error against the NVFP4 experts: 0.164 (fc1) and 0.163 (fc2); nearest rounding without the
scale search 0.187-0.195, the screened simulation's E2M1 level set 0.180-0.190. Serving the result replaces the
NVFP4 experts (they are not loaded at all), so the checkpoint is also 17.2 GiB less resident memory.

