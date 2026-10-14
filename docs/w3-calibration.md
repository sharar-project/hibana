# Calibrated three-bit decode weights

The 3-bit decode copies store each weight as one of eight levels of an E4M3 codebook, with an E4M3 scale per
32 weights and a per-row multiplier: `3.25 + 32/K` bits per weight for input width `K`. Inside the kernel, codes
expand to FP8, multiply FP8 activations and accumulate in FP32; activations are not 3-bit. The copies serve
calls of at most 64 rows (decode and verification); larger calls use the original NVFP4/FP8 weights, which stay
resident. On the 27B they cover 304 target matrices (MLP `gate_up` and `down` in all 64 layers, GDN `in_qkv`,
`in_z`, `out`, attention `qkv`, `o`) plus 20 in the DFlash2 drafter (nearest-rounded at load, not calibrated);
on Flash-Next, the trunk and the routed experts (below). Coverage and dispatch: `sharar/weights/w3_checkpoint.py`, `sharar/kernels/ops.py`,
`sharar/model/draft_linear.py`.

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

From the Python environment, with the GPU otherwise idle and the memory guard watching:

```bash
SHARAR_W3_DECODE=0 uv run python -m sharar.weights.w3_calibrate \
  --inputs calibration-inputs.json \
  --source $SHARAR_HF_CACHE/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --output $SHARAR_RESULTS/w3-checkpoints/<date>-qwen38-gptq
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
the decode-path KL it causes, though it did not move task accuracy, so the served checkpoint does not use it.

## Where the served copies come from

The profiles that use 3-bit copies (`fast`, `throughput`)
serve `$SHARAR_RESULTS/w3-checkpoints/<target>-current` (`qwen38-current`,
`flashnext-current`). Every `sharar serve` (one Spark or two) keeps that link on the target's published
checkpoint, pinned by revision in the target's config (`W3_REPO`, `W3_REVISION`)
(`python -m sharar.weights.w3_checkpoint fetch`; `HF_TOKEN` is passed through for a private repository):

- no link, or a link to a revision fetched earlier (`<target>-<12-character revision>`): the pinned revision is
  downloaded into `w3-checkpoints/<target>-<revision>`, verified, and linked; older directories are kept;
- a link to any other directory is a local calibration: it stays, and the serve prints a note with the command that
  switches to the published copies (`fetch --target T --root R --switch`).

Both targets have one: Flash-Next's (`yemara/Qwen3.8-Flash-Next-NVFP4-sharar-w3`: the 3-bit trunk and the GPTQ
routed experts) and the 27B's (`yemara/Qwen3.8-27B-NVFP4-sharar-w3`, GPTQ from the FP8 release on the `bc7984f`
inputs, 2026-10-01). To serve a new calibration, point the link at its directory or set
`SHARAR_W3_CHECKPOINT=/path/to/calibrated-w3`.

The loader checks the target revision, projection coverage, dimensions, dtypes
and hashes before graph capture. `/get_server_info` exposes `w3_target_identity`,
the manifest's SHA256 (nearest rounding reads `nearest:lloyd`). Without a
checkpoint the 27B packs nearest-rounded copies and Flash-Next serves its FP8
trunk (its copies are calibrated only); the preflight prints a warning and
`/get_server_info` reports it as `w3_warning`.

The loader stages each mapped tensor in ordinary CPU storage before copying
it to CUDA (warm-cache target construction 63 s to 12 s, byte-identical).

A fitted artifact is a candidate until it passes the lossy-change gate
(`tools/lossy-gate.py`) against the intended serving profile.

## Routed experts (Flash-Next, #73)

`sharar/weights/flashnext_w3_experts.py` writes a new Flash-Next checkpoint directory: the source checkpoint's trunk entries
(copied and verified) plus every layer's routed experts as `layers.N.experts.w13` (fc1, rows in
`w3_moe.fc1_order`) and `layers.N.experts.w2`, each one W3 matrix stacked over the 512 experts, with the same
manifest format and per-file SHA256; `manifest.json` gains an `experts` block (method, codebook, scale grid,
relative weight and output error per layer). There is no BF16 expert source: both methods fit the checkpoint's NVFP4
experts, dequantized, with the trunk's Lloyd codebook.

- `--method cheap` (2026-09-30): group scales on a 0.5-1.0 clipping grid minimizing the weight error
  (`w3_calibration.fit_scales` with unit importance) and the nearest level. No activations. It lost recall
  (MMLU-Pro law -13 pp against `exact`).
- `--method gptq`: GPTQ on routed activations. `capture` prefills the calibration samples through the serving
  model (FP8 trunk, NVFP4 experts) and stores each layer's MoE input rows on disk (about 4.5 GB a layer for 0.9M
  rows); `fit` then takes one layer at a time: it routes the rows with the layer's router, accumulates per expert
  H = sum g^2 x x^T (g the routing weight) for fc1 and the same over silu(gate) * up of the NVFP4 fc1 for fc2, and
  runs `w3_calibration.quantize`'s pass batched over 32 experts (`gptq_batched`, tested against it). Experts with
  fewer than `--min-tokens` routed rows keep the cheap fit; sparse experts' covariances are shrunk toward the
  layer's pooled one (weight n0 / (n + n0), `--prior-tokens`). `--cheap-output` writes the cheap checkpoint from
  the same pass.

Samples: the frozen coding-proxy set (`w3_calibration_inputs` at bc7984f, the trunk's, sha256 `1f041589…`) plus
knowledge questions (`sharar/weights/flashnext_expert_inputs.py`): MMLU's and MMLU-Pro's validation splits with
anything overlapping MMLU-Pro test or GPQA removed, answered by the `fast` server itself (thinking off), rendered with
the chat template.

Reasoning traces make the samples look like real use (long thinking, other languages): `prompts-sharegpt` takes first
user turns from ShareGPT (`--english`, `--other`), `generate --thinking --temperature 0.7` keeps each reply's
`reasoning_content`, and `build --max-tokens 4096` renders it as the model writes it (`<think> ... </think>`, then the
answer). In the 2-bit expert study (#143) adding 3.6M such tokens to the samples above took a 2-bit GPTQ fit from -4.8
to -3.0 points on the lossy gate's screen (HumanEval -9.8 to -3.0); its effect on the 3-bit fit was not measured.
Samples of 4,096 tokens need the capture's default `--pages 16384`; `capture --resume` continues an interrupted capture
after its last whole sample.

```bash
SHARAR_TARGET=flashnext python3 -m sharar.weights.flashnext_w3_experts capture --inputs expert-inputs.json --activations ACT
SHARAR_TARGET=flashnext python3 -m sharar.weights.flashnext_w3_experts fit --method gptq --activations ACT \
  --source $SHARAR_RESULTS/w3-checkpoints/flashnext-current --output $SHARAR_RESULTS/w3-checkpoints/<date>-flashnext-experts3-gptq \
  --cheap-output $SHARAR_RESULTS/w3-checkpoints/<date>-flashnext-experts3
```

Mean relative weight error against the NVFP4 experts: 0.164 (fc1) and 0.163 (fc2); nearest rounding without the
scale search 0.187-0.195, the screened simulation's E2M1 level set 0.180-0.190. Serving the result replaces the
NVFP4 experts (they are not loaded at all), so the checkpoint is also 17.2 GiB less resident memory.
The GPTQ fit (output error against the NVFP4 experts 0.072 fc1 / 0.120 fc2) matched `fast` with NVFP4 experts
within noise on MMLU-Pro, GPQA Diamond, GSM8K and HumanEval, and is the pinned checkpoint's (`694e279`).

