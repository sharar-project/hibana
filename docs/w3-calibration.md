# Calibrated three-bit decode weights

The 3-bit decode copies store each weight as one of eight levels of an E4M3 codebook, with an E4M3 scale per
32 weights and a per-row multiplier: `3.25 + 32/K` bits per weight for input width `K`. Inside the kernel, codes
expand to FP8, multiply FP8 activations and accumulate in FP32; activations are not 3-bit. The copies serve
calls of at most 128 rows on the 27B and 64 on Flash-Next (decode and verification); larger calls use the original NVFP4/FP8 weights, which stay
resident. On the 27B they cover 304 target matrices (MLP `gate_up` and `down` in all 64 layers, GDN `in_qkv`,
`in_z`, `out`, attention `qkv`, `o`) plus 20 in the DFlash2 drafter (nearest-rounded at load, not calibrated);
on Flash-Next, the trunk and the routed experts (below), and the 20 of its DFlash2 drafter when it serves one. Coverage and dispatch: `sharar/weights/w3_checkpoint.py`, `sharar/kernels/ops.py`,
`sharar/model/draft_linear.py`. Calibrated files store the 27B's `gate_up` rows in the checkpoint's order (gate, then
up); the loader interleaves them in blocks of 64, like the NVFP4 rows, for the fused prefill MLP
(`sharar/kernels/mlp_fusion.py`).

`sharar.weights.w3_calibrate` fits the target's optional W3 copies using activation covariances and the block-128 FP8
release. It uses the [GPTQ column-error update](https://arxiv.org/abs/2210.17323) with the eight-level E4M3 codebook,
32-weight groups and E4M3 group scales. There is no activation permutation or new serving kernel. The NVFP4/FP8
prefill weights and the vocabulary head are unchanged.

Freeze a JSON object containing `samples`, each with `input_ids`
(`python3 -m sharar.weights.w3_calibration_inputs --tokenizer SNAPSHOT/tokenizer.json --commit C --output FILE`
builds the coding-proxy set from this repository's source files at commit C, reproducibly). Use representative
prompts, at most 2,048 tokens per sample, excluding task accuracy benchmarks and held-out drafter requests.
Source-file separation is useful for a repository coding proxy. The calibration driver collects one layer's
covariances at a time; it materializes the BF16 SiLU product to observe the down-projection inputs. It does not train
on answers or logits.

From the Python environment, with the GPU otherwise idle and the memory guard watching:

```bash
SHARAR_W3_DECODE=0 uv run python -m sharar.weights.w3_calibrate \
  --inputs calibration-inputs.json \
  --source $SHARAR_HF_CACHE/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --output $SHARAR_RESULTS/w3-checkpoints/<date>-qwen38-gptq
```

`--layers 1` runs an incomplete development screen. `--resume` verifies input identities and completed layer
checksums before resuming an interrupted run. Only a complete run publishes `manifest.json`; `progress.json` cannot be
loaded by serving. The manifest records the target/source revisions, frozen input hash, quantizer settings,
projection diagnostics and per-file hashes. Projection diagnostics reuse calibration rows and are fitting
diagnostics, not held-out quality evidence.

## Where the served copies come from

The profiles that use 3-bit copies (`fast`, `throughput`)
serve `$SHARAR_RESULTS/w3-checkpoints/<target>-current` (`qwen38-current`,
`flashnext-current`, `nemotron-current`). Every `sharar serve` (one Spark or two) keeps that link on the target's published
checkpoint, pinned by revision in the target's config (`W3_REPO`, `W3_REVISION`)
(`python -m sharar.weights.w3_checkpoint fetch`; `HF_TOKEN` is passed through for a private repository):

- no link, or a link to a revision fetched earlier (`<target>-<12-character revision>`): the pinned revision is
  downloaded into `w3-checkpoints/<target>-<revision>`, verified, and linked; older directories are kept;
- a link to any other directory is a local calibration: it stays, and the serve prints a note with the command that
  switches to the published copies (`fetch --target T --root R --switch`).

Every target has one: Flash-Next's (`yemara/Qwen3.8-Flash-Next-NVFP4-sharar-w3-qad`: the 3-bit trunk trained by
quantization-aware distillation, [below](#qad-trunk-flash-next), and the GPTQ routed experts) and the 27B's (`yemara/Qwen3.8-27B-NVFP4-sharar-w3`, GPTQ from the FP8 release on the `bc7984f`
inputs), and Nemotron's (`shadowpa0327/NVIDIA-Nemotron-3.5-Super-NVFP4-sharar-w3`: the GPTQ trunk projections and
routed experts; private), which its `fast` and `throughput` require: without one they refuse to start. To serve a new
calibration, point the link at its directory or set
`SHARAR_W3_CHECKPOINT=/path/to/calibrated-w3`.

The loader checks the target revision, projection coverage, dimensions, dtypes
and hashes before graph capture. `/get_server_info` exposes `w3_target_identity`,
the manifest's SHA256 (nearest rounding reads `nearest:lloyd`). Without a
checkpoint the 27B packs nearest-rounded copies and Flash-Next serves its FP8
trunk (its copies are calibrated only); the preflight prints a warning and
`/get_server_info` reports it as `w3_warning`.

The loader stages each mapped tensor in ordinary CPU storage before copying it to CUDA, which is much faster on
the GB10 than copying from the memory map.

A fitted artifact is a candidate until it passes the lossy-change gate
(`tools/lossy-gate.py`) against the intended serving profile.

## Routed experts (Flash-Next)

`sharar/weights/flashnext_w3_experts.py` writes a new Flash-Next checkpoint directory: the source checkpoint's trunk entries
(copied and verified) plus every layer's routed experts as `layers.N.experts.w13` (fc1, rows in
`w3_moe.fc1_order`) and `layers.N.experts.w2`, each one W3 matrix stacked over the 512 experts, with the same
manifest format and per-file SHA256; `manifest.json` gains an `experts` block (method, codebook, scale grid,
relative weight and output error per layer). There is no BF16 expert source: both methods fit the checkpoint's NVFP4
experts, dequantized, with the trunk's Lloyd codebook.

- `--method cheap`: group scales on a 0.5-1.0 clipping grid minimizing the weight error
  (`w3_calibration.fit_scales` with unit importance) and the nearest level. No activations; it lost recall against `exact` (MMLU-Pro law -13 pp).
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
answer). Samples of 4,096 tokens need the capture's default `--pages 16384`; `capture --resume` continues an interrupted capture
after its last whole sample.

```bash
SHARAR_TARGET=flashnext python3 -m sharar.weights.flashnext_w3_experts capture --inputs expert-inputs.json --activations ACT
SHARAR_TARGET=flashnext python3 -m sharar.weights.flashnext_w3_experts fit --method gptq --activations ACT \
  --source $SHARAR_RESULTS/w3-checkpoints/flashnext-current --output $SHARAR_RESULTS/w3-checkpoints/<date>-flashnext-experts3-gptq \
  --cheap-output $SHARAR_RESULTS/w3-checkpoints/<date>-flashnext-experts3
```

The pinned checkpoint's GPTQ fit (`694e279`; relative weight error against the NVFP4 experts 0.164 fc1 / 0.163 fc2,
output error 0.072 / 0.120) matched `fast` with NVFP4 experts within noise on MMLU-Pro, GPQA Diamond, GSM8K and
HumanEval. Serving it replaces the NVFP4 experts (they are not loaded), 17.2 GiB less resident memory.

## QAD trunk (Flash-Next)

The pinned Flash-Next checkpoint (`yemara/Qwen3.8-Flash-Next-NVFP4-sharar-w3-qad` at `5c4c62a`) keeps `694e279`'s routed
experts byte for byte and replaces its 231 GPTQ trunk matrices with copies trained by quantization-aware distillation
(NVIDIA ModelOpt): the student runs the decode numerics (3-bit trunk and experts, E4M3 activations per 32-group); each
trunk weight's code (straight-through to the eight Lloyd levels) and a per-row gain on its row scale are trained on
KL(teacher || student) over the vocabulary, the teacher being the checkpoint as `exact` serves it (BF16 trunk, NVFP4
experts). The E4M3 group scales are kept. 33M tokens of Flash-Next's own thinking-on answers (Open-PerfectBlend,
OpenCodeInstruct, UltraChat prompts; GPQA and MMLU-Pro test overlaps removed); 2.0% of the trunk codes changed. Its model
card has the method and data.

On H100 through the SGLang port of these kernels (`w3` arm: 3-bit trunk + routed experts, exact DFlash2 verification),
GPQA Diamond at 64K tokens, three seeds: 83.3% against 76.4% for `694e279` (paired +6.9 points, 95% CI +3.5 to +10.4)
and 86.4% exact; capped answers 18 a seed against 30. With the `fast` techniques (expert sharing 4:12:64, relaxed
top-2 within 1.0), one seed, 82.3-82.8% at batch 8, 16 and 32 against 83.8% without them. LiveCodeBench v6 (131
problems from 2025-02), one seed, 128K tokens: 84.7%; 73.3% counting only answers within 64K, against 77.1% exact at 64K.
These are port numbers, not GB10 numbers: the lossy-change gate (`tools/lossy-gate.py --target flashnext --profile fast`)
against `exact` is the check on a Spark.


## Nemotron

Nemotron's checkpoint (`nemotron-current`) holds GPTQ copies of the 97 FP8 projections `Nemotron.linears()` lists
(the 40 Mamba blocks' `in_proj` and `out_proj`, the 8 attention blocks' q|k|v and o, the lm_head) and of every latent
MoE block's 512 routed experts (the 40 trunk blocks' and the MTP head's). The trunk copies are decode copies (calls of
at most 64 rows read them, prefill keeps FP8); the experts are the only expert copy, at every call size. Two tools
write it, `sharar/weights/nemotron_w3_calibrate.py` (trunk) and `sharar/weights/nemotron_w3_experts.py` (experts),
each a capture through the serving model (FP8 trunk, NVFP4 experts; hold the GPU lock) and a fit from it. Nemotron's
tokenizer is its own, so the samples are rebuilt from the Qwen targets' sources in its tokenizer and chat template
(`sharar/weights/nemotron_w3_inputs.py` builds both sets):

```bash
python3 -m sharar.weights.w3_calibration_inputs --tokenizer SNAP/tokenizer.json --commit bc7984f --output code.json
# trunk: the coding proxy (64 files) and 80 ShareGPT reasoning traces (48 English, 32 other languages; Flash-Next's
# `flashnext_expert_inputs generate --thinking`): 144 samples, sha256 edc0bd1d...
python3 -m sharar.weights.nemotron_w3_inputs --tokenizer SNAP/tokenizer.json --code code.json \
  --traces answers-sharegpt.json --output nemotron-calibration-inputs.json
SHARAR_TARGET=nemotron python3 -m sharar.weights.nemotron_w3_calibrate capture --inputs nemotron-calibration-inputs.json --activations ACT
SHARAR_TARGET=nemotron python3 -m sharar.weights.nemotron_w3_calibrate fit --activations ACT --output TRUNK
# experts: the coding proxy, 180 ShareGPT reasoning traces (110 English, 70 other) and 250 MMLU / MMLU-Pro validation
# answers packed into 104 samples: 348 samples, sha256 bba92b75...
python3 -m sharar.weights.nemotron_w3_inputs --tokenizer SNAP/tokenizer.json --code code.json \
  --traces answers-sharegpt.json --knowledge answers.json --traces-en 110 --traces-other 70 --knowledge-items 250 \
  --seed 20261024 --output nemotron-expert-inputs.json
python3 -m sharar.weights.nemotron_w3_experts capture --inputs nemotron-expert-inputs.json --activations EXPERT_ACT
python3 -m sharar.weights.nemotron_w3_experts fit --activations EXPERT_ACT --source TRUNK \
  --output $SHARAR_RESULTS/w3-checkpoints/nemotron-<name>
ln -sfn nemotron-<name> $SHARAR_RESULTS/w3-checkpoints/nemotron-current
```

- **Trunk** (`capture`: each sample prefills alone and every projection call adds x^T x, every eighth sample held out
  as probes; `fit`: no model load, `--resume`): GPTQ as for the 27B (damping 0.01, the Lloyd codebook, 32-weight
  groups) from the served FP8 matrix (the checkpoint has no wider release), with every upstream block at its served
  precision. The manifest reports each matrix's relative weight and output error, under the calibration covariance
  and on the held-out probes, for GPTQ and for nearest rounding; GPTQ's output error is two to four times smaller, so
  there is no nearest-rounded fallback (`Nemotron` refuses `SHARAR_W3_DECODE=1` without a checkpoint).
- **Experts** (`capture`: every MoE block's routed latent rows with their top-22 ids and weights, the MTP head's from
  its teacher-forced depth-1 input; `fit`: about 135 s a block on the GB10, resumable per block, `--deadline` for
  locked windows; a resume refuses other settings or another capture): per expert H1 = sum g² x xᵀ over its routed
  rows for up and H2 the same over relu(x upᵀ)² from the NVFP4 up for down, shrunk toward the block's pool (n0 = 256
  rows), damping 0.01. Each matrix takes the codebook (Lloyd's eight levels, or `zero8` with an exact zero) with the
  smaller nearest-rounding output error: the checkpoint's NVFP4 down projections carry many exact zeros, so zero8
  wins them. Experts routed fewer than 16 rows keep that nearest-rounded fit. `--source TRUNK` copies the trunk
  entries in, verified; only a complete run publishes the manifest.

The 3-bit experts replace the NVFP4 ones (51.9 GiB resident after the load against 66.9 GiB); the trunk copies add
1.97 GiB beside the 4.9 GiB of FP8 prefill keeps. `fast` serves both, with expert sharing and the FP16 Mamba state:
GSM8K-300 97.7%, paired against `exact`'s 97.7% ([Nemotron](nemotron.md#serving)).
