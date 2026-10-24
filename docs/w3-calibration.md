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

Both targets have one: Flash-Next's (`yemara/Qwen3.8-Flash-Next-NVFP4-sharar-w3`: the 3-bit trunk and the GPTQ
routed experts) and the 27B's (`yemara/Qwen3.8-27B-NVFP4-sharar-w3`, GPTQ from the FP8 release on the `bc7984f`
inputs). Nemotron's (the GPTQ trunk projections and routed experts) is not published yet: its `W3_REPO` is unset,
so `fast` serves a local `nemotron-current` link, or the NVFP4 experts and FP8 trunk with a warning. To serve a new
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


## Nemotron

`sharar/weights/nemotron_w3_calibrate.py` fits 3-bit decode copies of Nemotron-3.5-Super's FP8 projections, the 97
matrices `Nemotron.linears()` lists: the 40 Mamba blocks' `in_proj` [18560, 4096] and `out_proj` [4096, 8192], the 8
attention blocks' q|k|v [4608, 4096] and o [4096, 4096], and the lm_head [131072, 4096]. They are 1.97 GiB next to
the 4.9 GiB of FP8 that stays resident for prefill. The source is the served FP8 matrix (the checkpoint has no wider
release; the copy approximates what `exact` serves), the covariance the FP8 activation each projection reads with every
upstream block at its served precision. Two steps keep the full-model part short:

```bash
python3 -m sharar.weights.w3_calibration_inputs --tokenizer SNAP/tokenizer.json --commit bc7984f --output code.json
python3 -m sharar.weights.nemotron_w3_inputs --tokenizer SNAP/tokenizer.json --code code.json \
  --traces answers-sharegpt.json --output nemotron-calibration-inputs.json
SHARAR_TARGET=nemotron python3 -m sharar.weights.nemotron_w3_calibrate capture --inputs nemotron-calibration-inputs.json --activations ACT
SHARAR_TARGET=nemotron python3 -m sharar.weights.nemotron_w3_calibrate fit --activations ACT \
  --output $SHARAR_RESULTS/w3-checkpoints/<date>-nemotron-gptq
```

- **Inputs.** Nemotron's tokenizer is its own, so the Qwen targets' sample files do not carry over. The set is the
  coding proxy at `bc7984f` (64 files) plus 80 ShareGPT first turns (48 English, 32 other languages) with Flash-Next's
  reasoning traces (`flashnext_expert_inputs generate --thinking`), rendered in Nemotron's chat template: 144 samples,
  202K tokens, sha256 `edc0bd1d…`.
- **Capture** (`capture`, 3.6 minutes after the 77 s load): each sample prefills alone through the serving model and every
  projection call adds x^T x; the lm_head's input is the final norm of every position. Every eighth sample is held out
  as probe rows (18 samples, 29K tokens). 15 GB of covariances on disk.
- **Fit** (`fit`, 4 minutes, no model load, `--resume`): GPTQ as for the 27B (damping 0.01, the Lloyd codebook,
  32-weight groups), one matrix at a time from the snapshot. The manifest reports each matrix's relative weight error
  and relative output error, tr(D H D^T) / tr(W H W^T) under the calibration covariance and on the held-out probes,
  for GPTQ and for nearest rounding of the same matrix. `--method nearest` writes the nearest-rounded comparison,
  `--exclude REGEX` leaves matrices FP8.

Relative output error (squared, median over the matrices of a kind; held-out probes):

| Projection | Matrices | Nearest | GPTQ | GPTQ weight error |
|---|---|---|---|---|
| Mamba `in_proj` | 40 | 0.0087 | 0.0044 | 0.215 |
| Mamba `out_proj` | 40 | 0.0212 | 0.0094 | 0.211 |
| attention q\|k\|v | 8 | 0.0120 | 0.0053 | 0.221 |
| attention o | 8 | 0.0363 | 0.0093 | 0.211 |
| lm_head | 1 | 0.0005 | 0.0001 | 0.203 |

GPTQ trades a larger weight error (nearest: 0.18-0.19, q|k|v up to 0.29) for a two- to fourfold smaller output error.
`--unbias weight` (the per-row gain Flash-Next uses) raised the logits KL against `exact` from 0.067 to 0.073 here and
is off.

Which matrices to serve 3-bit (Sharar, spark-9931, the 2026-10-23 GPTQ fit). `profiling/nemotron_w3_sensitivity.py`
scores the lossy gate's layer 1 for many coverages in one load: the gate's frozen sequences, 64 rows per forward,
KL(exact || arm), argmax agreement, excess over the 32-row floor (KL 0.0179, argmax 96.4%). Its `all` arm reproduces
the gate's candidate exactly (KL 0.0667):

| Served 3-bit | Matrices | Decode GB saved | KL | Argmax | Within the floor |
|---|---:|---:|---:|---:|---|
| none | 0 | 0 | 0.0000 | 99.0% | yes |
| lm_head | 1 | 0.32 | 0.0086 | 97.3% | yes |
| lm_head + out_proj of blocks 66-86 | 11 | 0.52 | 0.0181 | 96.3% | pooled yes (math, multilingual no) |
| lm_head + every out_proj | 41 | 1.11 | 0.0383 | 95.2% | no |
| out_proj only / in_proj only / attention only | 40 / 40 / 16 | 0.80 / 1.80 / 0.17 | 0.0314 / 0.0384 / 0.0232 | 95.8 / 94.9 / 95.9% | no |
| all but the lm_head | 96 | 2.77 | 0.0591 | 93.9% | no |
| all | 97 | 3.09 | 0.0667 | 93.3% | no |
| all, nearest rounding | 97 | 3.09 | 0.1008 | 91.4% | no |

The errors add sub-additively: the floor itself (two chunk sizes of the same weights) costs 3.5 points of argmax, and
any ten trunk matrices already land near it. Attention costs the most per byte. GPTQ is worth a third of the KL
over nearest rounding. Three refinements that lowered the per-matrix held-out output error did not lower the KL:
act order (`--act-order`, static groups, 5-10% lower output error; KL 0.0709), the per-row gain (`--unbias`), and
FP8 rows for the four massive residual channels (345, 3588, 1115, 3939) of every out_proj and o (KL 0.0673). So a 3-bit
trunk does not fit inside layer 1's floor. On tasks it holds (GSM8K, paired, `tools/evaluate.py --tasks gsm8k`):

| Served 3-bit | GSM8K | Paired with | Delta [95%] | Better / worse |
|---|---:|---|---:|---:|
| all 97 (spark-9931, first 300) | 97.0% | 2026-10-23 exact, 97.7% | -0.7 pp [-2.0, +0.6] | 1 / 3 |
| lm_head + every out_proj (Spark 3, 1319) | 96.6% | exact@8219ee1, 96.8% | -0.2 pp [-0.8, +0.3] | 5 / 8 |
| lm_head + out_proj of blocks 66-86 (Spark 3, 1319) | 96.7% | exact@8219ee1, 96.8% | -0.1 pp [-0.7, +0.5] | 7 / 8 |

Speed (Sharar, spark-9931, `profiling/nemotron_w3_kernels.py`, weights rotated past the L2): at 1-4 rows the 3-bit
GEMM reads the copies at the FP8 GEMV's bandwidth, 2.4-2.5 times faster: in_proj 134 µs against 327, out_proj 59
against 145, q|k|v 35 against 81, o 31 against 73, the lm_head 0.94 ms against 2.23 (at 5-16 rows, where FP8 changes
kernel, 2.5-2.9 times). One depth-3 verify reads 22.3 ms of FP8 projections, 9.2 ms of copies. Decode
(`profiling/nemotron_bench.py 0,3`, tok/s, FP8 → all 97 copies): MTP depth 3, one request 42.6 → 56.9 (+34%, the round
75 → 59 ms), four 71.6 → 84.0 (+17%); no drafts, one request 22.2 → 30.6, four 48.7 → 57.2. Head + out_proj reads 48.7
and 75.6; the head alone 46.7 and 80.4.

`capture-upstream` (with `fit` on its output) is the asymmetric variant: it pairs each projection's inputs in the
FP8 model with its inputs when the 3-bit copies of `--checkpoint` serve upstream (a whole-sample FP8 prefill, then
64-row chunks), and the fit quantizes the least-squares target W + W (Q - H~) H~^-1 under H~ = x~^T x~, Q = x^T x~:
each copy then answers for the error that reached its input as well as its own.

`Nemotron.__init__` loads the copies with `SHARAR_W3_DECODE=1 SHARAR_W3_CHECKPOINT=DIR` (calibrated only: without a
checkpoint the setting is refused) through `attach_calibrated`; a projection the checkpoint does not carry stays FP8.
`ops.fp8_linear` reads a copy for calls of at most 64 rows, FP8 activations at the layer's static scale as they are;
prefill keeps the FP8 weights. The MTP draft reads the lm_head's copy through the target head.

### Routed experts (Nemotron)

`sharar/weights/nemotron_w3_experts.py` writes 3-bit copies of every latent MoE block's 512 routed experts, the 40
trunk blocks' and the MTP head's, as `layers.N.experts.w1` / `.w2` and `mtp.experts.w1` / `.w2` (up [512 x 2688,
1024] in the checkpoint's row order, down [512 x 1024, 2688]). Served, they are the only expert copy, for every call
from decode to prefill (`kernels/w3_moe.py`'s ReLU² geometry; the NVFP4 experts are not loaded, and the block's
small-call NVFP4 kernels give way to route + `w3_moe`): 51.9 GiB resident after the load against 66.9 GiB, 9.02 GB
streamed per decode token against 9.77. A checkpoint may carry trunk entries too (`fit --source`); `attach_calibrated`
attaches those and leaves the experts to their blocks.

```bash
python3 -m sharar.weights.nemotron_w3_experts capture --inputs nemotron-expert-inputs.json --activations ACT
python3 -m sharar.weights.nemotron_w3_experts fit --method gptq --activations ACT \
  --output $SHARAR_RESULTS/w3-checkpoints/nemotron-experts3-gptq-<date> --cheap-output .../nemotron-experts3-rtnz-<date>
```

- **Inputs** (`inputs`): the coding proxy at bc7984f in Nemotron's tokenizer, 250 ShareGPT reasoning traces and 600
  packed MMLU/MMLU-Pro validation answers in Nemotron's chat template: 348 samples, 530K tokens.
- **Capture** (10 minutes after the load): every MoE block's routed latent rows with their top-22 ids and weights
  (the MTP head's from its teacher-forced depth-1 input), 529,827 rows a block, 45 GB.
- **Fit** (about 135 s a block on the GB10, 1.5 hours for 41; resumable per block, `--deadline` for locked windows,
  peak 9.2 GiB): per expert H1 = sum g² x xᵀ over its routed rows for up and H2 the same over relu(x upᵀ)² from the
  NVFP4 up for down, shrunk toward the block's pool (n0 = 256 rows), damping 0.01, GPTQ batched over 16 experts. The
  covariances are made one chunk of experts at a time, twice (pooling and the codebook choice, then the fit).
- **Codebooks.** The kernel takes fc2's codebook separately from fc1's. Each matrix takes the candidate (Lloyd's eight
  levels, or `zero8` = -3.5, -2.5, -1.5, -0.625, 0, 0.875, 2, 3.5) with the smaller nearest-rounding output error over
  a quarter of the block's experts. The checkpoint's NVFP4 down projections carry many exact zeros (block 3's are
  84-94% zeros), which Lloyd's levels cannot hold: zero8 won every down projection (Lloyd's error 1.01-5.5x zero8's)
  and two up projections (block 79's and the MTP head's).

Relative errors against the NVFP4 experts (the mean over 41 blocks; output error under the calibration covariances, so
in-sample): nearest rounding with the per-matrix codebook 0.164 / 0.166 weight and 0.088 / 0.165 output (up / down),
GPTQ 0.193 / 0.230 weight and 0.059 / 0.086 output.

Teacher-forced logits against NVFP4 experts on 16 held-out ShareGPT prompts (15,680 positions; Sharar, spark-9931):

| Experts | KL | top-1 agreement | NLL (NVFP4 1.3552) |
|---|---|---|---|
| nearest, Lloyd | 0.189 | 84.7% | 1.3715 |
| nearest, per-matrix codebook | 0.078 | 90.5% | 1.3522 |
| GPTQ, per-matrix codebook | 0.050 | 92.5% | 1.3622 |

Decode tok/s (`profiling/nemotron_bench.py 0,3`, 256 tokens, greedy, four fixed prompts; Sharar, spark-9931, on
feat/nemotron-super 8219ee1 with the experts merged):

| MTP depth, batch | NVFP4 experts | GPTQ 3-bit experts |
|---|---|---|
| 0, 1 | 22.2 | 22.4 |
| 0, 4 | 48.5 | 55.6 |
| 3, 1 | 42.5 | 46.6 |
| 3, 4 | 71.4 | 85.5 |

At one row the 3-bit path gains little: the block's NVFP4 small-call kernels fuse the router and the shared expert,
which route + `w3_moe` do not.

Quality, `exact` with the GPTQ experts against `exact` (`tools/evaluate.py --suite quick`, paired; Sharar, spark-9931):
GSM8K 97.7% vs 97.7%, HumanEval 94.5% vs 95.1%, MGSM-pl24 89.4% vs 90.5%, needles 6/6 both; pooled -0.5 pp
[-1.7, +0.6], p 0.48: within noise.
