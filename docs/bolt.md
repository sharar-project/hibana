# Nemotron 3.5 Bolt

`sharar serve bolt` serves the text backbone of
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` on one Spark, on port 30006.
`nemotron-bolt` is an alias. The checkpoint is pinned in `sharar/model/bolt/config.py` and downloaded by
the normal launcher. The C-RADIO vision encoder and projector are not loaded.

```bash
sharar serve bolt
sharar serve bolt --exact
sharar serve bolt --spec 3  # a shorter prefix of LiLiCorr's seven trained slots
sharar serve bolt --spec 0 # ordinary decode
sharar chat bolt
```

Bolt uses the Nemotron Mamba-2 runtime: prefill checkpoints, decode, prefix reuse, SSD eviction/restore and
CUDA graphs follow the same state ownership as Super. Its 88 residual blocks have the same ordering: 40
Mamba-2, 40 LatentMoE and 8 GQA attention blocks without positional embeddings. Its context limit is 262,144.

The Puzzle configuration changes the experts independently in each MoE block: 240, 304, 368, 432 or 496
experts, intermediate widths 1,536–2,560, top-9 routing everywhere. The router reads the full 4,096-wide
hidden state; routed ReLU² experts work at latent width 1,024. An always-on shared ReLU² expert works at
full width. The serialized global `num_experts=512` does not describe these blocks. `Target.moe_block_specs`
and config validation preserve their actual per-block geometry.

## Weights and arithmetic

The loader reads the native safetensors tensors:

| Component | Stored format |
|---|---|
| Routed and shared expert up/down | IQ2_XXS, `[N,K/256,66]` uint8 blocks |
| Attention, latent projections and LM head | Q8_0, `[N,K/32,34]` uint8 blocks |
| Mamba input/output projections | ModelOpt W4A16 NVFP4, with E4M3 block scales and an FP32 global gain |
| Embeddings, norms and router | BF16; router correction bias is FP32 |

Every dense projection runs b12x's SM120 BF16-activation GEMMs (`model/bolt/linear.py`), as vLLM's b12x
linears do: NVFP4 through the W4A16 recipe, Q8_0 and IQ2_XXS through the block-quant recipes. Weights stay in
their checkpoint encodings (Q8_0 and IQ2_XXS are losslessly retiled at load) and activations stay BF16. Plans
for the CUDA graphs' row counts are compiled at load, on the loading thread; `model/bolt/tuning.py` pins measured
launch configurations per projection and row count. The routed experts run b12x's fused IQ2_XXS MoE
(`kernels/b12x_iq2_moe.py`: routing, up projection, ReLU², down projection and the weighted sum in one launch,
BF16 activations). `SHARAR_BOLT_EXPERTS=fp8` selects Sharar's grouped operator (`kernels/iq2_moe.py`: E4M3
activations, four launches), about 5% slower for one request and up to 6% faster for four;
`SHARAR_BOLT_EXPERTS=bf16` keeps the Triton weight-only path (`kernels/ggml_bf16.py`). SSD cache identities
include the linears' b12x version and launch configurations, the expert backend and expert sharing.

Loaded model allocations are about 21.2 GiB before request pools and graphs. One FP32 Mamba slot costs
162.34 MiB across the 40 recurrent blocks; FP8 KV costs 4 KiB per token across eight attention blocks.
The launcher and ledger budget the pools separately. The checkpoint's 24GB name is not the total runtime budget.

## Profiles and limits

`exact` serves the checkpoint encodings, all nine routed experts, FP32 Mamba state and exact verification.
`fast` and `throughput` add the calibrated 3-bit copies of the trunk and vocabulary head (177 matrices, 2.1 GiB;
`weights/bolt_w3_calibrate.py`), the int8 Mamba state (LeapQuant windows) and expert sharing: Super's `route_budget`, each token keeping
its top 6 of 9 experts and a decode or verify call reading at most 20 distinct experts a request and 64 in all
(`SHARAR_MOE_KEEP=6`, `SHARAR_MOE_BUDGET=20:64`). Every profile verifies LiLiCorr's drafts exactly. All profiles use FP8 KV, up to four active requests and the SSD
tier. NVIDIA publishes no 3-bit copies for Bolt: `fast` serves `w3-checkpoints/bolt-current` under the results
directory (a link to a local calibration) or `SHARAR_W3_CHECKPOINT=DIR`, and without one the checkpoint encodings
with a warning.

One GB10, LiLiCorr at seven drafts, the 22 h2h prompts at 256 forced tokens (`tools/h2h-pr.py`'s method), tok/s for
one request (median / mean) and four streams; sampled at the checkpoint's defaults (temperature 1.0, top-p 0.95,
top-k 20):

| Configuration | greedy, thinking off | four streams | sampled, thinking on | four streams |
|---|---:|---:|---:|---:|
| vLLM (b12x linears and MoE, exact) | 73.9 / 76.6 | 147.1 | 70.3 / 72.7 | 147.9 |
| `exact` | 87.1 / 86.7 | 169.8 | 80.4 / 89.3 | 183.7 |
| `fast` | 118.6 / 122.9 | 226.3 | 121.7 / 124.4 | 234.1 |
| `fast` with relaxed acceptance, top-2 within 1.0 (`SHARAR_ACCEPT_DEFAULT=2:1.0`, `SHARAR_ACCEPT_DEFAULT_SAMPLED=1`, `SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD=2:1.0`) | 141.4 / 143.1 | 261.5 | 135.5 / 140.5 | 260.9 |

Greedy, thinking off, mean and four streams: the 3-bit copies take `exact`'s 86.7 / 169.8 to 111.0 / 187.0, the
int8 state to 114.2 / 213.7 and expert sharing to 121.7 / 228.9.

Paired against `exact` item by item (`evals.suite compare`). Sampled with thinking on (the checkpoint's defaults,
32,768-token cap), GPQA Diamond: `fast` 75.8% against 76.3% (-0.5 points, p 1.0; 11 better, 12 worse), 36 against
34 answers at the cap, 15.1K against 14.6K tokens an item. Greedy with thinking off (GSM8K 1,319 / GPQA 198): the
3-bit copies -0.5 / -2.5 points (p 0.27 / 0.56) and with the int8 state +0.1 / +1.5 (p 1.0 / 0.76); MMLU-Pro (first
1,000) with the 3-bit copies -0.1 points. Thinking on and greedy, the 3-bit copies score GPQA 74.2% against 73.7%
and GSM8K (first 300) 97.3% against 96.3%.

Relaxed acceptance is not a profile default: sampled with thinking on, top-2 within 1.0 costs `fast` GPQA 5.6 points
(p 0.052; nine more answers at the cap), and greedy top-3 within 2.0 cost GPQA 8.6 points (p 0.02). Expert sharing
at keep 4, 16 a request and 48 in all (8% faster than 6 / 20:64) thinks 1.23x longer on GPQA than no sharing
(95% interval 1.07-1.43) and costs 6.6 points (p 0.011), mostly answers cut off at the cap.

This pin has no built-in MTP head. The pinned companion
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` at `449d24140160b8e51871bfc0f0a7338abf640aa1`
uses the parallel DFlash proposer (`dflash2` in Sharar), sharing the target's embeddings and vocabulary head.
All profiles default to seven draft tokens (verified exactly under `exact`); `--spec N` selects 1–7 and
`--spec 0` turns speculation off. Longer drafts and MTP are rejected. The draft keeps its published NVFP4/BF16
weights (`--draft-precision bf16`); the precision name refers to its activations and floating-point matrices.

The five-layer draft uses six target feature taps, block-local grouped convolutions, RoPE and 1,024-token
sliding attention with learned virtual sinks. The two-layer LiLiCorr head scores transitions between eight
candidates per position; sampled proposals carry the actual conditional probabilities to the target verifier.
Draft features and projected KV follow the parallel adapter's request, prefix-cache and SSD lifecycle. Shorter
drafts use the learned slot prefix without resizing parameters. Tensor parallel and image/video
inputs are not implemented for this target. BF16 KV (`SHARAR_KV_FORMAT=sim-bf16`, with SSD off) is available for
numerical parity; packed NVFP4 KV is not enabled.

## Validation

`tests/gpu/test_bolt_kernels.py` checks each packed encoding against dequantized BF16 matmul, including routed
pairs, padded rows, ReLU², CUDA graph replay, b12x plans prepared at load and run from another thread, the fused
MoE and budgeted expert selection at Bolt's geometry. `tests/gpu/test_bolt_model.py` checks all 88 residual outputs
and logits against the independent PyTorch reference, the heterogeneous MoE blocks against Transformers,
chunked prefill, prefix reuse, graph decode and exact Mamba-state restoration from the SSD tier.
`tests/gpu/test_lilicorr.py` checks the published packed correlator against an independent unpacked reference,
trained slot prefixes, graph replay and virtual-sink attention. The Bolt LiLiCorr integration checks generation,
acceptance, target feature taps and continuation through the cache lifecycle.

```bash
SHARAR_RUN_GPU=1 uv run pytest tests/gpu/test_bolt_kernels.py tests/gpu/test_bolt_model.py tests/gpu/test_lilicorr.py tests/gpu/test_bolt_lilicorr.py
uv run python tools/evaluate.py --target bolt --profile exact --suite quick --name bolt
```

The source architecture and checkpoint encodings are specified by NVIDIA's
[model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) and
[pinned config](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/blob/cf2effc5c4b794ee9fb8d34eab3d1d65ce64142f/config.json).
