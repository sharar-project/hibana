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

`kernels/ggml_bf16.py` decodes tiles into BF16 MMA operands while keeping the resident matrices packed. It
does not quantize activations. NVFP4's global gain multiplies the FP32 accumulator before the output rounds
to BF16. IQ2 routed calls select experts using device ids; invalid ids on padded graph rows produce zero.
The weighted sum is accumulated in FP32 and rounded to BF16 before the latent output projection.

The default BF16 path uses one tensor-core tile per routed pair and output tile, including a padded row tile
for small calls. `SHARAR_BOLT_EXPERTS=fp8` selects `kernels/iq2_moe.py`: GPU grouping, grouped up/down projections,
fused ReLU² and intermediate quantization, and an FP32 weighted combination. The checkpoint's IQ2 bytes
are losslessly retiled at load and replace the original expert buffers. Activations use E4M3 with FP32 scales
per 32 values; the down-projection pairs stay FP32 until the weighted sum rounds to BF16. This changes the
arithmetic. The shared expert, latent, attention, Mamba and vocabulary projections keep the BF16 path.
Temporary scratch belongs to each call and is reused between layers by the graph allocator. SSD cache
identities distinguish the two expert arithmetic modes.

Loaded model allocations are about 21.2 GiB before request pools and graphs. One FP32 Mamba slot costs
162.34 MiB across the 40 recurrent blocks; FP8 KV costs 4 KiB per token across eight attention blocks.
The launcher and ledger budget the pools separately. The checkpoint's 24GB name is not the total runtime budget.

## Profiles and limits

Every profile uses native weights, all nine routed experts, FP32 Mamba state, FP8 KV, up to four active
requests and the SSD tier. `SHARAR_BOLT_EXPERTS=bf16|fp8` selects the routed-expert arithmetic; the profiles
default to `bf16` while the FP8 option is evaluated. There is no calibrated W3 copy or expert pruning.
FP16 Mamba storage is available through the shared runtime but is not a Bolt default.

This pin has no built-in MTP head. The pinned companion
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` at `449d24140160b8e51871bfc0f0a7338abf640aa1`
uses the parallel DFlash proposer (`dflash2` in Sharar), sharing the target's embeddings and vocabulary head.
All profiles default to seven draft tokens with exact target verification; `--spec N` selects 1–7 and
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
pairs, padded rows, ReLU² and CUDA graph replay. `tests/gpu/test_bolt_model.py` checks all 88 residual outputs
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
