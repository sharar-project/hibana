# Nemotron 3.5 Bolt

`sharar serve bolt` serves the text backbone of
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` on one Spark, on port 30006.
`nemotron-bolt` is an alias. The checkpoint is pinned in `sharar/model/bolt/config.py` and downloaded by
the normal launcher. The C-RADIO vision encoder and projector are not loaded.

```bash
sharar serve bolt
sharar serve bolt --exact
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

The first implementation uses one tensor-core tile per routed pair and output tile, including a padded row
tile for small calls. It prioritizes a working weight-only baseline; it does not yet group tokens by expert
to reuse weight reads during prefill. The optional FP8-compute GGML experiments remain separate from serving.

Loaded model allocations are about 21.2 GiB before request pools and graphs. One FP32 Mamba slot costs
162.34 MiB across the 40 recurrent blocks; FP8 KV costs 4 KiB per token across eight attention blocks.
The launcher and ledger budget the pools separately. The checkpoint's 24GB name is not the total runtime budget.

## Profiles and limits

`exact`, `fast` and `throughput` currently share the same conservative defaults: native weights, all nine routed
experts, FP32 Mamba state, FP8 KV, up to four active requests and the SSD tier. There is no calibrated W3 copy
or expert pruning. FP16 Mamba storage is available through the shared runtime but is not a Bolt default.

This pin has no built-in MTP head. Speculation defaults to zero and an enabled MTP is rejected before loading.
The companion LiLiCorr/DFlash model needs a separate proposer integration. Tensor parallel and image/video
inputs are not implemented for this target. BF16 KV (`SHARAR_KV_FORMAT=sim-bf16`, with SSD off) is available for
numerical parity; packed NVFP4 KV is not enabled.

## Validation

`tests/gpu/test_bolt_kernels.py` checks each packed encoding against dequantized BF16 matmul, including routed
pairs, padded rows, ReLU² and CUDA graph replay. `tests/gpu/test_bolt_model.py` checks all 88 residual outputs
and logits against the independent PyTorch reference, the heterogeneous MoE blocks against Transformers,
chunked prefill, prefix reuse, graph decode and exact Mamba-state restoration from the SSD tier.

```bash
SHARAR_RUN_GPU=1 uv run pytest tests/gpu/test_bolt_kernels.py tests/gpu/test_bolt_model.py
uv run python tools/evaluate.py --target bolt --profile exact --suite quick --name bolt
```

The source architecture and checkpoint encodings are specified by NVIDIA's
[model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) and
[pinned config](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/blob/cf2effc5c4b794ee9fb8d34eab3d1d65ce64142f/config.json).
