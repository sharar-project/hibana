# Qwen3.8-Flash-Next in Sharar

The second serving target (`sharar serve flashnext`, `sharar/model/flashnext/`). Checkpoint:
`RadixArk/Qwen3.8-Flash-Next-NVFP4` (ModelOpt NVFP4 W4A4 routed experts, everything else BF16; 135 GB,
206 shards). Settings: `TARGET_SETTINGS` in `sharar/launch/settings.py`.

## The model as served

48 hyper-connected decoder layers (4 residual streams of 2,560, rank-320 mixers): 36 Gated DeltaNet layers
(16 K / 48 V heads of 128, sigmoid output gate) and 12 attention layers (24 q / 2 kv heads of 256, quarter
rotary, gated output, a sparse-attention indexer with a 2,048-token budget). Every layer routes to 10 of 512
NVFP4 experts (intermediate 640) plus a sigmoid-gated shared expert. A prefix-lookup n-gram embedding (PLE)
before layer 1 reads 16 rows of a 320M-row FP8 table per token (bigram and trigram hashes, 8 heads each) and
adds a gated, causally convolved projection to the streams; a global mixer feeds the BF16 head. The checkpoint
also ships a one-layer MTP head (the served drafter) and a vision tower (not loaded).

Serving formats: the trunk is quantized to FP8 at load (per-tensor, dynamic activation scale), with calibrated
3-bit decode copies in `fast` and `throughput`. The routed experts are 3-bit too when the W3 checkpoint carries
them, as the pinned one does: GPTQ-calibrated on routed activations (#73), read by `kernels/w3_moe.py` in decode
and prefill, and the only resident expert copy (46.1 GiB instead of 63.3; 53 GiB of weights in all). Without
expert entries, with `SHARAR_FLASHNEXT_EXPERTS=nvfp4`, in `exact` and on two Sparks, the experts are the
checkpoint's NVFP4 through FlashInfer's `cutlass_fused_moe` ([calibration](w3-calibration.md)). The PLE table
stays on the NVMe as memory maps.

## Package

| File | Role |
|---|---|
| `config.py` | pinned revision and geometry; `sharar/target.py` builds `FLASHNEXT` from it |
| `reference.py` | plain-torch Gemma norm, NeoX RoPE and QSA token mask (the indexer and the QSA tests) |
| `hyper.py` | hyper-connection mix/combine and the grouped Gemma norm |
| `linear.py` | `TrunkLinear`: FP8 at load (`SHARAR_FLASHNEXT_TRUNK=bf16` for parity) with its 3-bit copy |
| `gdn.py`, `attention.py` | the 27B modules with these projections and the sigmoid gate; attention runs the indexer |
| `indexer.py` | sparse-attention indexer: index keys in page aux rows, group compression, top-512 selection |
| `moe.py` | routed experts: the 3-bit experts through `kernels/w3_moe.py`, or `cutlass_fused_moe` on the checkpoint's NVFP4 packing (fc1 rows are `[up; gate]`); never both resident |
| `ple.py` | the table as memory maps of its 128 shards; hash, gather, gate, convolution; window state in `StatePool.aux` |
| `extras.py` | the side inputs a forward needs (PLE history and rows, indexer bookkeeping) on `ForwardContext.extras` |
| `model.py` | `FlashNext`: the model contract; `norm` is the hyper stream, `logits()` applies the mixer |
| `mtp.py` | `FlashNextMtpHead`: the checkpoint's MTP layer (FP8 experts) for the `mtp` proposer |

## Conventions that cost a day

- The GDN output gate is `sigmoid(z)` (`output_gate_type`), not `silu(z)`.
- The fused MoE kernel wants the expert matrix as `[up rows; gate rows]`.
- Activations are quantized once per layer with the loudest expert's calibrated scale (the reference must
  model it that way to match).
- `norm.weight` of the GDN is a plain weight (~1); every other norm is Gemma (1 + w).
- `in_proj_qkv` is `[q | k | v]` head-first; `q_proj` is per-head `[q | gate]`.
- The PLE history before a sequence is EOS; a trigram falls back to EOS when an EOS lies inside its window.
  The convolution (kernel 4, dilation 3) needs the nine previous gated rows per sequence.
- Cold prefills and eager decode are bit-identical to each other, but a prefix-cache-hit continuation can
  diverge from the cold one (FP8 cached-prefix attention, which the routing amplifies into a token flip).

## Sparse attention past 2,048 tokens

`indexer.py` implements Qwen Sparse Attention: every token's raw index key and each completed group's compressed
key live in its KV page (`KvPool.page_aux`, so they follow pages through the prefix cache, the cold tier and the
session store). A row past the budget scores the visible groups, keeps the top 512 plus its pending tail and
attends over those pages through the vendored sparse kernel (`kernels/qsa.py`). Below the budget the dense path
runs unchanged.

Decode and verification replay CUDA graphs at every context length. Inside the budget the dense graphs run with
graph-safe key maintenance; once any row is past it, a second set of graphs selects on the device
(`kernels/qsa_select.py` scores every visible group through the plan's page table in one Triton kernel and takes
the top 512 of a fixed-width buffer), so no shape depends on the context. `SHARAR_QSA_SPARSE_GRAPHS=0` keeps
rows past the budget eager.

## The MTP draft

- **Depth.** k=4 (the target's settings, cap 8). Deeper drafts win on code and math and lose on
  prose; adaptive depth is #43.
- **Draft-only head.** `SHARAR_DRAFT_HEAD=w3` gives the draft steps a 3-bit copy of the lm_head; its top 8
  candidates (`SHARAR_DRAFT_HEAD_RESCORE`) are rescored with the target's FP8 rows, so the draft picks what the
  full head would except on near-ties. Exact acceptance is unaffected. `SHARAR_DRAFT_VOCAB` restricts the draft
  to the first N lm_head rows (off by default: it costs multilingual text).
- **Fused steps.** Each draft step is one graph replay of 66 kernels (`kernels/flashnext_fused.py`): FP8 GEMVs
  quantize their own input, the attention's q/k norm, RoPE and KV write are one kernel, and the experts' gather,
  SiLU-mul and routed sum are fused. Every fusion is bitwise against the chain it replaced
  (`tests/gpu/test_flashnext_fused.py`). A step is now bandwidth-bound: the 3-bit draft head is about 1.1 of its
  1.9 ms.
- **No host gaps.** The verify plan and indexer rows are prepared before the PLE read, cached PLE rows are read
  without the thread pool (`RWF_NOWAIT`), and each drafted token's PLE rows are read while the next draft step
  runs. Device-resident acceptance (on in `fast`) covers MTP too ([interfaces](interfaces.md#device-resident-acceptance)).
- **Confidence.** `SHARAR_TRACE_CONFIDENCE=1` beside traces records the draft's top-1 probability and top-two
  margin per drafted position ([operations](operations.md#request-traces)).

## Tools

`profiling/flashnext_round.py` splits a speculative round into draft, verify, acceptance and GPU idle
(`--phase-kernels` by kernel; `SHARAR_BENCH_VERIFY_STATE=reconstruct --accept 3:2.0` is the served `fast` round);
`profiling/flashnext_draft_step.py` times one draft step and checks it bitwise across builds;
`profiling/flashnext_moe_w3.py` compares the 3-bit and NVFP4 expert kernels; `profiling/flashnext_profile.py`
breaks decode steps down by kernel; `profiling/flashnext_quant.py` scores a
trunk precision against the BF16 trunk with a rounding-noise control. Flash-Next loads stall when earlier loads
left the model files in the page cache (CUDA on the GB10 sees MemFree): drop them first, as `tools/ci/gb10.sh`
does.
