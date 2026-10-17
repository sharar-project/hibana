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
3-bit decode copies in `fast` and `throughput`. The routed experts are 3-bit too when those copies are on and the W3
checkpoint carries them, as the pinned one does: GPTQ-calibrated on routed activations, read by `kernels/w3_moe.py`
in decode and prefill, and the only resident expert copy (46.1 GiB instead of 63.3; 53 GiB of weights in all). On two
Sparks each rank holds half of them, whole experts ([tensor parallel](tensor-parallel.md#flash-next)). In `exact` (no
3-bit copies), without a W3 checkpoint or its expert entries, or with `SHARAR_FLASHNEXT_EXPERTS=nvfp4`, the experts
are the checkpoint's NVFP4 through FlashInfer's `cutlass_fused_moe` ([calibration](w3-calibration.md)). The PLE table
stays on the NVMe as memory maps; on two Sparks each rank keeps its half of the heads in device memory. On one Spark a
prompt's first prefill chunk starts reading the rows of the rest of the prompt into the page cache in the background
(`PleTable.prefetch`), so its later chunks do not wait on the NVMe.

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
key live in its KV page (`KvPool.page_aux`, so they follow pages through the prefix cache and the
SSD tier). A row past the budget scores the visible groups, keeps the top 512 plus its pending tail and
attends over those pages through the vendored sparse kernel (`kernels/qsa/attention.py`). Below the budget the dense path
runs unchanged.

Decode and verification replay CUDA graphs at every context length. Inside the budget the dense graphs run with
graph-safe key maintenance; once any row is past it, a second set of graphs selects on the device
(`kernels/qsa/selection.py` scores every visible group through the plan's page table in one Triton kernel, takes
the top 512 of a fixed-width buffer and assembles the pages in a second kernel), so no shape depends on the context.
`SHARAR_QSA_SPARSE_GRAPHS=0` keeps rows past the budget eager. Eager prefill computes the forward's indexer layout
once (`FlashNextIndexer._plan`) and ranks 256 rows per top-k.

## The MTP draft

- **Depth, per request.** Deeper drafts win on code and math and lose on prose and multilingual text, so no fixed
  depth suits both. `fast` drafts up to 8 (`SHARAR_SPEC=8`) and sets the depth per request
  (`engine/depth.AcceptanceDepth`): a moving average of the request's accepted drafts per position tested, q, and
  the depth d that maximizes the expected emitted tokens (1 - q^(d+1)) / (1 - q) per round cost
  1 + c (d - 1). `SHARAR_SPEC_STEP_COST` is c: 0.14 on one Spark (a round measured 31.5 ms at depth 1 and about
  4.3 ms per token more), 0.2 on two, where 0.14 drafted chat and prose too deep. A batch verifies the deepest depth
  its requests want, up to 4 for two to four requests (`SHARAR_SPEC_DEPTHS=2:4,4:4`). The choice is made before
  drafting, so drafting is not held up. `exact` keeps depth 4. `throughput` drafts 4 and verifies 3 from 9 requests up
  (`SHARAR_SPEC_DEPTHS=16:3,32:3,64:3,128:3`, with 3-bit experts on one Spark, where at 32-128 requests a verified row
  costs about 0.5 ms and depth 3 was the best fixed depth); with NVFP4 experts or on two Sparks only the 16-request
  bucket verifies 3 (`16:3`).
- **Draft-only head.** `SHARAR_DRAFT_HEAD=w3` gives the draft steps a 3-bit copy of the lm_head; its top 8
  candidates (`SHARAR_DRAFT_HEAD_RESCORE`) are rescored with the target's FP8 rows, so the draft picks what the
  full head would except on near-ties. Exact acceptance is unaffected. `SHARAR_DRAFT_VOCAB` restricts the draft
  to the first N lm_head rows (off by default: it costs multilingual text).
- **Fused steps.** Each draft step is one graph replay of 66 kernels (`kernels/flashnext_fused.py`): FP8 GEMVs
  quantize their own input, the attention's q/k norm, RoPE and KV write are one kernel, and the experts' gather,
  SiLU-mul and routed sum are fused. Every fusion is bitwise against the unfused chain
  (`tests/gpu/test_flashnext_fused.py`). A step is bandwidth-bound: the 3-bit draft head is about 1.1 of its 1.9 ms.
- **No host gaps.** The verify plan and indexer rows are prepared before the PLE read, and each drafted token's PLE
  rows are read while the next draft step runs. Device-resident acceptance (on in `fast`) covers MTP too
  ([interfaces](interfaces.md#device-resident-acceptance)). `FlashNextIndexer.static_rows` builds every row's page
  indices on the host and uploads them once, and the PLE gather (`kernels/csrc/ple_gather.cpp`) reads rows from a
  native thread pool outside the GIL: `preadv2(RWF_NOWAIT)` copies the rows the page cache holds, then the others are
  read blocking, one per pool thread, so their NVMe latencies overlap.
- **Confidence.** `SHARAR_TRACE_CONFIDENCE=1` beside traces records the draft's top-1 probability and top-two
  margin per drafted position ([operations](operations.md#request-traces)).

## Tools

The Flash-Next profilers (`profiling/flashnext_*.py`: a speculative round, one draft step, the 3-bit against the NVFP4
expert kernels, decode by kernel, trunk quantization error) are listed in [profiling](../profiling/README.md).
