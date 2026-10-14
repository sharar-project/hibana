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
stays on the NVMe as memory maps; on two Sparks each rank keeps its half of the heads in device memory
([tensor parallel](tensor-parallel.md)). On one Spark a prompt's first prefill chunk starts reading the rows of the rest
of the prompt into the page cache in the background (`PleTable.prefetch`, #178), so its later chunks do not wait on
the NVMe: three fresh 10K-token prompts reach their first token in 4.66 s on average with a cold page cache, against
4.94 s without the prefetch and 4.74 s with every row already cached.

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
attends over those pages through the vendored sparse kernel (`kernels/qsa.py`). Below the budget the dense path
runs unchanged.

Decode and verification replay CUDA graphs at every context length. Inside the budget the dense graphs run with
graph-safe key maintenance; once any row is past it, a second set of graphs selects on the device
(`kernels/qsa_select.py` scores every visible group through the plan's page table in one Triton kernel and takes
the top 512 of a fixed-width buffer), so no shape depends on the context. `SHARAR_QSA_SPARSE_GRAPHS=0` keeps
rows past the budget eager.

## The MTP draft

- **Depth, per request (#43).** Deeper drafts win on code and math and lose on prose and multilingual text, so no
  fixed depth suits both: on the h2h set (one Spark, `fast`, greedy) depth 8 beats 4 by 12-17% on code, math and
  structured output and loses 9-13% on chat, prose and multilingual. `fast` sets the depth per request
  (`engine/depth.AcceptanceDepth`, `SHARAR_SPEC_STEP_COST=0.14`, 0.2 on two Sparks): a moving average of the request's
  accepted drafts per position tested, q, and the depth d up to 8 that maximizes the expected emitted tokens (1 -
  q^(d+1)) / (1 - q) per round cost 1 + 0.14 (d - 1) (measured: 31.5 ms at depth 1, about 4.3 ms per token more). On
  two Sparks a round costs 26.7 ms plus 2.9 ms per token (0.11), but drafted tokens deep in a round are accepted less
  often than the model's single q assumes, and at 0.14 chat and prose drafted too deep there (-12% and -7% against
  depth 4); the profile prices a token at 0.2 on two Sparks, which measured best of 0.14, 0.2 and 0.3. A batch
  verifies the deepest depth its requests want, up to 4 for two or more requests (`SHARAR_SPEC_DEPTHS=2:4,4:4`: four
  streams measured best at 4). The choice is made before drafting: stopping the draft on its own confidence step by
  step (which the confidence predicts well, 95% accepted above 0.9 against 21% at 0.1-0.2) cost 17% in host waits
  between steps. Measured 2026-10-14 against depth 4 on the h2h set, the means of two runs:

  | | median | chat | code | math | multilingual | prose | structured | 4 streams |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | one Spark, greedy | 90.9 -> 105.3 | +6% | +17% | +11% | 0% | +1% | +19% | 158.5 -> 161.7 |
  | one Spark, t=0.7 | 89.0 -> 92.3 | -1% | +9% | +7% | 0% | -2% | +14% | 147.5 -> 147.0 |
  | two Sparks, greedy | 121.6 -> 131.3 | -5% | +17% | +11% | +4% | 0% | +15% | 226.6 -> 222.2 |
  | two Sparks, t=0.7 | 118.4 -> 124.7 | -5% | +9% | +8% | -1% | -1% | +12% | 207.1 -> 204.9 |

  The two-Spark rows use 0.2 (four streams: one run). Chat has two prompts, and depth 4's own two runs differed by 7%
  on it. Under exact acceptance a different depth changes the rows each verification computes, so near-tie tokens can
  round differently: main at a fixed depth 8 and at depth 4 diverge on 5 of 6 prompts (from token 6 to 106), as
  adaptive depth does; each setting repeats its own outputs exactly. Under `fast`'s relaxed acceptance a deeper round
  gives the rule more positions; GSM8K (first 300) and HumanEval stayed within run-to-run noise (pooled -0.2 points
  against a repeat of depth 4, whose two runs themselves differed by 1.3). `exact` and `throughput` keep depth 4.
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
