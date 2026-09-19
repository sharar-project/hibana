# Qwen3.8-Flash-Next in Sharar

The second serving target (`--target flashnext`, `sharar/model/flashnext/`).
Checkpoint: `RadixArk/Qwen3.8-Flash-Next-NVFP4` (ModelOpt NVFP4 W4A4 routed
experts, everything else BF16; 135 GB, 206 shards). Quality in the SGLang
reference (v0.5.20, GB10): GSM8K-300 95.3%, HumanEval 94.5% (27B: 95.0 / 90.9).

## Architecture as served

48 hyper-connected decoder layers (4 residual streams of 2,560, rank-320 mixers):
36 Gated DeltaNet layers (16 K / 48 V heads of 128, the 27B's geometry, sigmoid
output gate) and 12 attention layers (24 q / 2 kv heads of 256, quarter rotary,
gated output, sparse indexer with a 2,048-token budget); every layer routes to
10 of 512 NVFP4 experts (intermediate 640) plus a sigmoid-gated shared expert; a
prefix-lookup n-gram embedding (PLE) before layer 1 reads 16 rows of a 320M-row
FP8 table per token (bigram and trigram hashes, 8 heads each) and adds a gated,
causally convolved projection to the streams; a global mixer feeds the BF16 head.
The checkpoint also ships a one-layer MTP head (BF16 experts) and a vision
tower (not loaded).

## Package

| File | Role |
|---|---|
| `config.py` | pinned revision and geometry; `sharar/target.py` builds `FLASHNEXT` from it |
| `reference.py` | plain-torch prefill, exact to the SGLang server at the noise floor (parity tool) |
| `hyper.py` | hyper-connection mix/combine and the grouped Gemma norm (shared with the reference) |
| `linear.py` | `TrunkLinear`: FP8 at load with a dynamic activation scale (`SHARAR_FLASHNEXT_TRUNK=bf16` for parity) |
| `gdn.py`, `attention.py` | the 27B modules with these projections and the sigmoid gate; dense attention |
| `moe.py` | FlashInfer `cutlass_fused_moe` on the checkpoint's packing (fc1 rows are `[up; gate]`) |
| `ple.py` | table as numpy memory maps of the 128 shards; hash, gather, gate, convolution; window state in `StatePool.aux` |
| `model.py` | `FlashNext`: the executor's model contract; `norm` is the hyper stream, `logits()` applies the mixer |
| `mtp.py` | `FlashNextMtpHead`: the checkpoint's MTP layer on the `MtpHead` surface for the `mtp` proposer |

Engine extensions: `Target.family`/`cache_env` and optional family fields;
`StatePool.register_aux` (per-slot state cleared, copied and parked with the
GDN states; verify windows committed like the conv windows);
`ForwardContext.ple_history` / `ple_rows` (the executor gathers table rows on
the host for graph-captured decode and verify); `Executor.register_state`.

## Conventions that cost a day

- The GDN output gate is `sigmoid(z)` (`output_gate_type`), not `silu(z)`.
- The fused MoE kernel wants the expert matrix as `[up rows; gate rows]`.
- Activations are quantized once per layer with the loudest expert's calibrated
  scale (the reference must model it that way to match the server).
- `norm.weight` of the GDN is a plain weight (~1); every other norm is Gemma (1 + w).
- `in_proj_qkv` is `[q | k | v]` head-first; `q_proj` is per-head `[q | gate]`.
- The PLE history before a sequence is EOS; a trigram falls back to EOS when an
  EOS lies inside its window. The convolution (kernel 4, dilation 3) needs the
  nine previous gated rows per sequence.

## Numbers (2026-09-19, GB10, FP8 trunk unless noted)

| Measurement | Value |
|---|---|
| Reference vs SGLang, 5 prompts (18 to 512 tokens) | argmax 91 to 100%, confident positions 97 to 100%, next token 5/5 |
| Engine vs SGLang, BF16 trunk | argmax 88 to 100%, confident 99 to 100%, next token 5/5 |
| Engine vs SGLang, FP8 trunk | argmax 82 to 100%, confident 94 to 100%, next token 5/5 |
| Load | 115 s, 69.4 GiB (76.8 with pools and graphs) |
| Bytes per decode token | 6.33 GB (26 ms at 239 GB/s) |
| Plain decode, batch 1 / 4 | 20.7 tok/s (48.3 ms/step) / 80.3 aggregate (49.8 ms/step); SGLang 16.7 at batch 1 |
| GPU time of one batch-1 decode step (graph replay) | 38.1 ms; about 3,600 kernels per step, so ~12 ms of tiny kernels over the 26 ms byte floor and ~10 ms of graph launch and syncs per step |
| Phase 3 fused kernels + FP8 mixers (2026-09-19 19:00) | batch-1 step 48.0 -> 42.1 to 43.2 ms (23.2 to 23.8 tok/s), batch 4 49.9 -> 42.9 to 43.2 ms (93 tok/s); parity unchanged |
| MTP k=3, batch 1 (`bench-decode --spec 3`, reconstruct strategy) | JSON prompt 22.3 -> 44.4 tok/s (1.99x, 79% accepted, 3.32 tok/round), needle prompt 1.79x; per round draft 11.6 ms + verify 59 ms + accept 5 to 40 ms (the accept commit phase varies by prompt and run: 2 to 36 ms, 376 ms once; unexplained, profile next) |
| MTP k=3, batch 4 | JSON 1.97x (172 tok/s), needle 1.54x (136 tok/s); k=2 is not better at either batch size |

Evidence and launchers: `~/sharar-results/reviews/2026-09-18-flash-next/`
(`parity-*.json`, `components.log`, `decode-*.log`, `run_*.sh`).

## Not done

Sparse-attention indexer (dense is exact only to 2,048 tokens), layer feature
capture (DFlash2-style drafters), W3 trunk and experts, fused hyper-connection
and small-GEMV kernels, per-row FP8 scales for the trunk, family-gate inputs,
accuracy gate, production image. Determinism: cold prefills and eager decode
are bit-identical to each other; a prefix-cache-hit continuation diverges from
the cold one (token 9 on the 512-token code prompt), consistent with the FP8
cached-prefix attention the 27B also has, which this model's routing amplifies
into a token flip. Profile (12 eager steps, graphs 50 ms/step): fused-MoE GEMMs
15.2 ms per step with the default tactic hint (the kernel screen got 8.1 ms
with the hint set to the token count, now passed per call), FP8 trunk GEMVs
14.1 ms (at bandwidth), BF16 hyper-connection mixers 5.7 ms, elementwise and
norm kernels 5 to 8 ms, dynamic quantization 0.8 ms, GDN recurrence 0.6 ms.
