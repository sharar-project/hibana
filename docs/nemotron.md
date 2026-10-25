# NVIDIA-Nemotron-3.5-Super in Sharar

The third serving target (`sharar serve nemotron`, `sharar/model/nemotron/`), on one Spark. Checkpoint:
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` at `6aae0d6` (NVIDIA's ModelOpt export: NVFP4 routed and shared
experts, FP8 Mamba, attention and lm_head projections, BF16 embeddings, router and latent projections). The text model
and its MTP head are served; the C-RADIO vision tower and its projector are never read. Settings: `TARGET_SETTINGS`
and `nemotron_defaults` in `sharar/launch/settings.py`.

## The model as served

88 pre-norm residual blocks, each one mixer: 40 Mamba-2 (128 heads of 64, 8 B/C groups of state 128, conv kernel 4),
40 latent MoE and 8 attention blocks (32 query and 2 KV heads of 128, no positional encoding, no q/k norm, no output
gate). A latent MoE block routes each token to 22 of 512 experts (sigmoid scores plus a selection-only bias, the plain
sigmoids renormalized, scaled by 5) that work in a 1,024-wide latent space between `fc1_latent` and `fc2_latent`, beside
an NVFP4 shared expert on the full 4,096 hidden; both use ReLU², not SwiGLU. The checkpoint's own MTP layer (an
attention block and an MoE block) is the drafter.

Formats are the checkpoint's: the routed NVFP4 experts through Sharar's small-call kernels for calls of up to 16 rows
(`kernels/nemotron_moe.py`: router, routing, experts, shared expert and `fc2_latent` in four chained launches) and
FlashInfer's b12x MoE beyond (fc1 is `up_proj` alone), FP8 GEMVs for the Mamba and attention projections and the head,
FP32 router logits. 66.4 GiB of weights are resident after the load (78 s from the page cache); the KV cache is small
(8 attention layers, 4 KiB a token in FP8) and a request's recurrent state is 160 MiB (FP32; 80 MiB in FP16).
GPTQ-calibrated 3-bit copies of the trunk (Mamba, attention and lm_head projections) and of the routed experts load
beside the checkpoint with `SHARAR_W3_DECODE=1` and `SHARAR_W3_CHECKPOINT=<dir>` (`kernels/w3_moe.py` for the
experts). `fast` and `throughput` turn them on and serve
`w3-checkpoints/nemotron-current` ([where it comes from](w3-calibration.md#nemotron)); without one they refuse to
start (`exact` serves the checkpoint's NVFP4 experts and FP8 trunk). A checkpoint that carries the routed experts serves
them as the only expert copy, at every call size (the MTP head's load with the drafter).

## Package

| File | Role |
|---|---|
| `config.py` | pinned revision, geometry and the block pattern; `sharar/target.py` builds `NEMOTRON` from it |
| `model.py` | `Nemotron`: the model contract; block indices are the checkpoint's, so the Mamba blocks are the target's recurrent layers and the attention blocks its KV layers |
| `mamba.py`, `kernels/mamba2.py` | the Mamba-2 mixer on the recurrent-layer contract (prefill checkpoints, decode in place, verify by snapshots or reconstruction) |
| `attention.py` | the 27B's attention module without gate, norms and rotary |
| `moe.py` | the latent MoE block: fused router and `fc1_latent` GEMM, the routing kernels, budgeted expert selection |
| `mtp.py` | `NemotronMtpHead`: the checkpoint's MTP layer for the `mtp` proposer |
| `norm.py` | plain RMSNorm (weight, not Gemma's 1 + w) |

## Serving

The profiles (`nemotron_defaults` in `sharar/launch/settings.py`):

| Profile | Settings |
|---|---|
| `exact` | MTP depth 3, exact acceptance, FP32 Mamba state, every token's 22 experts |
| `fast` | both 3-bit copies, relaxed acceptance (top 3 within 2.0 logits, greedy and sampled), device-resident acceptance, 2-bit draft head, depth 5 at one request and 4 at two to four, FP16 Mamba state with stochastic rounding, expert sharing `SHARAR_MOE_KEEP=12` `SHARAR_MOE_BUDGET=48:128` |
| `throughput` | `fast` up to 16 requests, depth 3 past four |

Expert sharing (`moe.py`, `route_budget`): a verify call's rows share the experts they read. Each row is guaranteed
its top `KEEP` experts; the call then reads at most `P` distinct experts per request and `CAP` in all, filled by
summed routing weight, and every row also takes its other top-22 experts that the call reads anyway (free riders).
The KV cache is FP8 only: NVFP4 KV (`SHARAR_KV_FORMAT=nvfp4`) and the `sim-*` screens are not supported yet (the
attention's fused K/V writer serves the FP8 pool), and the launcher and the executor refuse them.

Measured (Sharar, spark-9931; h2h median, `tools/h2h-pr.py`, greedy; tok/s) and GSM8K-300 paired against `exact`:

| Profile | 1 request | 4 streams | GSM8K-300 |
|---|---|---|---|
| `exact` | 45.9 | 79.8 | 97.7% |
| `fast` | 102.5 | 185.2 | 97.7% (+0.0 pp, [-1.6, +1.6]) |

Not supported yet, to revisit:

- NVFP4 KV cache: the attention's fused K/V writer serves the FP8 pool only.
- An int8 (or int4) Mamba-2 state: the window formats and kernels (`kernels/ssm_window.py`) implement the GDN
  recurrence; Mamba-2 needs its own record and replay.
- `throughput`: its 16-request cap follows the 27B's and is unmeasured past 16, and its expert sharing (`fast`'s:
  128 experts a call, about 8 a request at 16) is gated at one to four requests only; 8 / 16 streams ran
  277 / 353 tok/s with it against 190 / 239 without.

## Checkpoint conventions

- The checkpoint is the omni model: every text weight sits under `language_model.` (`language_model.backbone.layers.N`,
  `language_model.mtp.layers.N`), and `config.json` holds the text model under `llm_config`, not `text_config`.
- The end of a turn is `<|im_end|>` (11), the end of text `</s>` (2): `Target.stop_token_ids` is (11, 2), not the
  Qwen ids the other targets share.
- Thinking off renders as `<|im_start|>assistant\n<think></think>` with no newline after it (the snapshot's
  `chat_template.jinja`), and the template opens every chat with a system turn, empty when none is given. Thinking on
  ends the prompt with `<think>\n`, so the reasoning parser starts inside the block as it does for Qwen.
- Router logits must be FP32: a block's selection bias spans under 0.1, so the 22nd and 23rd choices sit a median
  7e-4 to 1.6e-3 apart and BF16 logits change the top-22 set of 9-22% of rows.
- The routing weights come from the plain sigmoid, the bias only steers the choice; `routed_scaling_factor` (5) rides
  in fc2's dequantization scale.
- The MTP head consumes the target's output after `norm_f` and feeds back its output after its own final layernorm.
- Attention has no positional encoding at all; nothing reads `rope_theta`.
