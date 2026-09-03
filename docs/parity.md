# Parity findings (Phase 1)

Measured on 2026-09-03 against the frozen reference container
`sparklang-gb10-qwen38-nvfp4` (SGLang, image `sparklang-gb10-engine:qwen38-27b`).
Raw evidence lives under `${SHARAR_RESULTS}/parity/`; nothing here is derived
from anything else.

## What was compared

- Prompts (`sharar parity-prompts`): random token IDs of length 8, 128, 2,048
  and 8,192; a 1,024-token natural-text prompt (a paragraph repeated); the four
  bench-harness canaries rendered through the chat template with thinking off
  (23, 28, 46 and 9,642 tokens).
- Reference evidence: per-layer hidden states from the offline SGLang engine
  with a patched tensor dumper (`tools/reference/dump_reference.py`), and
  top-8 logprobs at every position plus the greedy continuation from the
  serving container (`tools/reference/http_oracle.py`).
- Sharar: `sharar parity` runs the same token IDs through `sharar.model.qwen38`
  and compares.

## Result

Sharar reproduces the reference to within the model's own quantization noise
floor on every prompt.

| Prompt | Block cosine with reference inputs (min over layers 0,3,31,47,58,59,63) | Top-1 agreement, confident positions | Next token |
|---|---:|---:|---|
| rand8 | 0.9997 | 3/4 | equal |
| rand128 | 0.9996 | 80/82 (97.6%) | equal |
| rand2048 | 0.9998 | 1117/1157 (96.5%) | equal |
| rand8192 | (no dump) | 4057/4128 (98.3%) | equal |
| text1024 | 0.9994 | 825/977 (84.4%) | equal (6.3 nat margin) |
| canary_exact | 0.9994 | 21/21 | equal (7.1 nat margin) |
| canary_arith | 0.9995 | 24/25 | equal (9.8 nat margin) |
| canary_json | 0.9995 | 39/39 | differs (reference margin 2.4 nats) |
| canary_needle | (no dump) | 9632/9636 (100%) | equal (11.9 nat margin) |

"Confident" means the reference's top-1 leads its top-2 by at least 0.5 nats.

## Why exact hidden-state matching is the wrong criterion

Each block, when fed the reference's own dumped input, matches at cosine
0.9994 or better at every probed depth. The embedding is bit-exact and the
first GDN block matches to 0.14% mean relative error because its input is
bit-identical. Yet the full 64-layer stream diverges on the text prompt to
cosine 0.64 by layer 58, and the MLP fed the same dumped input differs from
the reference by 4.6% mean relative error.

Three measurements explain this:

1. NVFP4 GEMM equivalence. `tools/reference/ab_nvfp4.py` feeds identical
   activations and identical weights to Sharar's wrapper and SGLang's
   `fp4_gemm`: the outputs are bit-identical. Against a torch reference with
   explicit NVFP4 activation quantization Sharar's GEMM is within 0.18%; the
   activation quantization alone costs 6.9% relative error.
2. Sensitivity. `sharar.parity.truth` perturbs the MLP input by one bf16 ulp on
   10% of elements: the W4A4 MLP output moves by 3.9% mean relative error.
   Sharar and the reference are each about 8% from an fp32 dequantized truth
   and 4.6% from each other, with cosine 0.9993. Any two implementations that
   round the norm differently will differ by this much.
3. Chaos through depth. `sharar.parity.chaos` and
   `tools/reference/engine_oracle.py` measure self-agreement. The reference
   against itself, changing only its prefill chunk size from none to 512
   tokens, agrees with the serving container at 84.4% top-1 overall and 86.0%
   at confident positions on text1024, 88.5% / 98.3% on rand2048 and
   85.3% / 98.3% on rand8192, and changes its own greedy continuation of
   rand8192. Its flip rate on text1024 by its own margin is 35% at 2 to 3
   nats and 4.6% above 5 nats. Sharar against a one-ulp-perturbed copy of
   itself: 86.6% / 88.8% on text1024. Sharar against the reference:
   82.6% / 84.4%. Same floor.

The canary_json next-token difference (Sharar predicts a code fence, the
reference predicts `{`, reference margin 2.4 nats) sits inside that floor. It
is still recorded as a quality risk for Phase 2, where the harness's JSON
canary will judge the decoded text.

## Criteria adopted (`sharar parity`)

1. Every probed block, fed the reference's own input, matches at cosine
   at least 0.995.
2. Top-1 agreement at confident positions at least 80% wherever a prompt has
   at least 20 confident positions.
3. The next token equals the reference's whenever the reference's margin is
   at least 5 nats.

## Consequences for later gates

- Serial-output hashes in the controlled matrix must be Sharar's own frozen
  hashes. Hash equality with the reference is not achievable by any second
  implementation of this W4A4 checkpoint, including SGLang itself under a
  different chunk size.
- Quality canaries remain a semantic check (4/4 must pass) and the throughput,
  TTFT and memory thresholds are unchanged.
- Bit-level determinism of Sharar against itself is a separate test
  (Phase 2), and is expected to hold because the kernels are deterministic.
