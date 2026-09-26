# Sharar's 3-bit quantization, from the beginning

This guide explains the implementation and local serving profile as of
2026-09-18. **Sharar stores selected weights as 3-bit lookup-table codes,
then expands them inside a GPU kernel for FP8 matrix multiplication.** The
scales bring the actual storage to a little over 3.25 bits per weight.
Calibration chooses better codes; adapting the drafter helps it predict the
resulting target model. These are separate steps.

The underlying ideas are established. Sharar's contribution is a specific
format, CUDA implementation, and serving integration for the GB10 and this
Qwen model. We have evidence that this implementation helps this workload;
we have not established a new general quantization algorithm or priority
over all earlier implementations.

## 1. What are we making smaller?

A language model contains large tables of numbers called **weights**.
Training chooses those numbers. When the model reads your prompt or produces
a token, it repeatedly multiplies intermediate numbers by weights and adds
the results. A token is a piece of text, often part of a word.

For example, one tiny calculation might be:

```text
inputs:    [2, 3]
weights:   [0.50, -0.25]
output:    2 × 0.50 + 3 × (-0.25) = 0.25
```

Real layers do this for thousands of inputs and outputs at once. That is a
**matrix multiplication**. The intermediate inputs and outputs are called
**activations**. Weights are reused across requests; activations depend on
the current request.

The GPU must fetch the weights from memory. Reading billions of numbers
repeatedly can take longer than multiplying them. Smaller representations
can reduce that traffic, provided unpacking them does not consume the gain.

## 2. What does “3-bit” mean?

A bit is either zero or one. Three bits have eight possible patterns:

```text
000  001  010  011  100  101  110  111
```

Instead of storing each weight with a format such as BF16, which uses 16
bits, we store one of eight choices. This is **quantization**: replacing a
number with an approximation from a smaller set.

Fewer choices generally mean more error. In the example above, replacing
`0.50` with `0.45` changes the output from `0.25` to `0.15`. Many small
changes across a model can alter which token it chooses. Smaller storage
alone therefore does not establish acceptable quality.

## 3. Our eight choices: a codebook, plus scales

Sharar's default codebook is:

| 3-bit code | Codebook value |
|---|---:|
| 000 | -3.5 |
| 001 | -2.25 |
| 010 | -1.25 |
| 011 | -0.40625 |
| 100 | 0.40625 |
| 101 | 1.25 |
| 110 | 2.25 |
| 111 | 3.5 |

The values are closer together near zero, where many weights lie. This is
a **nonuniform** codebook. The default is based on Gaussian Lloyd–Max
quantization levels, rescaled and rounded to values representable in FP8
E4M3. The code also supports an equally spaced `uniform` codebook; the
calibrated artifact discussed here uses `lloyd`.

A **scale** stretches or shrinks these eight values. Each group of 32
consecutive weights along a matrix's input dimension shares a scale. There
is also one multiplier for each output row of the weight matrix. Together:

```text
approximate weight = codebook[code] × stored group scale × row multiplier
```

For a concrete example, suppose a group's stored scale is `2.0` and its
row multiplier is `0.05`. Its effective scale is `0.1`. A weight of `0.12`
can use code `101`, meaning `1.25 × 0.1 = 0.125`. The error is `+0.005`.
A weight of `-0.34` can use `000`, meaning `-3.5 × 0.1 = -0.35`.

That example illustrates nearest rounding. Calibration can deliberately
choose a different code to compensate for other weights' errors.

The stored group scale is itself FP8 E4M3: an eight-bit floating-point
format with one sign bit, four exponent bits and three fraction bits. The
row multiplier is FP32, a 32-bit floating-point number. It lets the compact
group scales cover the needed range. The eight default codebook entries
contain no exact zero; a group with zero scale can still reconstruct zeros.

## 4. The exact format and storage cost

For a weight matrix with `N` output rows and `K` input columns:

| Component | Stored representation | Cost |
|---|---|---:|
| Weight choices | Three packed bit planes | 3 bits per weight |
| Group scales | One E4M3 byte per 32 weights | 0.25 bits per weight |
| Row multipliers | One FP32 number per output row | `32 / K` bits per weight |
| Codebook and artifact metadata | Eight values, dimensions, hashes, etc. | Small additional overhead |

The tensor payload therefore costs **`3.25 + 32/K` bits per weight**.
At `K = 5120`, that is `3.25625` bits. Thirty-two codes take 96 bits,
or 12 bytes; their group scale adds one byte. The row multiplier is
amortized across all groups in the row.

Ignoring the small row and metadata overhead, 3.25 bits is about 20% of
BF16's 16 bits, 41% of FP8's 8 bits, and 72% of NVFP4's approximately
4.5 bits including its per-16-weight scales. These are comparisons of
weight representations, **not whole-server memory savings or speedups**.

This is a custom Sharar packed format. “GPTQ” describes how we fit the
numbers; it does not mean these files use a standard third-party GPTQ
checkpoint layout. They are also not GGUF files, native three-bit floating
point numbers, or native INT3 Tensor Core operands.

In code, a `W3Weight` contains `planes`, `scales`, `row_scale`, dimensions,
and a codebook. The loader reads these tensors from safetensors files and
checks their manifest. Current packing requires `N` and `K` to be multiples
of 128. See [the format and reference unpacker](../sharar/kernels/w3.py).

## 5. Why the GPU can compute with this format

A **kernel** is a function executed by many GPU threads. A **Tensor Core**
is GPU hardware that accelerates small matrix multiplications. Our kernel
uses its FP8 operations.

The stored weight codes are arranged as three **bit planes**: one collection
contains the first bit of each code, another the second, and another the
third. Their order matches the small matrix fragments consumed by the
Tensor Core instructions.

During a multiply, the kernel:

1. Reads compact codes and scales from GPU-accessible memory.
2. Reassembles codes with bit operations and looks up their E4M3 bytes in
   registers, the small storage local to GPU threads.
3. Multiplies those FP8 values by FP8 activations using Tensor Cores.
4. Applies the group scales and accumulates partial results in FP32.
5. Applies the remaining row multiplier and writes BF16 output.

It does not expand the entire matrix into a large FP8 copy before each
multiply. Expansion happens as fragments are consumed. Thus **3-bit weight
storage and FP8 computation describe different parts of the same path**.

MLP inputs that arrive as BF16 are also converted to FP8 with a separate
scale for each group of 32 activation values in each token row. Projection
inputs already in FP8 use their existing scales. Activation quantization
has its own errors: one large activation can hide smaller values if too
many share the same scale. This is why the MLP path uses small groups.

The implementation is in [w3_gemm.cu](../sharar/kernels/csrc/w3_gemm.cu).
Memory bandwidth matters, but unpacking instructions, scaling, registers,
cache behavior and the amount of work per launch also affect speed. A
smaller representation does not guarantee a faster kernel at every shape.
The [MLP counter study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-mlp-profile.md) shows how we
measure these limits and reject changes that make the kernel slower.

## 6. Which parts of Sharar actually use it?

**Prefill** processes the prompt. **Decode** produces the continuation.
Sharar retains the original NVFP4/FP8 matrices for larger calls and attaches
additional W3 copies for calls with at most 64 activation rows. Very short
prefill calls can also meet that condition; dispatch is based on shape.

With DFlash2 proposing 15 tokens, the target verifies 16 rows per request.
One, two and four concurrent requests therefore commonly produce 16, 32
and 64 physical rows in the target's matrix multiplications.

The local serving profile covers **all 64 target layers**, with **304
calibrated target projection matrices**: 128 MLP matrices, 144 matrices in
the 48 Gated DeltaNet layers, and 32 matrices in the 16 full-attention
layers. The five-layer drafter has **20 additional W3 matrices**: Q,
output, combined gate/up, and down in each layer. A combined projection
counts as one packed matrix.

The target copies are calibrated offline and loaded before inference. The
drafter is separately trained, then its selected FP8 weights are packed
into W3 by nearest rounding at construction or reload. Its matrices are
not part of the target's GPTQ calibration. We do not quantize the weights
again for every generated token.

The vocabulary output head (`lm_head`) keeps its existing NVFP4 path.
Embeddings, normalization, small recurrent control projections, convolution
weights, draft K/V context projections, KV caches and recurrent states
retain their existing formats. Attention scores and softmax are not 3-bit
operations, and neither are activations. The
[optimization inventory's coverage table](optimizations.md#w3-coverage-and-dispatch)
lists the exact matrix names and dispatch rules. It is therefore more
accurate to say “W3 decode copies” than “the whole model is exactly three
bits.”

Because Sharar keeps the original matrices as well, attaching W3 copies
adds resident storage. Its purpose here is to reduce the bytes fetched by
the selected decode operations, while keeping the larger-call path. The
calibrated target's tensor files occupy about 9.90 GB (decimal); that is
not the server's total memory use.

## 7. Calibration: choosing better approximations

The simplest approach rounds each weight to its closest codebook value.
It considers the weight alone. But some errors matter much more than
others for the inputs the model actually receives.

Calibration runs representative prompts through the model and records
statistics of the activations. The objective is to keep layer outputs
close to the reference outputs on those inputs. In mathematical shorthand,
it tries to reduce `XWᵀ - XW_quantizedᵀ`, where `X` contains example inputs.

Our offline fitter uses the established **GPTQ column-error update**. It
uses correlations between input channels to compensate for rounding error
in columns that have not yet been quantized. This is based on approximate
second-order information, rather than training the whole model again.
The method comes from [GPTQ (Frantar et al., 2022 / ICLR 2023)](https://arxiv.org/abs/2210.17323).

Specifically, Sharar:

1. Uses the pinned block-128 FP8 release as its weight source. It is not
   recovering original full-precision weights from the NVFP4 files.
2. Collects input second-moment matrices, called covariances in the code,
   one layer at a time to control memory use.
3. Tries scale clipping ratios `0.6, 0.7, 0.8, 0.9, 1.0`, weighted by
   activation energy. It rounds scales to the actual E4M3 storage format.
4. Applies the GPTQ updates in 128-column blocks, with damping for numerical
   stability. Groups stay in their original order; serving needs no extra
   activation permutation. The eight codebook values remain fixed.
5. Packs the fitted codes and exact scales without rounding them again.
6. Writes a manifest with revisions, input identity, coverage and hashes.
   Serving validates it before capturing its CUDA graphs.

Starting from a more detailed weight source matters: quantizing weights
that have already been heavily rounded can compound their errors. The FP8
release is still a quantized source; calibration cannot recover information
that source no longer contains.

The current artifact used 64 source-file prompts and 53,702 input tokens.
Held-out drafter files and task-accuracy benchmarks were excluded. This is
a synthetic repository coding corpus, not proof of coverage of all users'
workloads. The [calibration instructions](w3-calibration.md) and
[fitter](../sharar/weights/w3_calibration.py) describe the reproducible path.

## 8. Why adapt the drafter as well?

The **target** is the main model. The smaller **drafter** guesses several
upcoming tokens so the target can check them together. When more guesses
are accepted, the server emits more tokens per expensive target pass.

Quantization and calibration can change the target's token choices and
intermediate features. A drafter adapted to an earlier target may then
guess less well. We collect fresh traces from the selected target and
train the drafter against that behavior, keeping validation and test data
separate. This is actual drafter training, distinct from target calibration.

“Exact acceptance” refers to checking proposals against the selected
target's rule. It does not undo quantization error or restore the original
unquantized model. Relaxed acceptance is a separate serving policy; the
calibration comparisons below explicitly use exact acceptance.

The local deployment selected a step-200 adapted draft and checked it
through the real FP8/W3 serving path. Its BF16 training replay was only a
selection proxy. “Calibrated W3 target” and “adapted draft” are separately
identified artifacts; the 304-matrix target manifest does not calibrate
the draft automatically.

## 9. What did the measurements establish?

Against the preceding **nearest-rounded W3 serving profile**, the calibrated
target produced these results with exact acceptance and thinking disabled:

| Task | Previous W3 | Calibrated W3 |
|---|---:|---:|
| GSM8K | 1,214 / 1,319 (92.04%) | 1,246 / 1,319 (94.47%) |
| HumanEval | 147 / 164 (89.63%) | 150 / 164 (91.46%) |

The GSM8K paired comparison supports an improvement. HumanEval's smaller
change was not statistically conclusive. These are not comparisons against
an unquantized model, and do not establish lossless quantization.

Across 101 task families, calibrated-target decode throughput was 1.0862×
the previous W3 profile, with a 95% interval of [1.0513, 1.1230]. Round cost
was essentially unchanged; more accepted tokens per round explained the
gain. Subsequent draft adaptation improved served throughput on a separate
coding proxy by 3.60% at concurrency one and 4.43% at concurrency four.
Do not add or multiply these results into an unmeasured universal speedup.
See the [calibration and adaptation study](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-calibration-adaptation.md)
for controls, datasets and limitations.

The newer CPU staging fix speeds up loading the artifact, not its
arithmetic. The narrowly targeted QKV kernel change has a separate
controlled batch-four decode gain of about 0.66%; a general HTTP gain was
not established. Those findings are documented in the
[runtime profile](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-runtime-profile.md) and
[QKV qualification](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/studies/w3-qkv-tiling.md).

## 10. How novel is this?

The following precedents rule out calling the broad ideas new:

| Idea | Established work | Relationship to Sharar |
|---|---|---|
| Post-training 3-bit weights and second-order error compensation | [GPTQ, 2022 / ICLR 2023](https://arxiv.org/abs/2210.17323) | We explicitly use its column-error update, adapted to our fixed codebook and scale format. |
| Nonuniform low-bit quantization, including 3-bit LLM weights | [SqueezeLLM, 2023 / ICML 2024](https://arxiv.org/abs/2306.07629) | Codebook quantization is established. Sharar does not implement SqueezeLLM's dense-and-sparse outlier representation. |
| Fast GPU multiplication for lookup-table-quantized weights, including difficult 3-bit packing | [FLUTE, 2024](https://arxiv.org/html/2407.10960v1) | Offline weight rearrangement and fused low-bit lookup kernels already exist. Our fragment layout and FP8 implementation need a direct comparison before claiming novelty. |

The part specific to this repository is the combination: E4M3-compatible
eight-value codes, compact E4M3 scales, a three-plane layout matched to FP8
matrix fragments, register lookup, group-scaled activation support, GB10
shape tuning, calibrated artifact validation, and integration with
speculative decoding and drafter adaptation.

That is useful systems engineering. Whether any individual kernel detail
is a publishable research contribution remains unestablished. A stronger
claim would require a closer implementation-level prior-art comparison,
controlled comparisons with appropriate existing kernels on supported
hardware, and quality and performance results across more models and
workloads. **“A specialized W3 serving implementation with measured local
benefits” is the claim the current evidence supports.**

## 11. Where to look next in the repository

| File | What it explains |
|---|---|
| [w3.py](../sharar/kernels/w3.py) | Codebook, bit packing, reference reconstruction and dispatch |
| [w3_gemm.cu](../sharar/kernels/csrc/w3_gemm.cu) | GPU unpacking, FP8 matrix instructions and scale application |
| [w3_calibration.py](../sharar/weights/w3_calibration.py) | Scale fitting and GPTQ error updates |
| [w3_checkpoint.py](../sharar/weights/w3_checkpoint.py) | Artifact validation and loading |
| [Calibration guide](w3-calibration.md) | How to build and serve a fitted artifact |
| [Draft adaptation](draft-adaptation.md) | How the drafter learns from target traces |
| [Optimization inventory](optimizations.md) | Enabled features and their measured evidence |
