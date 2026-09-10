# Why the GB10 code benchmark is still slower

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

2026-09-08. This investigation repairs FP4 tie rounding, measures target
arithmetic differences on real operands, and profiles the remaining GPU work.
It does **not** establish a wide win over SGLang or matched cross-engine text.
Raw evidence and reproduction scripts are indexed in
[the manifest](gb10-target-path.json).

## The work count matters more than another small fusion

The earlier exact-fusion engine took 33.803 s on code, with 264 verification
rounds. The pinned SGLang comparison took 30.904 s with 205 rounds, on a different
continuation. Sharar's effective round cost was lower: 113.88 versus 130.30 ms.
Those are `(request latency − TTFT) / rounds`, not isolated kernel timings.
The [earlier same-history replay](gb10-acceptance-parity.md) found only a 1.3%
aggregate proposal deficit over 128 SGLang histories. That does not establish
a large acceptance-implementation defect behind the native 59-round gap.

Fresh profiling after the tie repair measures 114.48 ms per native round on
the first eight code rounds, averaging three repeats after one warmup. A
separate four-round trace has 464.69 ms of elapsed scope and 449.89 ms of GPU
activity: **96.82% active**. Matrix multiplication kernel durations account
for approximately **92.65% of GPU activity**. The trace includes SM121 FP8
Tensor Core kernels and SM120 block-scaled FP4 Tensor Core kernels.

Approximate kernel time per profiled round is 51.11 ms for NVFP4 projections,
35.85 ms for FP8 projections, and 17.24 ms for other GEMMs. The remainder of
GPU activity is about 8.27 ms. Profiling overhead and possible interval overlap
prevent treating these as an exact accounting of the HTTP latency. This is
GPU activity evidence, not a physical DRAM bandwidth-counter measurement.

The [hardware/traffic analysis](gb10-next-speedups.md) still applies: the hot
matrices dominate, and another small launch or elementwise optimization cannot
produce a large overall reduction. Improving accepted tokens per traversal
or substantially reducing matrix cost is necessary.

## Retained repair: exact FP4 ties and signed zero

Both NVFP4 quantizers used `a < 0.25` at the zero/0.5 boundary. Consequently,
+0.25 became +0.5 and −0.25 became −0.5. The CUDA reference rounds these ties
to signed zero. This is the round-to-nearest-even rule, not a choice of
matrix multiplication backend. [NVIDIA PTX rounding specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt).

The standalone and fused-normalization quantizers now use the inclusive
boundary and preserve the sign bit for zero. Exact regression tests cover
all E2M1 ties, adjacent BF16 values, signs, four scales, partial row tiles,
and the fused normalization path with both residual-sum modes.

The existing fused-SiLU test also compared scale tensors that aliased the
quantizer's reusable buffer. It now clones the expected scales before the
second call, so the comparison can detect a difference.

This correction changes real model computation: in the target-only 10K probes,
observed layer outputs change within the first few layers, and the reasoning
prompt's first token changes. It affects NVFP4 projections, including the
vocabulary projection used by both target verification and draft selection.
It is a correctness repair, not a claimed kernel speedup.

## A larger arithmetic mismatch: SiLU product precision

On the corrected engine's actual first-layer, 2,048-token code operands,
the standalone MLP input quantizer matches the CUDA reference exactly:
5,242,880 packed bytes and 655,360 scale bytes agree.

The fused SiLU/product quantizer does not. It keeps the activation product in
FP32 before quantization. Compared with the CUDA fused reference,
**143,949 of 17,825,792 packed bytes differ (0.81%)**, and
**39,217 of 2,228,224 scale bytes differ (1.76%)**. Computing the product in
FP32, rounding it to BF16, then using CUDA quantization reproduces the native
fused result exactly on these operands.

A diagnostic Triton candidate uses precise SiLU arithmetic and restores that
BF16 product boundary. It also matches all those real operand bytes and
scales exactly, and passes 16 equation/graph regression cases. This is a
specific primitive result; it does not prove full-model SGLang parity.

The candidate passes all four quality canaries but changes all three long
continuations. Code gets faster while the other cases get slower. It remains
outside the engine: **a partial arithmetic alignment is not promoted as a
general performance improvement**. Completing target parity needs direct
comparison of corresponding SGLang and Sharar layer intermediates, including
the remaining normalization, weight-loading and quantization differences.

## Fresh audited AIPerf results

All profiles use exactly 10,000 input IDs and 1,000 generated IDs, concurrency
one, temperature zero, seed 42, no stops, ignored EOS, and a verified empty
prefix cache before each request. Target/draft revisions, FP8 target KV,
FP32 recurrent state, BF16 draft, verification width 16, and 2,048-token
prefill chunks remain pinned. Each cell has two measured requests and one
excluded full-generation warmup, with one fresh model load per profile.

| Profile | Code seconds / rounds | Design seconds / rounds | Reasoning seconds / rounds |
|---|---:|---:|---:|
| Tie repair; retained engine | 34.427 / 269 | 28.127 / 214 | 22.304 / 163 |
| Also fuse QKV/Z; diagnostic only | 34.228 / 269 | 28.041 / 214 | 22.223 / 163 |
| Restore BF16 SiLU product; diagnostic only | 31.236 / 242 | 30.915 / 239 | 27.168 / 206 |

QKV/Z fusion preserves the corrected baseline's complete IDs and text, but
saves only **0.58% / 0.31% / 0.36%**, below its frozen 3% primary gate. The
BF16-product candidate changes the work: code latency falls 9.27%, design
rises 9.91%, and reasoning rises 21.80% versus the tie-corrected baseline.
Neither candidate is added to serving execution.

All **27 requests** pass raw-response reauditing: exact input IDs, output IDs,
usage counts, zero cache hits, length finish and stream completion. The 18
measured requests exclude nine warmups. Each profile repeats deterministically;
all 12 quality canaries pass. **Cross-engine output equality still fails on
all three prompts.** No new SGLang HTTP benchmark was run in this investigation.
The earlier 30.904 s code result is a historical reference: the retained engine
is 11.40% longer, while the unpromoted BF16-product candidate is 1.08% longer.

## Why the projection screen overestimated its benefit

An initial separate-kernel screen suggested 0.521 ms for split QKV/Z versus
0.445 ms fused. Measuring the actual sequential split pair together, with
interleaved ABBA samples and L2 eviction, instead gives 0.4574 versus 0.4513 ms
in eager mode and 0.4576 versus 0.4513 ms in CUDA graphs. This small advantage
agrees with the full-request result. CUDA graphs alone do not explain the
initial discrepancy; summing separately measured component medians was not a
reliable estimate of the actual sequence.

The fresh native round diagnostic is similarly small: 114.48 versus 114.37 ms.
Fusion reduces projection launches but reads the same weight bytes. It and
its strided convolution experiment remain in the evidence directory.

## Other checks and delivery

Nine target-only arithmetic profiles do not recover SGLang's first-token
scores. Switching FP8 library dispatch or fusing QKV/Z leaves all three
prefill logit vectors unchanged in that screen. Changing FP4 backend or
quantization can change them substantially. Reference-like residual sums,
FP16 weight loading, and the combined variant still do not establish parity.

Source inspection finds SGLang enables fused SiLU/FP4 in its 48 GDN MLPs but
not its 16 full-attention MLPs. An additional offline diagnostic using the
pinned image's actual activation kernel finds **identical tested prefill
logits** with these two MLP paths. That difference is not the missing cause
on these inputs. The initial host-network diagnostic was rejected by automatic
approval review; the approved replacement disabled networking and image pulls,
used a read-only container root, and excluded production state mounts.

Focused GPU validation passes **38 tests**: 22 retained-kernel tests and 16
diagnostic-candidate tests. The eight distinct strided-convolution checks also
preserve outputs, candidate windows and committed state under graph replay.
The CPU regression suite passes **373 tests with eight skips**.
An initial profiling harness with one draft slot failed before timing because
the engine warmup admits multiple requests; the rerun uses the benchmark's
four-slot request capacity. The failed attempt is retained and excluded.

The production container and memory guard are restored after every GPU window.
The production image is unchanged. The retained source changes are the FP4 tie
and signed-zero repair, exact boundary tests, and the scale-buffer test repair.
