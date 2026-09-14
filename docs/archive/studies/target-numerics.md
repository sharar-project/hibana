# Target prefill numerics: where the shape dependence comes from

Date: 2026-09-14. Evidence: `reviews/2026-09-11-draft-traces/numerics/` (phase
scripts and JSON) in the results tree. The M-invariant kernels described
below were prototyped on branch `feat/stable-numerics` (flag
`SHARAR_STABLE_NUMERICS=1`) and are not merged; main's serving numerics are
unchanged.

## The property

Prefilling a fixed prefix as part of a longer prefill changes its hidden
states: for a 26-token prefix, every prefill length from 27 to 4096 tokens
differs from the 26-token prefill (final-norm relative RMS 0.09 to 0.13 over
the prefix, prefix argmax agreement 96 to 100%). Repeating a length is
bit-exact. Two prompts, same picture.

## Bisect

Layer-level: run the prefix alone and inside a longer prefill, compare the
post-block stream of the prefix rows layer by layer. Op-level: inside the
longer forward, run every kernel a second time on the prefix rows only, with
byte-identical inputs, and compare its own output. This isolates kernels
whose arithmetic depends on the row count M from downstream amplification.

| Kernel | Backend | Depends on M? |
|---|---|---|
| GDN b/a projection (BF16 `x @ W.T`, N=96, K=5120), 48 layers | cuBLAS | Yes: six algorithm classes below 4096 rows (boundaries 17, 115, 425, 897, 1697, 2625) and rounding flips in 20 to 48 layers at every tested length pair |
| Output projections 5120x6144 (GDN out, attention o), 64 layers | cuBLASLt `_scaled_mm` FP8 | Yes, one window: M in 27..32 uses a different algorithm, every other M from 8 to 4096 agrees |
| FlashInfer ragged prefill attention | FlashInfer | By sequence length: schedule classes at 177, 209, 465, 497, 593, 721 tokens; earlier rows rarely change (15 of 16 layers at 26 vs 1000 tokens, 5 layers at 16 vs 17, none at 26 vs 200) |
| Gated RMS norm (`rms_norm_gated`) | Triton | Rows-per-block heuristic from M; one layer flipped at 26 vs 200 and 26 vs 1000 |
| Input/z/QKV FP8 GEMMs, NVFP4 MLP GEMMs, both activation quantizers, fused norms, QK norm and rotary, conv, GDN chunk kernel | | No difference at any tested M |

The per-kernel differences are single BF16 rounding flips (an FP32
accumulation order change of one part in ten million shows only when a value
sits on a rounding boundary), which the 64-layer NVFP4 target then amplifies
to the 10% level. That is why a first-eight-rows hash over M found only the
27..32 window: the flips are rare per row and need the whole prefix over all
layers to show.

## Fix prototype: `SHARAR_STABLE_NUMERICS=1` (branch `feat/stable-numerics`, not merged)

- b/a projection: a Triton kernel with fixed [16, 64] x [64, 128] tiles, a
  fixed split of K into 16 ranges and a fixed-order reduction of the partial
  sums (`sharar/kernels/ba_proj.py`), so a row's arithmetic is the same for
  every M. Used in prefill, verify and decode, so all three phases agree on
  these gates.
- Output projections: zero-row padding to 33 rows for M in 27..32
  (`sharar/kernels/stable.py`), which keeps cuBLASLt on its wide algorithm.
- Gated RMS norm: constant rows per block.
- Attention: unchanged; its length classes are documented above.

Result, prefix rows compared bit for bit against the prefix-only prefill:

| Prefix | Prefill lengths tested | Bit-exact (flag off) | Bit-exact (flag on) |
|---:|---:|---:|---:|
| 16 | 34 (17 to 4096) | 0 | 34 (prompt 1); prompt 2: 0, one attention flip at T=16 in layer 11 |
| 26 | 34 (27 to 4096) | 0 | 34 / 34 on both prompts |
| 100 | 24 (128 to 4096) | 0 | 24 / 24 on both prompts |

Prefills of 16 tokens or fewer take a different FlashInfer path; the
remaining dependence is that one, and it only concerns prefills that short.

Prefill cost (stand-alone `Qwen38.prefill`, median of 20):

| Tokens | 16 | 27 | 64 | 200 | 512 | 1000 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| First kernel (one program per 16 rows, serial K) | +4.4% | +4.4% | +4.0% | +2.3% | +2.0% | +0.3% | +0.3% | +0.2% |
| Split-K kernel (shipped) | +0.1% | +0.9% | +0.3% | +0.2% | +0.1% | +0.1% | +0.2% | +0.1% |

Served (standard benchmark, FP8 draft, three fixed prompts, flag on vs off):
round time is unchanged (104.7 vs 104.5 ms) and the b/a kernel inside the
decode graphs costs 0.30 ms per 48 layers against cuBLAS's 0.24 ms (at four
rows it is faster, 0.26 vs 0.98 ms). End-to-end throughput moved -18.6% on
the code prompt, +4.8% on design and +8.8% on reasoning, with different
output IDs on all three. That spread is not a kernel cost: with the flag,
layer 0 is identical and layer 2 differs by 0.2% relative RMS, yet by the
final norm the 200-position prefill differs by 45% and one argmax in ten
flips. The NVFP4 target amplifies a one-ulp change in a gate to a different
greedy continuation, and the drafter's acceptance on the new text is a
per-prompt lottery. Three prompts cannot rank the two configurations.

Test-set replay (29 unseen prompts, pinned FP8 draft, batch one): flag off
4.884 tokens per round and 45.4 tokens/s; flag on 4.654 and 43.3, both
-4.7%. With per-prompt tokens per round ranging from 3.6 to 10, the standard
error of that mean is about 2 to 3%, so the drop is suggestive, not
established, and there is no mechanism that would make a correctly rounded
gate systematically worse for the drafter. The kernels stay on their branch:
they are the tool for reproducible measurement (a prefix's features do not
depend on what follows it), and a decision to serve with them needs an
acceptance measurement over a few hundred prompts.

## Consequences

- Recorded draft features and evaluation replays become reproducible across
  prefill lengths; the trace sink's reason for recording features (rather
  than recomputing) still stands for cached-prefix and chunked prefills,
  which run different attention paths.
- Cross-engine greedy parity: the b/a gates were already implicated by the
  fixed-b/a control in the verifier study; a shape-stable b/a removes that
  term but not FlashInfer's or SGLang's own schedule dependence.
- Not covered: chunked prefill (2048-token chunks) and cached-prefix prefill
  (paged path plus merge) use different attention kernels and state
  checkpoints; a prefix prefilled in chunks still differs from the same
  prefix prefilled whole.
