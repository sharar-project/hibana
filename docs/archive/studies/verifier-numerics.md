# Verifier and prefill numerical controls

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

The repeated eight/sixteen-row verifier difference starts in **layer 3's
FlashInfer FA2 attention output**, with identical common queries, keys, values
and preceding layer tensors. Changing the attention reduction geometry removes
the recorded verifier drift; changing linear-projection row shapes does not.
Independent FP64 attention controls support split-KV reduction rounding as the
cause. No committed-history mutation or incorrect KV ownership was found.

The prefill result is separate: fixing b/a gates restores all four prompts'
first-token choices across partitions, but some late tensor/state differences
remain. Neither result establishes a new production numerical profile or a
speedup. Target, executor, proposer, memory and serving-kernel sources remain
unchanged. The failed tree-coverage screen remains failed.

## Verifier evidence

The protocol was frozen at **2026-09-06 23:01:24 UTC**, on source `47cd370`.
It reuses the exact versioned-map cost fixtures and proposal paths from the
[tree study](tree-coverage-study.md), at 8,192 / 16,384 / 31,616 input tokens.
These are known diagnostic cases, not new held-out performance or quality tests.
Each physical sequence verifies eight target inputs: the pending token and
seven draft tokens. The second sequence shares immutable committed prefix KV
and recurrent state, with eight distinct writable candidate KV entries.

Two independently loaded windows run two repetitions of both physical shapes
for seven controls at each context: **168 observed verifier forwards**. Native
eager observation must reproduce the captured graph's normalized hidden rows
and target argmax exactly, and the graph must reproduce the prior study's
recorded target IDs. Every repetition and all 42 corresponding cross-load
trace/result pairs match exactly. Every single-sequence self-control matches
its native trace. Committed KV/SSM/convolution hashes remain unchanged and
release/flush recovers all usable pool entries.

All common-row tensors before layer 3 attention are identical. Its initial
relative RMS difference is 0.31–0.33%, with maximum absolute difference
0.015625. Through the remaining quantized model, this grows to an 11–13%
relative RMS difference in final normalized hidden rows:

| Input tokens | First differing stage | Final norm relative RMS difference | Changed target argmax rows | First-chain emission length, 8 / 16 rows |
|---:|---|---:|---:|---:|
| 8,192 | Layer 3 attention output | 11.94% | 1 | 6 / 5 |
| 16,384 | Layer 3 attention output | 12.53% | 1 | 4 / 4 |
| 31,616 | Layer 3 attention output | 11.17% | 0 | 4 / 4 |

The 16K argmax change follows the first rejection. Near 32K, equal argmax IDs
hide substantial internal drift; output-token agreement alone is insufficient.
Native and common-eight-row vocabulary projections produce identical logits
for each given final hidden tensor, excluding head row shape in these cases.

| Control on the common eight rows of the larger physical shape | Result in all contexts and both loads |
|---|---|
| Recompute b/a at M=8 | Native drift remains |
| Recompute FP8 projections at M=8 | Native drift remains |
| Recompute NVFP4 projections at M=8 with the same packed operands | Native drift remains |
| Combine all three linear controls | Native drift remains |
| Use the already planned single-chain attention wrapper with the same Q/KV | Every recorded stage and final norm matches the single-chain result |
| Combine attention and linear controls | Every recorded stage and final norm matches |

Fixed-operand projection probes at layer 0 and the first divergent layer (3)
also produce identical M=8/M=16 outputs for all tested FP8, NVFP4 and b/a
projections. References use FP32 accumulation with TF32 disabled; the small
b/a projection additionally has an independent CPU FP64 reference. NVFP4
weights are decoded in bounded blocks using tested nibble and scale layouts.
These controls locate this verifier effect without assigning it to the earlier
prefill b/a finding merely because both depend on shape.

## Attention reduction and reference control

A follow-up was frozen at **23:10:21 UTC** after localization. It holds the
first attention layer's BF16 queries and stored FP8 KV values fixed and compares
three plans at physical batches 1, 2 and 4. Every sequence has eight query rows;
extra sequences duplicate the query and use private suffix pages with the same
immutable prefix. The reference implements right-aligned causal grouped-query
attention in FP32 and FP64, without FlashInfer attention kernels.

The first attempt exposed a concrete integration limit: `fixed_split_size=1024`
works in its initial B1/B2 controls, then **B4 planning fails** because the new
batch size exceeds the current graph wrapper's padded batch size. Its five
completed partial records, source and error log are retained. The initial
amendment's overly broad missing-data description is corrected in
`attention-attempt-audit.json`; none of those partial records enter the final
matrix. The installed FlashInfer documentation explicitly warns that fixed
split sizes are not generally compatible with changing-length CUDA graphs.

The retry freezes a separate **non-graph wrapper for fixed splitting only**.
Native and unsplit controls retain graph-enabled wrappers, but alternative
graph capture/replay lifetime is not validated by their eager `run()` calls.
All other operands, resources and cases stay fixed. It completes **54 attention
evaluations**: three contexts, three batch sizes, three plans, two repetitions.
Every repeated output is exact, and committed history remains unchanged.

| Input tokens | Native FP64 relative RMS error across B1/B2/B4 | Fixed 1,024-page split error | Unsplit error |
|---:|---:|---:|---:|
| 8,192 | 0.241–0.248% | 0.242% | 0.170% |
| 16,384 | 0.242–0.252% | 0.242% | 0.168% |
| 31,616 | 0.244–0.249% | 0.245% | 0.168% |

**Fixed-split and unsplit outputs are bit-identical across B1/B2/B4** at every
context; native split outputs are not. FP32 and FP64 reference outputs differ
by only about 2.2e-7 relative RMS. Six independent CPU FP64 checks, covering
the first/last query rows and head groups, agree with the GPU FP64 reference
within 1e-12. The unsplit plan's smaller local error does not establish better
whole-model task quality or acceptable performance. Its BF16 result is also
not universally identical to a single final BF16 rounding of the FP64 oracle.

This supports reduction-order rounding, amplified downstream, rather than a
state-ownership repair. We do not enable fixed splitting or disable split-KV
in serving based on these numerical observations.

## Fixed-b/a prefill control

The original four 128-token prompts run as groups `[4]`, `[1,3]`, `[2,2]`,
`[3,1]` and `[1,1,1,1]`, with two repetitions of native and fixed-b/a execution
in each load: **88 prefill forwards covering 160 request prefills**. All paths
use paged FP8 attention, including singleton partitions, matching the corrected
[earlier trace](agent-context.md). A common M=1 vocabulary head isolates body
drift from output-projection shape.

The fixed control injects each layer/token's b/a values from the native full
four-request batch at the correct partition offset. All injected tensors match
exactly; the full-batch self-control reproduces the native result. This is a
causal clamp, not an implementable serving policy that knows future admissions.

| Partition | Native first-token IDs | Fixed-b/a first-token IDs | Fixed-b/a requests with exact full-batch endpoint state |
|---|---|---|---:|
| `[4]` | 279, 279, 279, 271 | 279, 279, 279, 271 | 4/4 |
| `[1,3]` | 279, 279, 279, 279 | 279, 279, 279, 271 | 3/4 |
| `[2,2]` | 279, 279, 279, 271 | 279, 279, 279, 271 | 3/4 |
| `[3,1]` | 271, 279, 279, 279 | 279, 279, 279, 271 | 4/4 |
| `[1,1,1,1]` | 279, 279, 279, 279 | 279, 279, 279, 271 | 2/4 |

Both loads reproduce this table and their corresponding traces/state hashes
exactly. Without the clamp, all four request endpoint states differ from the
full batch in every split partition, even when first-token IDs agree. Fixing
b/a reduces final-norm relative RMS drift from 84–96% to 0–4.73%. Residual
endpoint differences first appear at layer 59's attention-output projection
for the first request and, in the singleton partition, layer 61 for the second.
The prefill observer records endpoints, so this does not locate the first
differing *interior token* or prove the same cause as the verifier's fully
observed eight-row attention case. Complete prefill batch invariance is not
established.

## Decision and next step

Keep the current production arithmetic. No correctness defect requiring an
engine repair was established. The actionable result is a localized verifier
reduction effect, a demonstrated fixed-split graph constraint, and a measured
b/a contribution to prefill drift with explicit residuals.

The next bounded optimization study should test **graph-safe, stable attention
reduction geometry**. Compare native splitting, a correctly provisioned fixed
geometry and an unsplit control across changing KV lengths and B1/B2/B4, including
shared prefixes, private suffixes and replay after growth. Validate captured
against eager/reference outputs before timing. Measure complete verifier rounds
and completed workload time; these diagnostic forwards provide no TPS estimate.
Keep a shape-stable b/a arithmetic candidate and the residual prefill trace as
separate controls, with independent references and prospective quality checks.
Do not choose arithmetic by a favorable benchmark token or revive the failed
tree rule without its own cost, winner-state, quality and speed gates.

## Reproduction and evidence

The validated image is
`sha256:4a7206b66ead662a1dd2a5de811ef270f4dd7a27926273b2e5b812bbe0bade9b`.
Target and DFlash2 revisions, runtime source hashes, full protocols and all
interventions are recorded under
`$SHARAR_RESULTS/reviews/2026-09-06/verifier-numerics`. Original protocols/source
copies and the failed graph attempt are preserved. Raw tensor traces remain
outside Git; the [compact manifest](verifier-numerics-results.json) indexes the
complete sealed bundle and summaries.

Original numerical windows use 35,000 KV entries, twelve recurrent slots,
eight CPUs and four OMP threads. Each has a 64 GiB container limit, 64 GiB
maximum host-available-memory drop and 16 GiB host floor. Peak measured drops
are 34.07 GiB in both original windows, with at least 79.02 GiB available;
the completed attention follow-up stays within the same budget. The guard
restores the original production image and memory guard and verifies health,
profile and exact chat canary after every window, including the failed attempt.

The final CPU suite passes **338 tests with six GPU-only skips**. Eight new
cases check the FP4 nibble/sign/scale oracle, scale layout across tile boundaries,
trace and emission accounting, and causal/GQA reference geometry and scalar
accumulation. Run `python -m bench.verifier_numerics --inputs INPUTS --output OUT
--window 1` (then 2) in the pinned guarded environment. The retained follow-up
scripts and raw-audit programs reproduce the reference controls. All observer
copies synchronize; none of these runs are performance samples.
