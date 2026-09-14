# GB10 draft training: traces, reference checks and a first fine-tune

2026-09-12, branch `feat/draft-traces`. Dated evidence for the source and
protocol recorded below; the current guide is [draft adaptation](../../draft-adaptation.md).
Evidence root: `${SHARAR_RESULTS}/reviews/2026-09-11-draft-traces/` (window
scripts, server and client logs, traces, the reference report and the run).

**Decision:** the DFlash2 drafter can be fine-tuned inside Sharar on recorded
request traces with an objective that matches SpecForge's, a forward that
matches the serving proposer and the publisher's reference model, and an
evaluation that reproduces serving acceptance. A first 150-step run on
ShareGPT replays raises held-out tokens per round monotonically from 3.871 to
3.926 (+1.4%). On a narrow workload, coding requests over this repository,
300 steps raise held-out tokens per round from 5.367 to 5.739 (+6.9%) and the
promoted checkpoint served through `--draft` delivers **52.3 versus 46.3
tokens/s (+12.9%)** on the held-out prompts at batch size one. No serving
default, checkpoint or production image changes; `SHARAR_DRAFT` is opt-in.

## What was run

One GB10, image `sharar:qwen38-27b` with the working tree mounted read-only,
production stopped and restored around every window. Target
`RadixArk/Qwen3.8-27B-NVFP4@319f741c`, draft `incoai/Qwen3.8-27B-DFlash2@dedf8df6`,
DFlash2 k=15 with accepted-state reconstruction, FP8 draft for serving windows
and BF16 draft for training.

1. **Trace smoke.** Twelve ShareGPT chat requests (one with `trace: false`, a
   quarter sampled at temperature 0.8) against the source server on port 30002
   with `--traces`. Eleven records, zero drops, projected context features for
   every attended position (5,839 rows for 3,379 prompt and 2,471 output tokens).
2. **Reference harness** (`bench/dflash2_reference.py`) on those traces, with
   the publisher's `model.py` and a SpecForge checkout mounted read-only.
3. **Corpus.** 600 ShareGPT first-turn prompts, thinking off, 768-token budget,
   concurrency four, 10% sampled: 599 traced requests, 82,502 prompt and
   353,879 output tokens, 92,769 rounds at 3.821 tokens per round, 435,756
   feature rows, 4.2 GB.
4. **Fine-tune** (`sharar train`): 533 training requests (317,496 output
   tokens), 59 held out by recency, evaluation on the 24 newest; 150 steps of
   two sequences, 128 anchors each, block 16, gamma 7, selector alpha 1,
   AdamW 5e-5 with 10 warmup steps, clip 1.0, seed 0; 1.74B trainable
   parameters (feature and K/V projections frozen).

## Target prefill numerics depend on shape

The first harness run recomputed features by teacher-forced prefill of prompt
plus output and disagreed with the proposer by 18% relative RMS. Prefilling the
prompt alone reproduces the proposer's ring exactly, so the cause is the
target, not the trainer. For one fixed 26-token prefix:

| Prefill length | Layer 5 | Layer 19 | Layer 33 | Layer 47 | Layer 61 | Final norm |
|---:|---:|---:|---:|---:|---:|---:|
| 26 (baseline) | 0 | 0 | 0 | 0 | 0 | 0 |
| 27 to 32 | 0.011 | 0.044 | 0.086 | 0.264 | 0.292 | 0.320 |
| 40 | 0 | 0.022 | 0.064 | 0.214 | 0.223 | 0.228 |
| 64 to 100 | 0 | 0.012 | 0.039 | 0.164 | 0.154 | 0.151 |
| 128 | 0 | 0.012 | 0.036 | 0.097 | 0.083 | 0.097 |
| 200 to 300 | 0 | 0.015 | 0.041 | 0.164 | 0.157 | 0.153 |

Relative RMS over the 26 prefix positions; a 300-token prefill in 100-token
chunks differs from the whole 300-token prefill by 0.228 at the final norm.
Between 26 and 27 tokens every prefix position moves, including position 0
(8%), and the prefix argmax agrees at 88.5%; against a 300-token prefill the
first positions match exactly and the last eight differ by 18 to 31%, with
92.3% argmax agreement. Repeating the same length is bit-exact. The values are
constant inside length buckets, which points at length-dependent kernel
dispatch rather than accumulation noise. This is outside the training track's
scope and is recorded for the target-numerics work; its consequence here is
that features must be recorded while serving, which the trace sink now does.

## Reference checks

All from `reference/report.json`, on the smoke traces, one process:

| Check | Result | Gate |
|---|---:|---|
| NVFP4 head kernel vs dequantized BF16 weight, real draft hidden states | cosine 0.9972, argmax agreement 94.2% | cosine > 0.99, argmax > 0.9 |
| Training forward vs serving proposer, same ring features | 1.26% relative RMS, cosine 0.99992 | < 2% |
| Recorded trace features vs proposer ring | 0 | < 2% |
| Proposals, teacher-forced on serving's path | 93.3% | at least 80% |
| Publisher `DFlash2DraftModel` vs training forward, same raw features | 1.52% relative RMS, cosine 0.99989 | < 2% |
| Gradients reach every backbone parameter | 56 of 56 | all |
| DFlash objective vs SpecForge naive formula | 2.1021829 vs 2.1021834 | < 1e-4 |
| Selector scores vs SpecForge `score_candidates` | exact | < 1e-4 |
| Selector CE numerator vs SpecForge | exact | < 1e-3 relative |
| Attention mask vs SpecForge | context identical to its sliding mask; block identical to its full mask | as designed |
| Save then reload through the serving loader | max difference 0 | 0 |
| After three training steps, serving proposer vs training forward | 93.3% teacher-forced | at least 80% |

The raw chain agreement of proposals is 40% before and 53% after training:
one near-tie flip at position seven changes every later token of a greedy
lattice walk, so the teacher-forced metric is the meaningful one. The block
mask is bidirectional because the published checkpoint declares
`is_causal: false` and both the publisher's model and Sharar's serving plan
attend that way; SpecForge's sliding-window mask is causal inside the block.

## Fine-tune

Held-out evaluation replays serving's greedy proposals from each round's
anchor on the recorded features and counts the accepted prefix plus one
correction, 3,928 to 3,984 rounds per evaluation:

| Step | Tokens per round | Rounds |
|---:|---:|---:|
| 0 | 3.871 | 3,984 |
| 50 | 3.895 | 3,959 |
| 100 | 3.912 | 3,942 |
| 150 | 3.926 | 3,928 |

Each step took 4.4 s median for about 3,800 labels; evaluation of 24 requests
took about 305 s. The untrained value matches the 3.82 tokens per round the
server recorded on the corpus itself. Minimum available host memory during
training was 63.9 GiB with production stopped; the trainer loads only the
draft, the target's embeddings and its NVFP4 head. Per-step loss is noisy
at two sequences per step (mean cross-entropy 2.77 over the first 25 steps,
2.73 over the last 25); the held-out gain is small because ShareGPT first
turns are diverse by construction, and this run establishes the loop, not a
serving speedup. No throughput claim follows from it.

## Repository-coding workload and served TPS

The user's own traffic did not exist yet, so a proxy with its shape was built:
400 requests asking the target to explain, review, test, document or
summarize one file of this repository (`bench/trace_corpus.py --repo`),
thinking off, 768-token budget, 5% sampled, concurrency four: 399 traced
requests, 655,814 prompt and 266,269 output tokens, 52,885 rounds at 5.053
tokens per round untrained, 768,966 feature rows. Fine-tune: 359 training
requests (238,966 output tokens), 40 held out by recency, evaluation on the
24 newest; 300 steps of two sequences, 128 anchors, learning rate 1e-4,
otherwise as above; 4.9 s per step.

| Step | Held-out tokens per round | Rounds |
|---:|---:|---:|
| 0 | 5.367 | 3,142 |
| 100 | 5.604 | 3,009 |
| 200 | 5.683 | 2,967 |
| 300 | 5.739 | 2,938 |

Mean training cross-entropy fell from 2.08 (first 25 steps) to 1.46 (last 25)
and block accuracy rose from 0.48 to 0.58, so unlike the ShareGPT run the
drafter is learning the workload, not only the target's style.

The step-300 checkpoint was then served (`sharar serve --draft`, FP8 draft
precision, otherwise the measured serving profile) and the 24 held-out prompts
replayed as exact token IDs through `/generate` at batch size one with each
request's recorded output budget, after one excluded warmup, in separate
windows for each draft:

| Draft | Aggregate tokens/s | Median per-request tokens/s | Server tokens per round | Rounds | Proposed |
|---|---:|---:|---:|---:|---:|
| Pinned publisher draft | 46.34 | 44.07 | 4.955 | 3,403 | 51,045 |
| Trained, step 300 | 52.33 | 49.19 | 5.619 | 3,001 | 45,015 |

Both runs emit 16,862 tokens. Two further pairs of windows repeated the
replay with output hashes and then full output IDs: 46.39 / 52.36 and
46.40 / 52.33 tokens/s, the same round counts every time, and each draft's 24
outputs bit-identical across its own runs. Between the two drafts, 7 of 24
outputs are identical and 17 diverge, first at token 20 to 747 of a 768 budget
(median 132); 44.5% of all output tokens precede any divergence. The replayed
outputs also differ from the traced outputs for both drafts, which were
produced at batch four. The verifier commits only the target's argmax, so in
exact arithmetic a drafter cannot change outputs; a different drafter changes
where round boundaries fall, so the same token is computed in a different row
of a verification block, and the target's shape-dependent numerics (above)
then pick a different near-tie. This is the continuation drift the existing
studies report for changed verification widths and batch shapes, not a draft
defect, and it is the same property that makes recomputed features unusable.
A drafter therefore changes outputs only through the target's own
shape sensitivity; fixing that sensitivity is a target-numerics task.

## A 1,000-request serve-train loop

The product loop was then exercised end to end on the same repository-coding
workload: a fixed test set of 30 requests generated once with the pinned draft
(one opted out, 29 replayed), whose file-and-task pairs were excluded from all
later traffic; then four rounds of serve 250 requests with the current draft
(concurrency four, traces recorded), train 150 steps continuing from the
current draft (learning rate 1e-4, evaluation every 75 steps, newest 20K
output tokens held out, promotion on improvement), and measure the promoted
draft on the test set at batch size one.

Acceptance on the traffic being served, from the serving counters of each
block, and the trainer's own held-out slice:

| Block | Draft serving it | Proposals accepted | Throughput at concurrency four | Trainer slice before -> best |
|---:|---|---:|---:|---:|
| 1 | untrained | 27.5% | 129.2 tok/s | 5.368 -> 5.736 |
| 2 | stage 1 | 28.5% | 135.9 tok/s | 5.065 -> 5.069 |
| 3 | stage 2 | 30.2% | 140.7 tok/s | 5.718 -> 5.835 |
| 4 | stage 3 | 31.5% | 145.2 tok/s | 5.838 -> 5.853 (step 75) |

The fixed test set, unseen pairs, batch size one, 19,868 output tokens per
measurement:

| Stage | Trained on | Tokens/s | Tokens per round | Proposals accepted |
|---:|---:|---:|---:|---:|
| 0 (incoai) | 0 | 45.38 | 4.884 | 26.1% |
| 1 | 250 | 46.16 | 4.954 | 26.6% |
| 2 | 500 | 44.76 | 4.798 | 25.5% |
| 3 | 750 | 45.68 | 4.901 | 26.2% |
| 4 | 1,000 | 46.44 | 4.988 | 26.9% |

Serving throughput on the workload rose 12.4% over the loop and acceptance
monotonically. The unseen-pair test set ends 2.3% faster than the untrained
draft but is not monotone: stage 2 fell below the baseline while the
trainer's slice stayed flat, because the repository offers only about 150
distinct file-and-task pairs after the exclusion, so from block 2 on every
block re-teaches pairs the draft already fits, at an unchanged learning rate;
the held-out slice shares those pairs and cannot see the damage to unseen
ones. Two consequences for the trainer: continuation from an adapted draft
should use a smaller learning rate, and promotion should also require no
regression on a fixed general set. A second observation: the recorded
features are 10 KiB per attended position, so the 16 GiB cap used here held
about 750 of the 1,000 requests and the oldest block was pruned during block
4; the trace cap sets the training window, and the default was raised to
32 GiB. On a real workload the pool of distinct requests is far larger than
150, so the stage-2 effect should be weaker, but the fixes above stand.

## Candidate masking to the user's vocabulary

Restricting the drafter's candidates to the vocabulary the workload uses was
proposed as a way to gain acceptance through specialization. The vocabulary
from the loop's 715 training requests has 15,667 tokens and covers 99.4% of
held-out output tokens. Masking the head's logits to it before the top-16 in
the replay evaluation lowered tokens per round in every cell: pinned draft
4.859 to 4.807 on the test set and 5.471 to 5.397 on the held-out slice;
stage-4 draft 5.130 to 5.078 on the test set. Rare tokens do not crowd the
candidate set, so the mask only removes the 0.6% of positions where an
out-of-vocabulary token was right. Not adopted; the remaining benefits would
be draft-head compute (a 3% ceiling) and a 15% cheaper training step.

## Draft precision for trained checkpoints

The same stage-4 checkpoint and the pinned draft on the test set at batch one:

| Draft | Precision | Tokens/s | Tokens per round |
|---|---|---:|---:|
| pinned | BF16 | 42.48 | 4.822 |
| pinned | FP8 | 45.38 | 4.884 |
| stage 4 | BF16 | 44.53 | 5.086 |
| stage 4 | FP8 | 46.48 | 4.988 |

FP8 remains the faster serving profile for trained drafts, but it keeps less
of their acceptance gain (5.086 to 4.988) than the replay evaluation shows in
BF16; the FP8 per-tensor scales are computed at load from weights the profile
was validated on. Recalibrating them for trained weights is a small follow-up.

## Soft labels, parameter subsets, compile: the v2 trainer

Three changes were measured together on a fresh 400-request repository corpus
recorded with the target's top-16 logits per verified row (368 training, 31
held out; test set as above). Each run is 150 steps from the pinned draft.

| Recipe | Trainable | Held-out slice (seed 0 / 1) | Test set (seed 0 / 1) |
|---|---:|---:|---:|
| pinned | 0 | 4.614 | 4.859 |
| CE only, all (previous default) | 1.74B | 4.853 / 4.853 | 5.030 / 5.017 |
| CE + KL, all | 1.74B | 4.847 | 5.048 |
| CE + KL, attention subset | 0.44B | 4.853 / 4.813 | 5.086 / 5.028 |
| CE + KL, attention subset, lr 3e-5 | 0.44B | 4.833 | 5.046 |

Seed-to-seed spread (about 0.8%) equals the spread between recipes, so on
acceptance they are equivalent at this step count; the KL to the target's
top-16 is neutral here and stays on as the default because it costs nothing.
The attention subset (Q/O projections, norms, convolutions, selector; no MLP
matrices) reaches the same acceptance with a quarter of the parameters, about
9 GiB less memory and a faster evaluation, and becomes the default. Served in
FP8 on the test set it gives 46.3 tokens/s and 4.99 tokens per round after one
session, what the all-parameter recipe reached after four sessions of the
1,000-request loop (46.4 / 4.99).

Training step cost: the block forward and backward for 128 anchors of one
sequence take 1.64 s eager at 32 anchors per chunk, 1.32 s at 64, 1.31 s at
128, and 0.93 s with `torch.compile` at 128 (54 s compile warmup, peak 35.7
GiB); compile at chunk 128 is the default. FP8 for trained drafts: per-tensor
and per-row weight scales give the same 2.9 to 3.9% error per matrix on both
the pinned and a trained checkpoint, so the BF16/FP8 acceptance gap is the
format, not calibration; nothing to recalibrate.

Merge gate for this trainer, standard benchmark against main's image: with
tracing off, +0.18% / +0.27% / +0.07% end-to-end TPS (code / design /
reasoning), outputs identical; with tracing on, including the top-k
recording, -0.29% / -0.51% / -0.74%, outputs identical.

## On-policy selector supervision and safer continuation: v3

Two follow-ups, same corpus, 150 steps each, attention subset with KL.

On-policy selector: the traces record every verification round (the drafter's
proposals and the target's argmax after each proposed prefix), so the selector
can be trained on the rows it sees at serving time instead of the
teacher-forced ones. Blocks are anchored at the recorded rounds; the unary and
KL terms are unchanged.

| Recipe (from pinned) | Held-out slice (seed 0 / 1) | Test set (seed 0 / 1) | Step |
|---|---:|---:|---:|
| teacher-forced selector (v2 default) | 4.853 / 4.813 | 5.086 / 5.028 | 2.7 s |
| on-policy selector | 4.837 / 4.846 | 5.087 / 5.104 | 2.3 s |

Means differ by 0.2% on the held-out slice and 0.8% on the test set, inside
the seed spread; on-policy is not the default, it stays available as
`--on-policy`. The step is faster because anchors come from recorded rounds
(one block per round, no random anchor sampling), not because the objective
is cheaper.

Continuation: a second session from the v2 attention checkpoint on the same
corpus, with the ShareGPT held-out slice (5,398 rounds) as a fixed regression
set in the promotion gate.

| Continuation from kl-attn | Held-out slice | Regression set | Test set |
|---|---:|---:|---:|
| base (kl-attn/best) | 4.853 | 3.757 | 5.086 |
| lr 1e-4 | 4.860 | 3.733 (-0.6%) | 5.034 (-1.0%) |
| lr 3e-5 | 4.897 | 3.771 (+0.4%) | 5.091 (+0.1%) |

At the original rate the second session repeats the earlier finding: the
trainer's own held-out slice is flat while the general set and the unseen
prompt pairs regress. At 3e-5 all three improve. The gate at its first
tolerance (1%) let the 1e-4 checkpoint through; the default is now 0.5%, which
blocks it and passes the 3e-5 one. `sharar-ctl train` therefore uses 3e-5
whenever it continues from a promoted checkpoint (`SHARAR_TRAIN_CONTINUE_LR`)
and takes `$SHARAR_RESULTS/draft-regression` as the regression set when that
directory exists; the regression set should be the user's own traffic that
they consider general, held fixed across sessions.

## Hot reload of the served draft

`POST /reload_draft` copies a checkpoint into the serving draft's tensors in
place between two verification rounds (FP8 matrices requantized on the
device), so captured graphs, rings and the prefix cache stay valid; it refuses
a checkpoint that changes the architecture or the frozen context weights.
`sharar-ctl draft use`, `draft pinned` and `draft reload` call it.

Gate, test set at batch one, FP8 draft: a fresh server on the pinned draft,
swap to the v2 attention checkpoint, replay, swap back while a 600-token
request is generating, replay. Every replay is compared token for token with
a cold start of the same draft.

| Step | Swap | Tokens/round | Throughput | Identical to cold start |
|---|---:|---:|---:|---:|
| pinned, fresh | | 4.884 | 45.4 | 29/29 |
| reload trained checkpoint | 2.6 s | 4.987 | 46.4 | 29/29 |
| reload pinned, mid-request | 0.8 s | 4.884 | 45.4 | 29/29 |

The in-flight request finished its 600 tokens without error. A GPU test
reloads the served snapshot and checks every tensor bit for bit in both
precisions.

Two things learned on the way. Packing FP8 on the CPU is not the same as on
the GPU: the first reload packed on the host, and 122,693 bytes and the scale
of one q_proj matrix differed, enough to change 19 of 29 greedy outputs; the
reload now packs on the device as startup does. And replaying the same
prompts against a warm prefix cache changes greedy outputs on its own (the
target's prefill shape sensitivity), so the gate flushes the cache before
each replay; a flushed server reproduces a fresh one exactly.

## Serving path with tracing off

Before merging, the standard benchmark (`sharar benchmark --workload all`,
10K input / 1K output, greedy, seed 42, cold prefix cache, one warmup and
three timed requests per workload, FP8 draft, tracing off) ran against an
image built from `main` and an image built from this branch in separate
windows with frozen inputs:

| Workload | main end-to-end / decode TPS | branch end-to-end / decode TPS | Change | Output IDs |
|---|---:|---:|---:|---|
| code | 38.46 / 45.28 | 38.26 / 45.02 | -0.53% / -0.58% | match |
| design | 30.72 / 34.94 | 30.66 / 34.85 | -0.18% / -0.25% | match |
| reasoning | 46.90 / 57.54 | 46.90 / 57.51 | +0.01% / -0.06% | match |

Outputs are identical and the differences sit inside the repeat spread of
three timed requests. With tracing off the only hot-path change is the
packed acceptance download, which now also carries the target argmax rows
for sampled batches. Raw results: `bench-main/` and `bench-branch/` under the
evidence root.

## Limits

Serving is unchanged: no reload command, no promotion into the launcher, no
collapse check on the 18-family set. ShareGPT replays are not a user's
workload. The GPU tests in `tests/gpu/test_train_forward.py` cover the head,
the forward and one optimizer step against the real checkpoint.
