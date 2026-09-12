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

## Limits

Serving is unchanged: no reload command, no promotion into the launcher, no
collapse check on the 18-family set. ShareGPT replays are not a user's
workload. The GPU tests in `tests/gpu/test_train_forward.py` cover the head,
the forward and one optimizer step against the real checkpoint.
