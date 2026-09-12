# Draft adaptation

Started 2026-09-11. This guide records the design for training the DFlash2
drafter on the requests a user actually sends, so that accepted tokens per
verification round rise on that workload. It is the only track that changes
weights; the pinned NVFP4 target, the shared vocabulary head, the verifier and
every engine contract stay fixed. Verification is exact, so a drafter can only
change speed, never the output. The gates are therefore tokens per round on
held-out traces, TTFT, memory and no regression on the general workload set.

## Why the drafter

The BS1 round is bandwidth-bound: about 107 ms, of which 84 ms is target
matrix work reading 17.6 GB of weights, close to the streaming floor. Only
fewer bytes or more accepted tokens per round change throughput. On the code
workload the drafter emits 4.7 tokens per round while the correct future sits
inside its own top-16 candidates 7.9 deep, so most first rejections are
selection errors on this workload. Lossy target quantization cut the round by
13% but lost more than that in acceptance because the drafter had been trained
against the original target. Aligning the drafter to the served target and to
the user's workload attacks both. Nothing here runs the drafter or target
beyond what serving already does.

## What is fixed and what is not

| Component | Adaptation track |
|---|---|
| Target weights, KV format, recurrent state | Fixed |
| Shared NVFP4 vocabulary head and embeddings | Fixed and frozen in training |
| DFlash2 draft feature projection and per-layer K/V projections | Frozen in the first version so retained draft rings, cold-tier payloads and captured graphs stay valid across a reload |
| DFlash2 noise-path weights: Q/O, MLP, convolution, selector | Trained |
| Verification width k=15, accepted-state reconstruction | Fixed |

## Steps

1. **Request traces (this branch).** `SHARAR_TRACES=/dir` records every
   finished request as one JSON line: prompt and output IDs, sampling
   parameters, cached prefix length and each verification round's accepted
   count, proposals and target argmax rows (sampled rows keep the correction
   token only). The outputs are the target's own tokens, which is exactly the
   target-generated training set the DFlash recipe uses. The per-round rows
   are on-policy labels: the target's answer after the drafter's own mistake.
   Recording is opt-in per server and per request (`"trace": false`), writes
   on a separate thread, rotates files and prunes the oldest above a cap.
   `sharar traces --dir DIR` summarizes a directory. DFlash2 requests also
   record the projected context features the drafter attended to, copied from
   the proposer's ring as they are written. They are the only usable training
   input: the target's prefill numerics depend on sequence length and
   chunking (for one 26-token prefix, deep-layer features move by up to 30%
   and about one prefix argmax in ten flips between a 26- and a 27-token
   prefill), so features recomputed later never match what the drafter saw.
   Recording them also removes the target from training entirely.
2. **SpecForge as the reference, not the pipeline.** SpecForge trains DFlash2
   and warm-starts from the incoai checkpoint, but its Qwen3.8 recipe takes
   target features only from patched SGLang capture servers (plus Mooncake
   when disaggregated), which this box does not run and the repository must
   not depend on. Its objective, selector scoring and mask construction are
   used as the numerical reference for the in-repo trainer instead. The gain
   question is answered by that trainer on held-out traces: if fine-tuning on
   a user's traces shows no tokens-per-round gain, the workload is not
   repetitive enough and the track stops.
3. **Training in Sharar (`sharar train`).** The `sharar/train` package holds a
   plain-torch differentiable copy of the serving draft forward over the same
   weight dictionary (`draft.py`), the recorded-feature samples (`data.py`),
   the block objective (`loss.py`) and the loop (`trainer.py`); `features.py`
   recomputes raw features by target prefill for the reference harness only. The objective is SpecForge's DFlash2 objective:
   cross-entropy at block positions 1..B-1 with weights `exp(-(k-1)/gamma)`
   over response positions, the selector trained as a classifier over the
   head's strict top-16 with the true predecessor, one shared denominator.
   The CPU tests check it against the naive formulas SpecForge's own tests use;
   `bench/dflash2_reference.py` checks the forward against the serving proposer
   and the publisher's reference model and the objective and selector scoring
   against SpecForge's code on the GPU. One deliberate difference: SpecForge's
   sliding-window mask is causal inside the draft block, while the published
   checkpoint declares `is_causal: false` and both the publisher's model and
   Sharar's serving plan attend bidirectionally inside the block, so training
   uses the served mask. Blocks are 16 wide, matching serving's k=15, not the
   publisher's block of 8. The vocabulary head runs serving's exact NVFP4
   kernel in the forward and the dequantized BF16 weight in the backward.
   Trainable weights keep FP32 masters; the serving weights are updated in
   place after every step. The trainer loads only the target's embeddings and
   head (about 6 GB with the draft and optimizer state for 1.7B parameters),
   so it fits beside a serving process in memory; it still runs in the runtime
   container while nothing is served, because the GPU has no spare bandwidth.
   The feature and K/V projections cannot be trained from recorded projected
   features, which matches the reload-compatibility decision above.
4. **Serving and promotion.** The trainer evaluates held-out tokens per
   round by replaying serving's greedy proposals along the newest traces and
   writes `best/` only when that beats the untrained drafter or the previous
   best; every evaluated step is also kept under `step-*/` in the publisher's
   checkpoint layout. `SHARAR_DRAFT=/run/best ./run-sharar.sh` (or
   `sharar serve --draft DIR`) serves a promoted checkpoint, in BF16 or FP8.
   Held-out data is always the user's own newest traces, never a public
   corpus: the drafter is meant to overfit the workload, and its TPS is
   benchmarked on held-out prompts from that same workload. The held-out
   budget is about 20K output tokens (roughly 4,000 rounds, enough to resolve
   a 1% change in a few minutes of evaluation). Still to do: an
   in-place reload between rounds, and a collapse check on the general
   workload set. The base checkpoint is never overwritten.
5. **User-run training.** `./sharar-ctl train` stops serving and trains
   until `./sharar-ctl train stop`, continuing from the current promoted
   checkpoint and holding out the newest requests until they cover about 20K
   output tokens (tokens per round is measured over rounds, so the held-out
   budget is tokens, not a share of requests); `stop` runs a final held-out
   evaluation, promotes on improvement by updating `draft-runs/current`, and
   relaunches serving, which follows that link. The GPU is never shared:
   start refuses while serving runs. A second machine can run the same
   trainer on a synced trace directory (`--keep-serving`) and the link can be
   copied back; shipping traces off the box is a separate explicit opt-in.

## Evidence

The [draft training report](archive/studies/gb10-draft-training.md) records
the reference checks, the target prefill shape dependence, the ShareGPT corpus
and the first fine-tune (held-out tokens per round 3.871 to 3.926 over 150
steps, monotone at every evaluation).

## Running the trainer

```bash
./sharar-ctl serve          # serve and record traces
./sharar-ctl train          # when the machine is free
./sharar-ctl status
./sharar-ctl train stop     # final evaluation, promote, serve again
```

The launcher wraps `sharar train --steps 0 --promote-to /runs/current
--continue-from /runs/current`; the command can also be run by hand inside the
runtime container with explicit `--steps`, `--draft` and hyperparameters.

`--draft DIR` continues from a saved checkpoint; `--train-context` also
trains the feature and K/V projections (a reload then invalidates retained
draft rings). `log.jsonl` records every step and evaluation; `manifest.json`
records data digests, the base evaluation and the best step.
`bench/trace_corpus.py` replays ShareGPT prompts against a tracing server to
build a corpus when no user traces exist yet.

## Budget on one GB10

| Item | Estimate |
|---|---:|
| Trainable draft parameters | about 1.8B |
| Full AdamW state | about 30 GB; far less with adapters |
| One 3K-token sample, 128 anchors | about 3e13 FLOPs, about 0.5 s |
| A 30-minute, 100K-token session, one epoch | about 90 s plus 40 s of feature prefill |

Memory, not compute, is the constraint beside a serving process. The trainer
must respect the memory ledger and the external guard's 16 GiB floor.

## Not in scope

Target training or quantization, changing k, tree verification and sampling
kernels are separate tracks. Traces are never used for anything but drafter
training and its evaluation.
