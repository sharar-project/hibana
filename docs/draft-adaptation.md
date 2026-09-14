# Draft adaptation

Sharar can train its DFlash2 drafter on the requests a machine actually
serves, so accepted tokens per verification round rise on that workload. Only
the drafter changes: the target, its vocabulary head, the verifier and every
engine contract stay fixed, so the gates are held-out tokens per round on the
user's own newest traffic, served throughput, and memory. This page is the
guide; the dated evidence is the [draft training report](archive/studies/gb10-draft-training.md).

## Using it

```bash
./sharar-ctl serve          # serve; every request is recorded under $SHARAR_RESULTS/traces
./sharar-ctl train          # stops serving, trains until told to stop
./sharar-ctl train stop     # final held-out evaluation, promotion, serving relaunches (or reloads if it kept running)
./sharar-ctl draft list     # runs, their held-out gains, the draft in use
```

`train` continues from the current promoted checkpoint when there is one
(at a smaller learning rate, 3e-5 instead of 1e-4) and trains on every recorded
request, holding out the newest ones until they cover about 20K output tokens
(at least four requests, at most a fifth of them). It evaluates every 100
steps and once more on `stop`, and promotes only when held-out tokens per
round improve and, when a regression set is configured, that set does not
drop by more than 0.5% below its pre-training score; promotion updates
`$SHARAR_RESULTS/draft-runs/current`, which `serve` follows. `serve --pinned`
forces the publisher's draft, `serve --draft DIR` any checkpoint, `draft use
RUN` another run, `draft pinned` removes the link; `use`, `pinned` and `draft
reload` swap the draft inside a running server (`POST /reload_draft`) in about
a second, without a restart: the checkpoint's weights are copied into the
serving tensors between two verification rounds, so captured graphs, rings and
the prefix cache stay valid. A reload refuses a checkpoint that changes the
architecture or the frozen context weights. Training never shares the
GPU with serving: `train` refuses while a server runs on this machine, and
`stop` waits for the final evaluation (up to 20 minutes by default) before
relaunching. The trainer loads only the draft, the target's embeddings and its
NVFP4 head, about 40 GiB in total.

Underneath, `sharar-ctl` drives `run-sharar-train.sh`, which runs
`sharar train --steps 0 --continue-from /runs/current --promote-to /runs/current`
in the runtime container; that command also takes explicit `--steps`,
`--draft`, `--lr`, `--continue-lr`, `--holdout-tokens`, `--train-params`,
`--kl-alpha`, `--on-policy`, `--regression-traces` and the other
hyperparameters for controlled runs. Launcher variables: `SHARAR_TRAIN_LR`
(1e-4), `SHARAR_TRAIN_CONTINUE_LR` (3e-5, used when a promoted checkpoint
exists), `SHARAR_TRAIN_PARAMS` (which weights to train), `SHARAR_TRAIN_KL_ALPHA`
(1.0), `SHARAR_TRAIN_ON_POLICY` (0), `SHARAR_TRAIN_REGRESSION` (a trace
directory used as the regression set, default `$SHARAR_RESULTS/draft-regression`
when it exists), `SHARAR_TRAIN_HOLDOUT_TOKENS` (20000), `SHARAR_TRAIN_EVAL_EVERY`
(100), `SHARAR_TRAIN_STOP_GRACE_S` (1200), `SHARAR_DRAFT_RUNS`.

## What is recorded

Each finished request is one JSON line: prompt and output token IDs, sampling
parameters, the cached prefix length and every verification round (accepted
count, the draft's proposals, the target's argmax rows). DFlash2 requests also
store the projected context features the drafter attended to, copied from the
proposer's ring as they are written: the last 2,048 prompt positions and every
committed output position, 10 KiB per row; and the target's top-16 ids and
logits for every verified output position (96 bytes per row), read from the
verify graph's logits only while a traced request is in the batch. No text or client metadata is
stored, nothing leaves the machine, and `"trace": false` opts a request out.
The directory is capped (32 GiB by default, about 1,300 requests of 2K-token
traffic); the oldest trace file and its features go first, so the cap is the
training window. Operational details are in [operations](operations.md#request-traces).

Features are recorded rather than recomputed because the target's prefill
numerics depend on sequence length and chunking (see
[performance](performance.md#limits)); features computed later never match
what the drafter saw. Recording them also removes the target from training.

## How training works

`sharar/train/` holds a plain-torch differentiable copy of the serving draft
forward over the same weight dictionary (`draft.py`), the recorded-feature
samples (`data.py`), the block objective (`loss.py`) and the loop
(`trainer.py`). The objective is SpecForge's DFlash2 objective: cross-entropy
at block positions 1..15 with weights `exp(-(k-1)/7)` over response positions,
the selector trained as a classifier over the head's strict top-16 with the
true predecessor, one shared denominator; plus, when the traces carry the
target's top-16, a KL from the target's distribution over those ids to the
draft's distribution over the same ids (`--kl-alpha`, soft labels). Blocks
are 16 wide, matching serving's k=15. The default `--train-params attn`
trains the attention projections, norms, convolutions and selector (about
0.44B parameters) and leaves the MLP matrices; `all` trains every non-context
weight (1.74B) for the same acceptance at four times the optimizer cost. The block forward is compiled with `torch.compile` and
processed in chunks of 128 anchors. The vocabulary head runs serving's exact NVFP4 kernel in the
forward and the dequantized BF16 weight in the backward. Trainable weights
keep FP32 masters and are written back into the serving weights in place
after every step. Evaluation replays serving's greedy proposals from each
round's anchor along the held-out outputs and counts accepted prefix plus one
correction, which reproduces the acceptance the server records.

The selector's supervision is teacher-forced by default: the true previous
token is the predecessor and the true next token the label. `--on-policy`
instead trains the selector on the rounds the server actually ran: the
drafter's own proposals are the predecessors and the target's argmax after
each proposed prefix (recorded in every round) is the label, so the selector
learns to rank candidates in the context it sees at serving time. The unary
and KL terms are unchanged.

Two deliberate choices:

- The feature projection, per-layer K/V projections and K norms are frozen.
  They define the cached draft context, so a promoted draft stays compatible
  with retained rings, cold-tier payloads and captured graphs, and the traces
  recorded under one draft remain valid training data for the next.
- The attention block is bidirectional, as the published checkpoint declares
  (`is_causal: false`) and as both the publisher's model and Sharar's serving
  plan attend. SpecForge's sliding-window mask is causal inside the block; the
  reference harness records that difference explicitly.

`bench/dflash2_reference.py` checks the forward against the serving proposer
and the publisher's reference model, and the objective and selector against
SpecForge's code; `tests/gpu/test_train_forward.py` covers the head, the
forward and one optimizer step on the real checkpoint. `bench/trace_corpus.py`
generates proxy traffic (ShareGPT or coding requests over a repository) and
`bench/draft_ab.py` replays a trace directory's held-out prompts at batch one
to compare drafts on throughput and tokens per round.

## Results and limits

On a repository-coding proxy workload: a 300-step session raised held-out
tokens per round 5.367 to 5.739 and served throughput 46.3 to 52.3 tokens/s
(+12.9%); a 1,000-request serve-then-train loop raised acceptance while
serving from 27.5% to 31.5% of proposals and throughput by 12.4%. On diverse
ShareGPT traffic the gain was 1.4%. The proxy has only about 150 distinct
prompts, and a second session over the same pool at the first-session rate
regressed an unseen-prompt test set by 1% while the trainer's own held-out
slice stayed flat; at the continuation rate (3e-5) the same session improved
the held-out slice, a fixed ShareGPT regression set and the test set together,
which is why continuation uses the smaller rate and a 0.5% regression
tolerance. On-policy selector supervision scored within seed spread of the
teacher-forced default and is optional.

A different drafter can change greedy continuations through the target's
own shape sensitivity: round boundaries move, the same token is computed in a
different verification row, and a near-tie flips. Each draft is deterministic
run to run; the sensitivity belongs to the target numerics track.

Not in scope here: target training or quantization, changing k, tree
verification and sampling kernels. Traces are used for drafter training and
its evaluation only.
