# Draft adaptation

Sharar can train its DFlash2 drafter on the requests a machine actually
serves, so accepted tokens per verification round rise on that workload. Only
the drafter changes: the target, its vocabulary head, the verifier and every
engine contract stay fixed, so the gates are held-out tokens per round on the
user's own newest traffic, served throughput, and memory. Only the 27B's
DFlash2 drafter trains today.

## Using it

```bash
uv run sharar serve 27b         # serve; every request is recorded under $SHARAR_RESULTS/traces
uv run sharar train             # (after stopping the server) train until Ctrl-C, then evaluate and promote
uv run sharar train --detach    # or as a service: sharar train stop finishes it, sharar logs --train follows it
uv run sharar draft list        # runs, their held-out gains, the draft in use
```

`train` continues from the current promoted checkpoint when there is one (at a smaller learning rate, 3e-5 instead of
1e-4) and trains on every recorded request, holding out the newest ones until they cover about 20K output tokens (at
least four requests, at most a fifth of them). It evaluates every 100 steps and once more when it stops (Ctrl-C,
`sharar train stop`, or `--steps N`), and promotes only when held-out tokens per round improve and, when a regression
set is configured (`--regression-traces`, default `$SHARAR_RESULTS/draft-regression` when it exists), that set does
not drop by more than 0.5% below its pre-training score; promotion updates `$SHARAR_RESULTS/draft-runs/current`, which
`sharar serve` follows. `serve --pinned` forces the publisher's draft, `serve --draft DIR` any checkpoint, `draft use
RUN` another run, `draft pinned` removes the link; `use`, `pinned` and `draft reload` swap the draft inside a running
server (`POST /reload_draft`) in about a second, without a restart: the checkpoint's weights are copied into the
serving tensors between two verification rounds, so captured graphs, rings and the prefix cache stay valid. A reload
refuses a checkpoint that changes the architecture or the frozen context weights. Training never shares the GPU with
serving: `train` stops a detached 27B server and refuses while one runs in the foreground; a detached run's `stop`
waits for the final evaluation (up to 20 minutes, `SHARAR_TRAIN_STOP_GRACE_S`). The trainer loads only the draft, the
target's embeddings and its NVFP4 head, about 40 GiB in total.

**Two Sparks.** Training runs on one Spark, the one where `sharar train` runs, from that Spark's traces. A pair serving
with `--tp 2` records them on rank 0's Spark (where `sharar serve 27b --tp 2` ran), so train there; `train` stops the
pair. The next `serve 27b --tp 2` serves the promoted draft and copies it to the peer, and `draft use | pinned | reload`
work on a running pair: the checkpoint goes to the peer at the same path before the swap, and a pair that restarts
after a failure starts on the current draft.

`sharar train --help` lists the hyperparameters for controlled runs (`--steps`, `--draft`, `--lr`, `--continue-lr`,
`--holdout-tokens`, `--train-params`, `--kl-alpha`, `--on-policy`, `--regression-traces`, `--output`;
`--continue-from none` starts from the pinned draft and `--promote-to none` promotes nothing).

## What is recorded

Requests served under relaxed acceptance record the block offsets of tokens
that were the drafter's pick rather than the target's argmax (`rx`); the
trainer keeps those requests but gives such positions no label weight, and
never holds them out, since replaying acceptance on them would score the
drafter against itself. Corpus and benchmark clients pin exact acceptance.

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
[numerical limits](operations.md#numerical-limits)); features computed later never match
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

`profiling/draft/dflash2_reference.py` checks the forward against the serving proposer
and the publisher's reference model, and the objective and selector against
SpecForge's code; `tests/gpu/test_train_forward.py` covers the head, the
forward and one optimizer step on the real checkpoint. `profiling/draft/trace_corpus.py`
generates proxy traffic (ShareGPT or coding requests over a repository) and
`profiling/draft/draft_ab.py` replays a trace directory's held-out prompts at batch one
to compare drafts on throughput and tokens per round.

## What to expect

From the long-use study (#118: a recorded corpus of code, chat and agent traffic replayed through a dozen
training sessions, then confirmed with served cycles):

- The method is sound: on 20 requests the drafter memorizes them (4.8 to 15.8 tokens per round against a
  maximum of 16), and replayed gains match served ones.
- The gains are modest and specific to the traffic: after eight sessions, code from libraries never trained on
  +4%, chat +2-3%, agent traffic about 0; served throughput about +3% on code and chat. Code from the same
  pool as the training data gains more (+4% to +10% as the data grows from 150 to 1,100 requests). Each session
  adds less than the one before, while the training loss keeps falling.
- The best settings are the defaults: lr 1e-4 (3e-5 when continuing), `attn`, about 1,000 steps. Training
  every weight does not help, and lr 3e-4 hurts.
- Batched and sequential scoring of the same checkpoint differ by up to 0.7% on the GPU, so differences
  under about 1% between checkpoints are noise.

A large speedup needs a stronger base drafter or engine changes, not local fine-tuning. A different drafter
can also change greedy continuations through the target's shape sensitivity (round boundaries move); each draft
is deterministic run to run.

Not in scope here: target training or quantization, changing k, tree verification and sampling kernels.
Traces are used for drafter training and its evaluation only.
