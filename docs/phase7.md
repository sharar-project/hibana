# Phase 7 notes: speculative decoding with the MTP head

## What exists

- `sharar/model/mtp.py`: the checkpoint's MTP head as the draft model. For a
  token x_i and the target's post-final-norm hidden state h_{i-1}:
  `d = final_norm(layer(fc(cat(norm_e(embed(x_i)), norm_h(h_{i-1})))))`, then
  the target's NVFP4 lm_head; one bf16 full-attention decoder layer with its
  own bf16 paged KV cache addressed by the target's page ids. About 0.85 GB
  of weights plus 1.3 GB of draft KV at the default pool size.
- `sharar/engine/spec.py`: chain drafting (k tokens; step 1 ingests the rows
  the draft has not seen, later steps feed the draft's own hidden back),
  one verify forward per round over [x, y_1..y_k] per request (attention over
  the prefix plus k+1 fresh pages, GDN in a "verify" mode that starts from
  the committed state, leaves it untouched and records the state after every
  token; the conv window is computed in torch), greedy acceptance (a draft is
  accepted only where the target's own argmax equals it), commit of the
  intermediate SSM state and conv window after the accepted prefix, pages
  kept for accepted tokens and freed otherwise, and the bonus token from the
  target's own choice at the first mismatch.
- Executor: `enable_spec(k)`; greedy requests decode through spec rounds,
  sampled requests through the plain step; prefill chunks also feed the
  draft (draft prefill of the prompt). `sharar serve --spec K`,
  `SHARAR_SPEC` for `run-sharar.sh`, `sharar chat --spec K`,
  `sharar canaries --spec K`, `sharar bench-decode --spec K`.
- `sharar.parity.mtp_check`: draft quality on real prompts. The draft's top-1
  equals the target's top-1 84.5% of the time on the text prompt and 98% on
  the needle prompt.

## Results (`sharar bench-decode --spec K`, 128 tokens, dev container)

| Prompt | Batch | Plain tok/s | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|---:|
| text1024 | 1 | 12.2 | 21.6 (1.77x) | 23.5 (1.94x) | 24.1 (2.00x) |
| text1024 | 4 | 44.4 | 79.3 (1.79x) | 86.4 (1.94x) | 72.3 (1.62x) |
| canary_needle | 1 | 11.8 | 22.8 (1.93x) | 28.4 (2.39x) | |
| canary_needle | 4 | 42.4 | 78.7 (1.86x) | 94.4 (2.25x) | |
| canary_json | 1 | 12.1 | 25.3 (2.09x) | 29.5 (2.44x) | |
| canary_json | 4 | 44.6 | 86.5 (1.94x) | 98.6 (2.19x) | |
| rand128 | 1 | 12.1 | 22.8 (1.88x) | 26.6 (2.20x) | |
| rand128 | 4 | 44.6 | 78.1 (1.75x) | 88.1 (1.97x) | |

At k=3: acceptance 64 to 100% per prompt, 2.9 to 4.0 tokens per round at
batch 1, draft 26 ms and verify 97 ms per round (eager, no CUDA graphs in the
spec path yet). k=3 is the default.

Canaries with k=3: 4/4 pass and every reference token matched (exact 5/5,
arithmetic 3/3, JSON 30/30, needle 12/12), at 26 to 32 tok/s.

SPEC_MATRIX_PLACEHOLDER

## Decisions and notes

- Greedy acceptance only. Requests with temperature above zero use the plain
  decode step; rejection sampling for sampled requests is a follow-up.
- Exactness is with respect to the target's argmax at each verified
  position. The verify forward rounds differently from the batch-1 decode
  graph (GEMMs at M=4 versus M=1), so free-text continuations diverge from
  plain decoding after a few tokens in the usual noise-floor way; the
  canaries, whose margins are large, match token for token.
- After a prefix-cache hit the draft has no KV for the cached prefix (its
  hidden states are not stored); drafting quality drops for that request
  until its own tokens dominate. Correctness is unaffected. Storing draft
  KV alongside the cache is a Phase 8 item.
- The spec path runs eagerly. Verify costs 97 ms against 83 ms for the
  graphed decode step, and the three draft steps cost 26 ms; capturing both
  into CUDA graphs is the first Phase 9 item and is worth roughly 25% more.
