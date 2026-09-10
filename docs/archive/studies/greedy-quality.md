# Greedy text parity and answer quality

> Dated evidence for the source and protocol recorded below. See the [current documentation](../../README.md) for supported behavior and priorities. Historical commands may require the recorded revision.

2026-09-09. The optimized engine passes the current quality canaries, but equal
overall quality with SGLang has not been established. The latest
[AIPerf results](gb10-tps.md) verify exact token counts and a throughput gain;
they also record different full greedy continuations.

## Why greedy outputs can differ

Greedy decoding selects the token with the highest computed score at each step.
Temperature zero removes random sampling. It does not require two implementations
to compute identical scores. Different GPU reduction orders or numerical
precision boundaries can change rounding. If a change reverses the ranking of
two candidate tokens, the next token changes; all later tokens then see a
different context. SGLang documents that temperature-zero inference can vary
with batching and GPU reduction order in its
[deterministic inference guide](https://docs.sglang.io/docs/advanced_features/deterministic_inference).

That explains a possible mechanism, not a completed diagnosis of our remaining
gap. The current benchmark runs at concurrency one. We have not isolated the
operation responsible for the remaining cross-engine or cross-process variation,
nor established that every first mismatch is a harmless close-score decision.

We did find and fix actual arithmetic differences: quantization boundaries,
weight conversion, normalization, rotary precision and prefill KV precision.
The repaired target exactly matches accepted layer outputs, top-10 logits and
committed recurrent states on 65 common-history verification blocks / 307
accepted rows. Independent full generations still differ. See the
[target repair evidence](gb10-target-fix.md) for the diagnostic's scope.

In the latest repeat, all six responses per task within each engine load are
identical, including the warmup. Sharar also reproduces its preceding FP8 load's
complete outputs. The three pinned SGLang loads produce different continuations
from one another. These observations show reference variability; they do not
prove that Sharar has no remaining defect.

## What FP8 drafting should change

The FP8 option compresses selected draft-model matrices; the target checkpoint
remains unchanged. Draft proposals still pass through target verification.
Exact speculative decoding preserves the target distribution when verification
is correct and target evaluation is consistent. For a fixed greedy target and
tie rule, replacing the draft should change acceptance and execution cost,
while preserving the target's chosen tokens. This is the principle behind
[exact speculative decoding](https://arxiv.org/abs/2211.17192).

Changing draft precision can change proposal histories and verification block
boundaries. If target computation varies with those execution paths, the
practical implementation may emit a different continuation. FP8 drafting alone
is therefore not a sufficient explanation or justification for the output gap.
Further diagnosis must distinguish numerical execution differences from a
verification or state-handling error.

## What the quality results establish

Different wording can have equal task quality: for example, both "42" and
"The answer is 42" can answer the same arithmetic question correctly. That is
an illustration, not a characterization of all our measured differences.
A changed token can also alter an answer's meaning or correctness.

The current four canaries check exact instruction following, arithmetic,
strict JSON and long-context retrieval. Both tested Sharar draft precisions
pass those checks. Kernel and lifecycle tests additionally check numerical
boundaries, graph replay and state ownership; they are not a broad evaluation
of generated answers.

The supported conclusion is **no failure on the current quality canaries**.
It is not equal overall quality, full greedy parity, or proof that every
divergence is harmless. The source and tests can be published for review with
FP8 kept opt-in. A broader quality claim needs a representative held-out task
suite with answer correctness and format checks on the current numerical
profile. Production and the BF16 draft default remain unchanged.
