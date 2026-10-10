"""Draft training on the real checkpoint: the differentiable forward matches the
serving proposer, the NVFP4 head backward uses a faithful dequantization, and
one optimizer step updates the serving weights in place."""
import gc

import pytest
import torch
import torch.nn.functional as F

from sharar.engine.contracts import StepInputs
from sharar.engine.sampler import SamplingParams



@pytest.fixture(scope='module')
def runtime():
    from sharar import config
    from sharar.engine.executor import Executor
    from sharar.kernels import ops
    from sharar.model.qwen38 import Qwen38
    from sharar.engine.prompts import PARAGRAPH
    from sharar.tokenizer import SnapshotTokenizer
    from sharar.train.draft import Head, TrainableDraft
    from sharar.weights.safetensors import SnapshotWeights
    gc.collect()
    torch.cuda.empty_cache()
    target = Qwen38(config.snapshot_dir(), log=print)
    serving = Executor(target, num_pages=35000, num_slots=12, use_cache=False)
    serving.enable_spec(15, proposer='dflash2', state_strategy='reconstruct', draft_precision='bf16')
    adapter = serving.spec.proposer
    draft = TrainableDraft(adapter.model, Head(target, SnapshotWeights(config.snapshot_dir())), block=16)
    ids = SnapshotTokenizer(config.snapshot_dir()).encode(PARAGRAPH * 16)[:700]
    assert len(ids) >= 400, len(ids)

    def serving_context(prompt):
        """Prefill through the proposer; return (projected ring rows for the prompt, first sampled token, proposals, hidden)."""
        with torch.inference_mode():
            req = serving.admit(prompt, SamplingParams(max_new_tokens=16, ignore_eos=True))
            serving.prefill(req)
            native, original = {}, adapter.model.forward

            def capture(*args):
                out = original(*args)
                native['hidden'] = out.clone()
                return out
            adapter.model.forward = capture
            inputs = StepInputs(torch.tensor([req.output_ids[-1]], device='cuda'), torch.tensor([req.committed_len], device='cuda'),
                                torch.tensor([req.slot], dtype=torch.int32, device='cuda'))
            proposal = adapter.propose([req], None, inputs, graphs=False)
            adapter.model.forward = original
            ring = adapter.projected_rows(req, max(0, len(prompt) - adapter.window), len(prompt)).clone()
            first = int(req.output_ids[-1])
            serving.release(req)
        projected = torch.zeros((len(prompt) + 1, 5120), dtype=torch.bfloat16, device='cuda')
        projected[len(prompt) - ring.shape[0]:len(prompt)] = ring
        return projected, first, proposal.tokens[0], native['hidden'][0]
    yield {'target': target, 'serving': serving, 'adapter': adapter, 'draft': draft, 'ids': ids, 'ops': ops, 'serving_context': serving_context}
    del draft, adapter, serving, target
    gc.collect()
    torch.cuda.empty_cache()


def _relative_rms(actual, expected):
    error = actual.float() - expected.float()
    return (error.square().mean() / expected.float().square().mean()).sqrt().item()


def test_head_backward_weight_matches_the_nvfp4_kernel(runtime):
    draft, target, ops, ids = (runtime[k] for k in ('draft', 'target', 'ops', 'ids'))
    projected, first, _, _ = runtime['serving_context'](ids)
    with torch.no_grad():  # real draft hidden states: the kernel also quantizes activations to FP4
        anchors = torch.tensor([len(ids) // 4, len(ids) // 2, 3 * len(ids) // 4], device='cuda')
        hidden = draft.forward_blocks(projected, torch.tensor(ids + [first], device='cuda'), anchors)[:, 1:].reshape(-1, 5120)
        kernel = ops.nvfp4_linear(hidden, target.lm_head)
        dense = hidden @ draft.head.dequantized.T
    assert F.cosine_similarity(kernel.float().flatten(), dense.float().flatten(), dim=0) > 0.99
    assert (kernel.argmax(-1) == dense.argmax(-1)).float().mean() > 0.9
    probe = hidden.clone().requires_grad_(True)
    logits = draft.head(probe)
    assert torch.equal(logits, kernel)  # forward is serving's kernel bit for bit
    logits[:, :10].sum().backward()
    torch.testing.assert_close(probe.grad, draft.head.dequantized[:10].sum(0).expand_as(probe).to(torch.bfloat16), rtol=0.05, atol=0.05)


def test_training_forward_matches_serving_proposer(runtime):
    draft, ids = runtime['draft'], runtime['ids']
    projected, first, serving_proposals, serving_hidden = runtime['serving_context'](ids)
    sequence = torch.tensor(ids + [first], dtype=torch.int64, device='cuda')
    with torch.no_grad():
        anchor = torch.tensor([len(ids)], device='cuda')
        hidden = draft.forward_blocks(projected, sequence, anchor)[0]
        proposals = draft.propose(projected, sequence, anchor)[0]
    assert _relative_rms(hidden, serving_hidden) < 0.02
    assert (proposals == serving_proposals).float().mean() >= 0.8


def test_one_step_updates_serving_weights_in_place_and_keeps_context_weights(runtime):
    from sharar.train.loss import block_targets, block_terms, sample_anchors
    draft, ids = runtime['draft'], runtime['ids']
    model = draft.model
    before = {name: weight.clone() for name, weight in model.weights.items()}
    projected, first, _, _ = runtime['serving_context'](ids)
    ids = ids + [first]
    sequence = torch.tensor(ids, dtype=torch.int64, device='cuda')
    prompt_len = 100
    anchors = sample_anchors(prompt_len, len(ids), 8, torch.Generator().manual_seed(1)).to('cuda')
    target_ids, predecessor_ids, weight_mask, loss_weights = block_targets(sequence, anchors, 16, prompt_len, 7.0)
    hidden = draft.forward_blocks(projected, sequence, anchors)
    logits = draft.head(hidden)
    hp, pc, sc = draft.selector_weights()
    terms = block_terms(logits, hidden, target_ids, predecessor_ids, weight_mask, loss_weights, top_k=16,
                        hidden_projection=hp, predecessor_codebook=pc, successor_codebook=sc)
    loss = terms.loss(1.0)
    assert torch.isfinite(loss) and 0 < terms.metrics()['accuracy'] <= 1
    loss.backward()
    grads = [p for p in draft.parameters() if p.grad is not None]
    assert len(grads) == len(draft.parameters())
    optimizer = torch.optim.SGD(draft.parameters(), lr=1e-3)
    optimizer.step()
    draft.sync()
    changed = [name for name, weight in model.weights.items() if not torch.equal(weight, before[name])]
    assert changed and all(name in draft.masters for name in changed)
    assert not any(name.endswith(('self_attn.kv', 'self_attn.k_norm.weight')) or name in ('fc.weight', 'hidden_norm.weight') for name in changed)
    for name, weight in before.items():  # restore the shared serving weights for other tests
        model.weights[name].copy_(weight)
        if name in draft.masters:
            draft.masters[name].data.copy_(weight.float())
    for p in draft.parameters():
        p.grad = None
