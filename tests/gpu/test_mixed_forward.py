"""Real mixed target numerics, private state, cache and continuation on GB10."""
import gc
import json
import os

import pytest
import torch

from sharar.engine.sampler import SamplingParams

pytestmark = pytest.mark.skipif(os.environ.get('SHARAR_RUN_GPU') != '1', reason='opt-in GB10 checkpoint tests')


@pytest.fixture(scope='module', params=[('dflash2', 'reconstruct'), ('dspark', 'snapshots'), ('mtp', 'reconstruct')],
                ids=['dflash2', 'dspark', 'mtp'])
def runtime(request):
    from sharar import config
    from sharar.engine.executor import Executor
    from sharar.model.qwen38 import Qwen38
    from sharar.parity.prompts import PARAGRAPH
    from sharar.tokenizer import SnapshotTokenizer
    gc.collect()
    torch.cuda.empty_cache()
    with torch.inference_mode():
        ex = Executor(Qwen38(config.snapshot_dir()), num_pages=35000, num_slots=12)
        name, strategy = request.param
        ex.enable_spec(3 if name == 'mtp' else 7, proposer=name, state_strategy=strategy)
        ex.enable_mixed('full')
        ex.capture_graphs(log=print)
        ex.warmup(log=print)
        ids = SnapshotTokenizer(config.snapshot_dir()).encode(PARAGRAPH * 16)
        yield ex, ids
        ex.flush_cache()


def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    if not a.count_nonzero() and not b.count_nonzero():
        return 1.
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def logits_gate(model, reference, actual):
    from sharar.parity.run import CONFIDENT_GAP_NATS, CONFIDENT_MIN_AGREEMENT, MIN_CONFIDENT_POSITIONS, NEXT_TOKEN_SURE_GAP
    ref = model.logits(reference).float()
    got = model.logits(actual).float()
    assert torch.isfinite(ref).all() and torch.isfinite(got).all()
    values, indices = ref.topk(2, dim=-1)
    margin = values[:, 0] - values[:, 1]
    equal = indices[:, 0] == got.argmax(-1)
    confident = margin >= CONFIDENT_GAP_NATS
    if confident.sum() >= MIN_CONFIDENT_POSITIONS:
        assert equal[confident].float().mean() >= CONFIDENT_MIN_AGREEMENT
    if margin[-1] >= NEXT_TOKEN_SURE_GAP:
        assert equal[-1]
    return {'top1_agreement': equal.float().mean().item(), 'confident_positions': confident.sum().item(),
            'confident_agreement': equal[confident].float().mean().item() if confident.any() else None,
            'last_equal': equal[-1].item(), 'last_margin': margin[-1].item(), 'norm_cosine': cosine(reference, actual)}


@torch.inference_mode()
def check_blocks(runtime, batch, prefix, share, tmp_path):
    from sharar.parity.run import BLOCK_MIN_COS
    ex, ids = runtime
    ex.flush_cache()
    owned = []
    selected = (0, 3, 31, 47, 58, 59, 63)
    captured, originals = {}, []
    try:
        old = [ex.admit(ids[:128], SamplingParams(max_new_tokens=24, ignore_eos=True)) for _ in range(batch)]
        owned.extend(old)
        for req in old: ex.prefill(req)
        new = ex.admit(ids[:prefix + 128], SamplingParams(max_new_tokens=8, ignore_eos=True))
        owned.append(new)
        if prefix: ex.prefill(new, prefix)
        prefill = ex._prepare_prefill([(new, 128)])
        verification = ex.spec.prepare(old)
        ctx = ex.spec.context(verification, ex._mixed_verify)
        contexts = (prefill.context, ctx)
        state, conv = ex.state_pool.ssm_all.clone(), ex.state_pool.conv_all.clone()
        # Borrowed/read-only prefix pages and the dummy page must not be written.
        guards = torch.cat([r.pages[:r.committed_len] for r in old] + [new.pages[:prefix], new.pages.new_zeros(1)])
        kv_before = {(kind, lid): pool[lid][guards].clone() for kind, pool in (('k', ex.kv_pool.k), ('v', ex.kv_pool.v))
                     for lid in ex.kv_pool.layer_ids}
        for lid in selected:
            layer, original = ex.model.layers[lid], ex.model.layers[lid].forward
            originals.append((layer, original))
            def record(h, r, context, original=original, lid=lid):
                captured[lid, context.mode] = (h.clone(), r.clone() if r is not None else None)
                return original(h, r, context)
            layer.forward = record
        reference = (ex.model.forward(prefill.ids, prefill.context), ex.model.forward(verification.tokens, ctx))
        for layer, original in originals: layer.forward = original
        originals.clear()
        report = {'share': share, 'batch': batch, 'prefix': prefix, 'blocks': {}}
        for lid in selected:
            layer = ex.model.layers[lid]
            def inputs():
                return [(h.clone(), r.clone() if r is not None else None)
                        for h, r in (captured[lid, 'prefill'], captured[lid, 'verify'])]
            def restore_layer():
                if lid in ex.state_pool.ssm:
                    index = ex.state_pool.layer_ids.index(lid)
                    ex.state_pool.ssm[lid].copy_(state[index])
                    ex.state_pool.conv[lid].copy_(conv[index])
            restore_layer()
            separate = [layer.forward(h, r, c) for (h, r), c in zip(inputs(), contexts)]
            restore_layer()
            mixed = layer.mixed(inputs(), contexts, share=share)
            values = [cosine(h + r, mh + mr) for (h, r), (mh, mr) in zip(separate, mixed)]
            assert min(values) >= BLOCK_MIN_COS, (lid, values)
            report['blocks'][lid] = values
        ex.state_pool.ssm_all.copy_(state)
        ex.state_pool.conv_all.copy_(conv)
        out_p, out_v = ex.model.forward_mixed(prefill.ids, prefill.context, verification.tokens, ctx, share=share)
        report['prefill'] = logits_gate(ex.model, reference[0]['norm'], out_p['norm'])
        width = verification.proposal.tokens.shape[1] + 1
        report['verify'] = [logits_gate(ex.model, reference[1]['norm'][i * width:(i + 1) * width],
                                      out_v['norm'][i * width:(i + 1) * width]) for i in range(batch)]
        writable = {new.slot, *prefill.context.checkpoint_slots}
        untouched = [slot for slot in range(ex.state_pool.num_slots) if slot not in writable]
        assert torch.equal(ex.state_pool.ssm_all[:, untouched], state[:, untouched])
        assert torch.equal(ex.state_pool.conv_all[:, untouched], conv[:, untouched])
        for (kind, lid), before in kv_before.items():
            after = getattr(ex.kv_pool, kind)[lid][guards]
            assert torch.equal(after.view(torch.uint8), before.view(torch.uint8))
        ex._finish_prefill(prefill, out_p)
        ex.spec.finish(verification, out_v)
        assert all(r.reservation is None and r.committed_len == r.context_len - 1 for r in owned)
        assert ex.spec.ready(new)
        (tmp_path / 'numerics.json').write_text(json.dumps(report, indent=2) + '\n')
    finally:
        for layer, original in originals: layer.forward = original
        for req in owned:
            req.cacheable = False
            ex.release(req)
        ex.flush_cache()
    assert ex.kv_pool.free_count == ex.kv_pool.num_pages - 1
    assert ex.state_pool.free_count == ex.state_pool.num_slots - 1


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('prefix', [0, 64])
@pytest.mark.parametrize('share', ['mlp', 'full'])
def test_real_blocks_logits_and_untouched_slots(runtime, batch, prefix, share, tmp_path):
    check_blocks(runtime, batch, prefix, share, tmp_path)


@pytest.mark.parametrize('depth', [1, 3])
def test_reduced_verification_width(runtime, depth, tmp_path):
    ex, _ = runtime
    if depth not in ex.spec.verify_depths:
        pytest.skip('adapter does not support this verification width')
    previous = ex.spec.verify_depth
    ex.spec.verify_depth = depth
    try:
        check_blocks(runtime, batch=1, prefix=64, share='full', tmp_path=tmp_path)
    finally:
        ex.spec.verify_depth = previous


@pytest.mark.parametrize('sampled', [False, True])
@pytest.mark.parametrize('cached', [False, True])
@torch.inference_mode()
def test_mixed_cache_stop_and_continuation(runtime, sampled, cached):
    ex, ids = runtime
    ex.flush_cache()
    owned = []
    try:
        if cached:
            prime = ex.generate(ids[:128], SamplingParams(max_new_tokens=1))
            assert prime.released
        old = [ex.admit(ids[:160], SamplingParams(max_new_tokens=2 + i * 2,
                         temperature=.7 if sampled else 0, seed=17 + i)) for i in range(3)]
        owned.extend(old)
        for req in old: ex.prefill(req)
        new = ex.admit(ids[:192], SamplingParams(max_new_tokens=5, temperature=.7 if sampled else 0, seed=29))
        owned.append(new)
        if cached: assert new.prefix_len > 0
        chunks = [(new, new.prefill_len - new.prefill_pos)]
        mixed = ex.mixed_prefill(chunks, old)
        if not mixed:
            # MTP's sampled requests retain the existing plain-decode fallback.
            assert sampled and ex.spec.proposer.requirements.greedy_only
            assert not new.output_ids and all(r.reservation is None for r in owned)
            ex.prefill_batch(chunks)
            ex.decode_step(old)
        while any(not r.finished for r in owned):
            ex.decode_step([r for r in owned if not r.finished])
        assert all(r.reservation is None and r.committed_len == r.context_len - 1 for r in owned)
        assert all(len(r.output_ids) <= r.params.max_new_tokens for r in owned)
        for req in owned: ex.release(req)
        repeat = ex.generate(ids[:192], SamplingParams(max_new_tokens=2))
        assert repeat.prefix_len > 0 and repeat.released
    finally:
        for req in owned: ex.release(req)
        ex.flush_cache()
    assert ex.kv_pool.free_count == ex.kv_pool.num_pages - 1
    assert ex.state_pool.free_count == ex.state_pool.num_slots - 1
