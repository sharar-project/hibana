"""Bolt LiLiCorr drafting, exact verification and retained-feature lifecycle."""
import gc
from pathlib import Path

import pytest
import torch

from sharar.engine import config
from sharar.engine.sampler import SamplingParams
from sharar.target import NEMOTRON_BOLT


def numerics(**extra):
    return config.using(config.current().replace(target='bolt', w3_decode=False, w3_checkpoint=None,
        moe_keep=0, moe_budget=0, moe_floors=None, moe_tau=0, moe_extra=0, ssd=False,
        kv_format='fp8', ssm_state='fp32', **extra))


@pytest.fixture(scope='module', params=('bf16', 'fp8'))
def runtime(request):
    from sharar.api.chat_template import ChatTemplate
    from sharar.cache.spec_budget import SpecConfig
    from sharar.engine.executor import Executor
    from sharar.launch.guard import start_thread
    from sharar.model.bolt.model import Bolt
    from sharar.proposers.factory import draft_snapshot
    from sharar.tokenizer import SnapshotTokenizer
    try:
        snap = NEMOTRON_BOLT.snapshot_dir(config.current().hf_cache or Path.home() / '.cache/huggingface')
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    if not (draft_snapshot('dflash2', None, NEMOTRON_BOLT) / 'model.safetensors').is_file():
        pytest.skip('pinned Bolt LiLiCorr snapshot is not cached')
    start_thread()
    with numerics(bolt_experts=request.param):
        model = Bolt(snap)
        ex = Executor(model, num_pages=8192, num_slots=24, max_batch=2, max_context=4096,
                      spec_config=SpecConfig('dflash2', 7, 'reconstruct', 'bf16'))
        ex.spec.verify_depths = (3, 7)
        ex.capture_graphs(log=print)
    tok, template = SnapshotTokenizer(snap), ChatTemplate(snap)
    def encode(text):
        return tok.encode(template.render([{'role': 'user', 'content': text}], enable_thinking=False))
    yield ex, tok, encode
    ex.flush_cache()
    del ex, model
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def test_target_taps_are_completed_residual_outputs(runtime):
    from sharar.engine.context import scratch_prefill_context
    from sharar.engine.contracts import FeatureRequest
    ex, _, _ = runtime
    ids = torch.tensor([1001, 1002, 1003], device='cuda')
    ctx = scratch_prefill_context(len(ids), 'cuda')
    ctx.feature_request = FeatureRequest(layers=ex.spec.proposer.model.features)
    with numerics():
        out = ex.model.forward(ids, ctx, capture=True)
    assert set(out['features']) == set(ex.spec.proposer.model.features)
    for lid, feature in out['features'].items():
        hidden, residual = out['layers'][lid]
        assert torch.equal(feature, hidden + residual)


@torch.inference_mode()
@pytest.mark.parametrize('depth', [3, 7])
def test_greedy_graphs_match_eager_and_accept_drafts(runtime, depth):
    ex, tok, encode = runtime
    ex.spec.verify_depth = depth
    ids = encode('Write a Python function that returns the first n Fibonacci numbers.')
    outputs, counts = [], []
    with numerics():
        for graphs in (False, True):
            ex.flush_cache()
            ex.use_graphs = graphs
            req = ex.generate(ids, SamplingParams(max_new_tokens=40, ignore_eos=True))
            outputs.append(req.output_ids)
            counts.append(req.spec_emitted / req.spec_rounds)
    print('Bolt LiLiCorr:', depth, counts, tok.decode(outputs[-1]), flush=True)
    assert min(counts) > 1.8
    assert outputs[0] == outputs[1]
    ex.use_graphs, ex.spec.verify_depth = True, 7


@torch.inference_mode()
def test_device_acceptance_and_cached_anchor_features(runtime):
    from sharar.benchmark_workloads import frozen_source
    ex, tok, encode = runtime
    document = tok.decode(tok.encode(frozen_source())[:300])
    ids = encode(document + '\n\nExplain why the sky appears blue in a few sentences.')
    outputs = []
    with numerics():
        for device_accept in (False, True):
            ex.spec.device_accept = device_accept
            ex.flush_cache()
            req = ex.generate(ids, SamplingParams(max_new_tokens=24, ignore_eos=True))
            outputs.append(req.output_ids)
        hits = ex.spec.proposer.cache_hits
        cached = []
        for _ in range(2):
            req = ex.generate(ids, SamplingParams(max_new_tokens=24, ignore_eos=True))
            assert req.prefix_len > 0 and req.spec_rounds > 0
            cached.append(req.output_ids)
        assert cached[0] == cached[1]
        assert ex.spec.proposer.cache_hits > hits
        assert ex.spec.proposer.cache_fallbacks == 0
        ex.state.check()
    assert outputs[0] == outputs[1]
    ex.spec.device_accept = False


@torch.inference_mode()
def test_sampled_mixed_batch_preserves_laws_and_repeats(runtime):
    ex, _, encode = runtime
    prompts = [encode('List three primary colors.'), encode('What is 17 plus 25?')]
    with numerics():
        outputs = []
        for _ in range(2):
            ex.flush_cache()
            reqs = ex.generate_batch([(p, SamplingParams(max_new_tokens=16, ignore_eos=True,
                                     temperature=.7 if i else 0., top_p=.95, top_k=20, seed=7))
                                     for i, p in enumerate(prompts)])
            assert all(r.spec_rounds > 0 for r in reqs)
            outputs.append([r.output_ids for r in reqs])
            ex.state.check()
    assert outputs[0] == outputs[1]


@torch.inference_mode()
def test_ssd_restore_keeps_drafting_features(runtime, tmp_path):
    ex, _, encode = runtime
    cfg = config.current().replace(target='bolt', w3_decode=False, w3_checkpoint=None,
        moe_keep=0, moe_budget=0, moe_floors=None, moe_tau=0, moe_extra=0, kv_format='fp8',
        ssm_state='fp32', ssd=True, ssd_dir=str(tmp_path), ssd_gib=2., ssd_block_tokens=64)
    with config.using(cfg):
        ex.flush_cache()
        ex.enable_ssd_tier()
        tier = ex.cache.ssd
        tier.min_tokens = tier.min_hit = 1
        prompt = encode('Explain photosynthesis in simple terms. ' * 18)
        req = ex.generate(prompt, SamplingParams(max_new_tokens=8, ignore_eos=True))
        prefix = prompt + req.output_ids[:-1]
        identity = ex.spec.proposer.cache_identity
        match = ex.cache.match(prefix, touch=False)
        assert identity in match.snapshot.auxiliary
        tier.drain(ex.cache)
        ex.cache.evict(need_pages=ex.kv_pool.num_pages)
        assert ex.cache.restore(prefix + [1001]) == len(prefix)
        assert identity in ex.cache.match(prefix, touch=False).snapshot.auxiliary
        resumed = ex.generate(prefix + [1001], SamplingParams(max_new_tokens=8, ignore_eos=True))
        assert resumed.prefix_len > 0 and resumed.spec_rounds > 0
        assert ex.spec.proposer.cache_fallbacks == 0
        ex.state.check()
        tier.drain(ex.cache)
        ex.flush_cache()
