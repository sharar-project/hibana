"""DFlash2 on Flash-Next with the pinned drafter: the feature taps, graphs against eager, device acceptance, the prefix
cache and the pools, with the FP8 draft the launcher serves. The exact profile's numerics (FP8 trunk, NVFP4 experts)."""
import gc

import pytest
import torch

from hibana import target as targets
from hibana.engine.sampler import SamplingParams

PROMPTS = ["Write a Python function that returns the n-th Fibonacci number iteratively, then explain it.",
           "A train leaves at 3:40 pm and travels 210 km at 84 km/h. When does it arrive? Think step by step.",
           "Summarize the causes of the French Revolution in five bullet points."]


def chat(tok, text: str) -> list[int]:
    return tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def numerics():
    from hibana.engine import config
    return config.using(config.current().replace(target="flashnext", w3_decode=False, w3_checkpoint=None, ssd=False))


@pytest.fixture(scope="module")
def flashnext():
    from hibana.proposers.factory import draft_snapshot
    from hibana.tokenizer import SnapshotTokenizer
    target = targets.get("flashnext")
    with numerics():
        try:
            snap = target.snapshot_dir()
        except FileNotFoundError as exc:
            pytest.skip(str(exc))
        if not (draft_snapshot("dflash2", None, target) / "model.safetensors").is_file():
            pytest.skip("the pinned Flash-Next DFlash2 draft is not in the model cache: %s@%s" % target.draft_pin("dflash2"))
        gc.collect()
        torch.cuda.empty_cache()
        model = targets.load_model(target, snap)
        torch.cuda.empty_cache()
    yield model, SnapshotTokenizer(snap)
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def runtime(flashnext):
    from hibana.cache.spec_budget import SpecConfig
    from hibana.engine.executor import Executor
    model, tok = flashnext
    with numerics():
        # the served shape: 9 drafts, one request verifying 9 and several 7, and throughput's 3; each round drafts only
        # the block it verifies (short drafts)
        ex = Executor(model, log=print, max_batch=4, max_context=32768, spec_config=SpecConfig("dflash2", 9, "reconstruct", "fp8", None))
        assert ex.spec.proposer.model.mask_token == 248077
        ex.spec.verify_depths = (3, 7, 9)
        ex.capture_graphs(log=print)
        ex.warmup(log=print)
    yield ex, tok
    ex.flush_cache()
    del ex
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
@pytest.mark.parametrize("length", [40, 1500])  # the narrow and the wide (hyper_prefill) paths
def test_taps_are_the_mean_of_each_layers_output_streams(flashnext, length):
    from hibana.benchmark_workloads import frozen_source
    from hibana.engine.context import scratch_prefill_context
    from hibana.engine.contracts import FeatureRequest
    model, tok = flashnext
    ids = torch.tensor(tok.encode(frozen_source())[:length], device=model.device)
    ctx = scratch_prefill_context(length, model.device)
    ctx.feature_request = FeatureRequest(layers=(5, 25, 45))
    out = model.forward(ids, ctx, capture=True)
    assert set(out["features"]) == {5, 25, 45}
    hc, h = model.target.hc_count, model.target.hidden_size
    for layer, feature in out["features"].items():
        stream = out["layers"][layer][0]  # layer L's completed output, as the drafter's SGLang capture reads it
        assert torch.equal(feature, stream.unflatten(-1, (hc, h)).float().mean(-2).to(torch.bfloat16))
        assert feature.data_ptr() != stream.data_ptr()
    assert "features" not in model.forward(ids, scratch_prefill_context(length, model.device))


@torch.inference_mode()
def test_graphs_match_eager_and_draft_well(runtime):
    # Prompt by prompt from an empty cache: Flash-Next's verify rows depend on their co-rows (the per-tensor FP8
    # activation scale), so equal outputs need equal drafts from the eager and the graphed draft. The first 64 tokens
    # must match: a near-tie in the MoE trunk can part a longer continuation from one run to the next.
    ex, tok = runtime
    prompts = [chat(tok, p) for p in PROMPTS]
    for depth in (9, 7, 3):
        ex.spec.verify_depth = depth
        outputs, rounds, emitted = {}, 0, 0
        for graphs in (False, True):
            ex.use_graphs = graphs
            outputs[graphs] = []
            for ids in prompts:
                ex.flush_cache()
                r = ex.generate_batch([(ids, SamplingParams(max_new_tokens=96, ignore_eos=True))])[0]
                outputs[graphs].append(r.output_ids)
                rounds, emitted = rounds + r.spec_rounds, emitted + r.spec_emitted
        print(f"depth {depth}: {emitted / rounds:.2f} tokens per round")
        assert emitted / rounds > (3.0 if depth > 3 else 2.5)  # wrong taps, mask token or head fall towards 1
        assert [o[:64] for o in outputs[False]] == [o[:64] for o in outputs[True]]
    ex.use_graphs, ex.spec.verify_depth = True, 9


@torch.inference_mode()
def test_a_round_writes_only_its_blocks_noise_slots(runtime):
    # depth 3 drafts a block of 4: each request's history ring and its noise slots past the block keep their bits, eager
    # and graphed, three requests in the bucket of four (the padded row writes the dummy slot 0)
    ex, tok = runtime
    p = ex.spec.proposer
    assert set(p.graphs.static) == {(b, d) for d, g in ex.spec.width_graphs.items() for b in g.sizes}
    checked, propose = [], p.propose

    def watched(requests, pages, inputs, *, graphs, depth=None):
        before = [k.view(torch.int16).clone() for k, _ in p.caches]
        out = propose(requests, pages, inputs, graphs=graphs, depth=depth)
        for layer, (keys, _) in enumerate(p.caches):
            for r in requests:
                base = r.proposer_state.slot * p.pitch
                for lo, hi in ((base, base + p.window), (base + p.window + depth + 1, base + p.pitch)):
                    assert torch.equal(keys[lo:hi].view(torch.int16), before[layer][lo:hi])
        checked.append((len(requests), depth))
        return out
    p.propose = watched
    ex.spec.verify_depth = 3
    try:
        for graphs in (False, True):
            ex.use_graphs = graphs
            ex.flush_cache()
            ex.generate_batch([(chat(tok, q), SamplingParams(max_new_tokens=16, ignore_eos=True)) for q in PROMPTS])
    finally:
        del p.propose
        ex.use_graphs, ex.spec.verify_depth = True, 9
    assert (3, 3) in checked


@torch.inference_mode()
def test_device_acceptance_matches_the_host(runtime):
    ex, tok = runtime
    outputs = []
    for device_accept in (False, True):
        ex.spec.device_accept = device_accept
        ex.flush_cache()
        outputs.append([r.output_ids for r in ex.generate_batch([(chat(tok, PROMPTS[0]), SamplingParams(max_new_tokens=64, ignore_eos=True))])])
    ex.spec.device_accept = False
    assert outputs[0] == outputs[1]


@torch.inference_mode()
def test_mixed_batch_repeats_and_reuses_the_prefix_cache(runtime):
    # one sampled request among greedy ones: the batch samples DFlash2's lattice with each row's Philox stream; a shared
    # document of ~600 tokens makes the later runs prefix-cache hits, which restore the draft's retained features. A
    # cached prefix's KV is equivalent to a cold prefill's, not bitwise (docs/cache_design.md), so the cached runs are
    # compared with each other
    from hibana.benchmark_workloads import frozen_source
    ex, tok = runtime
    document = tok.decode(tok.encode(frozen_source())[:600])
    prompts = [chat(tok, f"{document}\n\n{p}") for p in PROMPTS]

    def generate(cached=True):
        reqs = ex.generate_batch([(ids, SamplingParams(max_new_tokens=24, seed=7, temperature=.7 if i == 1 else 0., ignore_eos=True))
                                  for i, ids in enumerate(prompts)])
        assert not cached or all(r.prefix_len > 0 for r in reqs)  # the run before cached each prompt
        return [r.output_ids for r in reqs]
    ex.flush_cache()
    generate(cached=False)
    hits = ex.spec.proposer.cache_hits
    assert generate() == generate()  # seeded sampling repeats through the cached prefix
    assert ex.spec.proposer.cache_hits > hits and ex.spec.proposer.cache_fallbacks == 0
    ex.flush_cache()
    assert ex.kv_pool.free_count == ex.kv_pool.num_pages - 1
    assert ex.state_pool.free_count == ex.state_pool.num_slots - 1
