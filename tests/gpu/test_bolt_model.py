"""Pinned Bolt: publisher block equations, full text logits and cache continuation."""

import gc
from pathlib import Path

import pytest
import torch

from sharar.engine import config
from sharar.model.nemotron import reference as R


@pytest.fixture(scope="module")
def snapshot():
    from sharar.target import NEMOTRON_BOLT
    try:
        return NEMOTRON_BOLT.snapshot_dir(config.current().hf_cache or Path.home() / ".cache" / "huggingface")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def model(snapshot):
    from sharar.launch.guard import start_thread
    from sharar.model.bolt.model import Bolt
    start_thread()
    with config.using(config.current().replace(target="bolt", w3_decode=False, moe_keep=0, moe_budget=0,
                                               moe_floors=None, moe_tau=0, moe_extra=0, ssd=False, kv_format="sim-bf16")):
        model = Bolt(snapshot, log=lambda msg: print(msg, flush=True))
    yield model
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def reference(snapshot):
    model = R.NemotronReference(snapshot, dtype=torch.bfloat16, expert_cache_gib=1, mtp=False)
    yield model
    model.experts.clear()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), 0).item()


@pytest.mark.parametrize("text", [None, "The capital of France is"])
def test_full_model_layer_and_logit_parity(model, reference, text):
    from sharar.tokenizer import SnapshotTokenizer
    tokens = SnapshotTokenizer(reference.snap).encode(text) if text else [1001, 1002, 1003, 1004]
    ids = torch.tensor(tokens, device="cuda")
    with torch.inference_mode():
        actual = model.prefill(ids, capture=True)
        expected = reference.forward(ids, capture=range(88), logits=1)
        values = [cosine(h + residual, expected["hidden"][i]) for i, (h, residual) in enumerate(actual["layers"])]
        logits = model.logits(actual["norm"][-1:]).float()
    print("Bolt minimum per-block cosine:", min(values), flush=True)
    print("Bolt final/logit cosine:", cosine(actual["norm"], expected["final"]), cosine(logits, expected["logits"]), flush=True)
    assert min(values) > .99
    assert cosine(logits, expected["logits"]) > .99
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("layer", [1, 3, 8, 21])
def test_puzzle_moe_matches_publisher_module(model, reference, layer):
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHBlock
    raw = R.read_json(reference.snap / "config.json")["llm_config"]
    raw.update(raw["block_configs"][layer])
    # Transformers names the count num_experts; Puzzle's layer config uses
    # n_routed_experts. Override the stale serialized aliases as well.
    raw.update(num_experts=raw["n_routed_experts"], num_local_experts=raw["n_routed_experts"])
    cfg = NemotronHConfig(**raw)
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        hf = NemotronHBlock(cfg, layer)
    hf = hf.to_empty(device="cuda").bfloat16().eval()
    p = reference.blocks[layer].mixer
    state = {"norm.weight": reference.blocks[layer].norm,
             "mixer.gate.weight": p.gate_w, "mixer.gate.e_score_correction_bias": p.bias,
             "mixer.fc1_latent_proj.weight": p.fc1.weight(torch.bfloat16),
             "mixer.fc2_latent_proj.weight": p.fc2.weight(torch.bfloat16),
             "mixer.shared_experts.up_proj.weight": p.shared_up.weight(torch.bfloat16),
             "mixer.shared_experts.down_proj.weight": p.shared_down.weight(torch.bfloat16)}
    ups, downs = [], []
    for e in range(raw["n_routed_experts"]):
        up, down = reference.experts.get(p.prefix, e)
        ups.append(up.weight(torch.bfloat16))
        downs.append(down.weight(torch.bfloat16))
    state["mixer.experts.up_proj"] = torch.stack(ups)
    state["mixer.experts.down_proj"] = torch.stack(downs)
    hf.load_state_dict(state)
    del state, ups, downs
    # Restore FP32 correction bits after Module.bfloat16().
    hf.mixer.gate.e_score_correction_bias = p.bias.clone()
    torch.manual_seed(14 + layer)
    x = torch.randn(3, 4096, device="cuda", dtype=torch.bfloat16) * .1
    with torch.inference_mode():
        normed = model.layers[layer].norm(x)
        actual = model.layers[layer].mixer(normed)
        expected = hf.mixer(normed[None])[0]
    assert cosine(actual, expected) > .999
    del hf


def test_chunked_prefill_prefix_cache_and_graph_decode(model):
    from sharar.engine.executor import Executor
    from sharar.engine.sampler import SamplingParams
    with config.using(config.current().replace(target="bolt", w3_decode=False, ssd=False, kv_format="sim-bf16", ssm_state="fp32")):
        ex = Executor(model, num_pages=4096, num_slots=6, max_batch=2, max_context=1024, use_cache=True)
        ids = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]
        params = SamplingParams(temperature=0., max_new_tokens=4)
        req = ex.admit(ids, params)
        while not req.prefill_done:
            ex.prefill(req, max_tokens=3)
        ex.release(req)
        # Prefix reuse must start from the matching Mamba state, not zero state.
        hit = ex.admit(ids + [1009, 1010], params)
        assert hit.prefix_len > 0
        ex.prefill(hit)
        eager_first = hit.output_ids[0]
        ex.decode_step([hit])
        eager_next = hit.output_ids[-1]
        ex.release(hit)
        ex.flush_cache()
        ex.capture_graphs()
        replay = ex.admit(ids + [1009, 1010], params)
        ex.prefill(replay)
        assert replay.output_ids[0] == eager_first
        ex.decode_step([replay])
        assert replay.output_ids[-1] == eager_next
        ex.release(replay)
        ex.state.check()


def test_ssd_restores_mamba_state_and_continuation(model, tmp_path):
    from sharar.engine.executor import Executor
    from sharar.engine.sampler import SamplingParams
    with config.using(config.current().replace(target="bolt", w3_decode=False, ssd=True, ssd_dir=str(tmp_path), ssd_gib=2.0,
                                               ssd_block_tokens=64, kv_format="fp8", ssm_state="fp32")):
        ex = Executor(model, num_pages=4096, num_slots=10, max_batch=1, max_context=1024, use_cache=True)
        ex.enable_ssd_tier(log=print)
        assert ex.cache.ssd is not None
        ex.cache.ssd.min_tokens = ex.cache.ssd.min_hit = 1
        ids = [1001 + i % 20 for i in range(160)]
        params = SamplingParams(max_new_tokens=2, ignore_eos=True)
        req = ex.generate(ids, params)
        prefix = req.prompt_ids + req.output_ids[:-1]
        match = ex.cache.match(prefix, touch=False)
        saved_ssm = ex.state_pool.ssm_all[:, match.snapshot_slot].clone()
        saved_conv = ex.state_pool.conv_all[:, match.snapshot_slot].clone()
        tier = ex.cache.ssd
        tier.drain(ex.cache)
        ex.cache.evict(need_pages=ex.kv_pool.num_pages)
        assert ex.cache.match(prefix, touch=False).length == 0
        assert ex.cache.restore(prefix + [1001]) == len(prefix)
        restored = ex.cache.match(prefix, touch=False)
        assert torch.equal(ex.state_pool.ssm_all[:, restored.snapshot_slot], saved_ssm)
        assert torch.equal(ex.state_pool.conv_all[:, restored.snapshot_slot], saved_conv)
        continuation = ex.generate(prefix + [1001], SamplingParams(max_new_tokens=1))
        assert continuation.prefix_len > 0
        ex.state.check()
        tier.drain(ex.cache)
        ex.flush_cache()
