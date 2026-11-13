"""DFlash2 on Flash-Next with the SSD tier: a conversation whose prefix is demoted to the SSD and restored keeps the
draft's context (the projected features the tier stores with the KV), so the next turn drafts from it without rebuilding
it, its first token is a cold prefill's and it accepts nearly as well. The FP8 draft the launcher serves, the exact
profile's numerics (FP8 trunk, NVFP4 experts)."""
import gc

import pytest
import torch

from hibana import target as targets
from hibana.engine.sampler import SamplingParams

PROMPT = "Return only JSON with keys city and country for the capital of France."


def chat(tok, text: str) -> list[int]:
    return tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def user(tok, text: str) -> list[int]:
    return tok.encode(f"<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    from hibana.cache.spec_budget import SpecConfig
    from hibana.engine import config
    from hibana.engine.executor import Executor
    from hibana.proposers.factory import draft_snapshot
    from hibana.tokenizer import SnapshotTokenizer
    target = targets.get("flashnext")
    ssd = dict(ssd=True, ssd_dir=str(tmp_path_factory.mktemp("ssd")), ssd_gib=2.0)
    with config.using(config.current().replace(target="flashnext", w3_decode=False, w3_checkpoint=None, **ssd)):
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
        ex = Executor(model, log=print, max_batch=4, max_context=32768, spec_config=SpecConfig("dflash2", 9, "reconstruct", "fp8", None))
        ex.capture_graphs(log=print)
        ex.warmup(log=print)
        ex.enable_ssd_tier(log=print)
        # the test conversation is far shorter than the 4,096 tokens a backup needs and the 2,048 a restore must save
        ex.cache.ssd.min_tokens = ex.cache.ssd.min_hit = 1
    yield ex, SnapshotTokenizer(snap)
    ex.flush_cache()
    del ex, model
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def test_a_restored_prefix_keeps_the_drafts_context(runtime):
    ex, tok = runtime
    ex.flush_cache()
    before = ex.kv_pool.free_count, ex.state_pool.free_count
    proposer = ex.spec.proposer
    fallbacks = proposer.cache_fallbacks

    def cold(ids: list[int], n: int):
        saved, ex.cache = ex.cache, None
        try:
            return ex.generate(ids, SamplingParams(max_new_tokens=n, ignore_eos=True))
        finally:
            ex.cache = saved

    first = ex.generate(chat(tok, PROMPT), SamplingParams(max_new_tokens=24))
    second = first.prompt_ids + first.output_ids + user(tok, "Now return the same JSON with one more key, continent.")
    ex.cache.evict(need_pages=ex.kv_pool.num_pages)  # demote every leaf to the SSD tier
    assert ex.cache.match(second[:-1], touch=False).length == 0 and ex.cache.ssd.tails > 0
    restored = ex.generate(second, SamplingParams(max_new_tokens=48, ignore_eos=True))
    assert restored.prefix_len > 0 and ex.cache.ssd.restores > 0
    assert proposer.cache_fallbacks == fallbacks  # the draft's context came back with the prefix
    reference = cold(second, 48)
    assert restored.output_ids[0] == reference.output_ids[0]
    print("tokens per round: restored %.2f, cold %.2f" % (restored.spec_emitted / restored.spec_rounds,
                                                          reference.spec_emitted / reference.spec_rounds))
    assert restored.spec_emitted / restored.spec_rounds > 0.8 * reference.spec_emitted / reference.spec_rounds
    ex.flush_cache()
    assert (ex.kv_pool.free_count, ex.state_pool.free_count) == before
