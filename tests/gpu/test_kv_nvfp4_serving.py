"""The 27B on the NVFP4 KV pool with the prefix cache and the SSD tier: a prefix reused from the GPU cache, and one
demoted to the SSD and restored, come back with their codes and their scales (page aux rows): the next turn's first
token is a cold prefill's."""
import gc

import pytest
import torch

from sharar import target as targets
from sharar.engine.sampler import SamplingParams


def chat(tok, text: str) -> list[int]:
    return tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def user(tok, text: str) -> list[int]:
    return tok.encode(f"<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.fixture(scope="module", params=["nvfp4", "fp8"])
def runtime(tmp_path_factory, request):
    from sharar.cache.spec_budget import SpecConfig
    from sharar.engine import config
    from sharar.engine.executor import Executor
    from sharar.tokenizer import SnapshotTokenizer
    target = targets.get("qwen38")
    ssd = dict(ssd=True, ssd_dir=str(tmp_path_factory.mktemp("ssd")), ssd_gib=2.0)
    with config.using(config.current().replace(target="qwen38", kv_format=request.param, **ssd)):
        try:
            snap = target.snapshot_dir()
        except FileNotFoundError as exc:
            pytest.skip(str(exc))
        gc.collect()
        torch.cuda.empty_cache()
        model = targets.load_model(target, snap)
        ex = Executor(model, log=print, max_batch=4, max_context=32768, spec_config=SpecConfig("dflash2", 15, "reconstruct", "fp8", None))
        assert ex.kv_pool.nvfp4 == (request.param == "nvfp4")
        ex.capture_graphs(log=print)
        ex.warmup(log=print)
        ex.enable_ssd_tier(log=print)
        ex.cache.ssd.min_tokens = ex.cache.ssd.min_hit = 1  # the conversation is far shorter than a backup block
        yield ex, SnapshotTokenizer(snap)
    ex.flush_cache()
    del ex, model
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def test_cached_and_restored_prefixes_keep_their_codes_and_scales(runtime):
    ex, tok = runtime
    ex.flush_cache()
    before = ex.kv_pool.free_count, ex.state_pool.free_count

    def cold(ids, n):
        saved, ex.cache = ex.cache, None
        try:
            return ex.generate(ids, SamplingParams(max_new_tokens=n, ignore_eos=True))
        finally:
            ex.cache = saved

    first = ex.generate(chat(tok, "Name three rivers in Europe and one fact about each."), SamplingParams(max_new_tokens=40))
    second = first.prompt_ids + first.output_ids + user(tok, "Which of them is the longest? Answer in one sentence.")
    reference = cold(second, 24)
    ex.cache.evict(need_pages=ex.kv_pool.num_pages)  # demote every leaf to the SSD tier
    assert ex.cache.match(second[:-1], touch=False).length == 0 and ex.cache.ssd.tails > 0
    restored = ex.generate(second, SamplingParams(max_new_tokens=24, ignore_eos=True))  # the first turn from the SSD
    assert restored.prefix_len > 0 and ex.cache.ssd.restores > 0
    assert restored.output_ids[0] == reference.output_ids[0]
    hit = ex.generate(second, SamplingParams(max_new_tokens=24, ignore_eos=True))  # now from the GPU cache
    assert hit.prefix_len > 0
    assert hit.output_ids[0] == reference.output_ids[0]
    ex.flush_cache()
    assert (ex.kv_pool.free_count, ex.state_pool.free_count) == before
