"""Flash-Next executor lifecycle on the pinned checkpoint (#54): prefill, MTP speculative rounds, release, the
prefix cache and the cold tier, with every page and state slot back afterwards; repeat runs are identical."""
import gc

import pytest
import torch

from sharar import target as targets
from sharar.engine.sampler import SamplingParams

PROMPTS = [
    "Return only JSON with keys city and country for the capital of France.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain in three sentences how a hash map resolves collisions.",
    "List the first five prime numbers separated by commas.",
]


def chat(tok, text: str) -> list[int]:
    return tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.fixture(scope="module")
def runtime():
    from sharar.cache.spec_budget import SpecConfig
    from sharar.engine.executor import Executor
    from sharar.tokenizer import SnapshotTokenizer
    target = targets.get("flashnext")
    try:
        snap = target.snapshot_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    from sharar.engine import config
    gc.collect()
    torch.cuda.empty_cache()
    # The exact profile: the FP8 trunk without 3-bit copies (tools/gpu-tests.sh mounts the 27B's for the rest of the suite).
    with config.using(config.current().replace(target="flashnext", w3_decode=False, w3_checkpoint=None)):
        model = targets.load_model(target, snap)
        torch.cuda.empty_cache()
        ex = Executor(model, log=print, max_context=32768, spec_config=SpecConfig("mtp", 4, "reconstruct", "bf16", None))
        ex.capture_graphs(log=print)
        ex.warmup(log=print)
        ex.enable_cold_tier(budget_gib=2, log=print)
    yield ex, SnapshotTokenizer(snap)
    ex.flush_cache()
    del ex, model
    gc.collect()
    torch.cuda.empty_cache()


def free(ex) -> tuple[int, int]:
    assert len(ex.kv_pool._free) == len(set(ex.kv_pool._free)) and len(ex.state_pool._free) == len(set(ex.state_pool._free))
    return ex.kv_pool.free_count, ex.state_pool.free_count


@pytest.mark.parametrize("batch", [1, 4])
def test_speculative_batches_finish_and_return_every_page_and_slot(runtime, batch):
    ex, tok = runtime
    ex.flush_cache()
    before = free(ex)
    emitted = ex.spec.emitted
    reqs = ex.generate_batch([(chat(tok, p), SamplingParams(max_new_tokens=40)) for p in PROMPTS[:batch]])
    assert all(r.finished and 0 < len(r.output_ids) <= 40 for r in reqs)
    assert ex.spec.emitted > emitted  # the MTP drafts were verified and accepted
    ex.flush_cache()
    assert free(ex) == before
    ex.ledger.check_floor()


@pytest.mark.parametrize("graphs, batch", [(True, 1), (False, 4), (True, 4)])
def test_repeat_runs_in_one_process_are_identical(runtime, graphs, batch):
    """Bit-identical run to run, eager or with CUDA graphs. Batch 4 with graphs covers padded rounds (requests finish
    at different rounds, so three rows replay the batch-4 graph): padded rows once wrote page 0 together and the
    race reached the real rows through the per-tensor activation scale (#94)."""
    ex, tok = runtime
    spec = ex.spec
    vg, dg = spec.verify_graphs, spec.proposer.graphs
    ex.use_graphs = graphs
    if not graphs:
        spec.verify_graphs, spec.proposer.graphs = None, None
    try:
        def run():
            ex.flush_cache()
            if batch == 1:
                return [ex.generate(chat(tok, p), SamplingParams(max_new_tokens=32)).output_ids for p in PROMPTS]
            return [r.output_ids for r in ex.generate_batch([(chat(tok, p), SamplingParams(max_new_tokens=32)) for p in PROMPTS])]
        assert run() == run() == run()
    finally:
        spec.verify_graphs, spec.proposer.graphs = vg, dg
        ex.use_graphs = True


def test_prefix_cache_and_cold_tier_continue_like_a_cold_prefill(runtime):
    """A conversation: turn two reuses turn one's cached prefix; then everything is parked in the cold tier and
    turn three restores the parked session before matching. Each continues exactly as a cold prefill does."""
    ex, tok = runtime
    ex.flush_cache()
    before = free(ex)

    def user(text: str) -> list[int]:
        return tok.encode(f"<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def cold(ids: list[int]) -> list[int]:
        saved, ex.cache = ex.cache, None
        try:
            return ex.generate(ids, SamplingParams(max_new_tokens=1)).output_ids
        finally:
            ex.cache = saved

    first = ex.generate(chat(tok, PROMPTS[0]), SamplingParams(max_new_tokens=24))
    second = first.prompt_ids + first.output_ids + user("Now return the same JSON with one more key, continent.")
    hit = ex.generate(second, SamplingParams(max_new_tokens=1))
    assert hit.prefix_len > 0 and hit.output_ids == cold(second)
    ex.cache.evict(need_pages=ex.kv_pool.num_pages)  # park every leaf in the cold tier
    third = second + hit.output_ids + user("Return only the value of the continent key.")
    assert ex.cache.match(third[:-1], touch=False).length == 0 and ex.cache.tier.stats()["entries"] > 0
    restored = ex.generate(third, SamplingParams(max_new_tokens=1))
    assert restored.prefix_len > 0 and ex.cache.tier.stats()["restores"] > 0
    assert restored.output_ids == cold(third)
    ex.flush_cache()
    assert free(ex) == before
