"""The real Nemotron model, MTP reconstruction and commits on eager and captured batches."""
import gc

import pytest
import torch

from sharar.cache.spec_budget import SpecConfig, verify_bytes
from sharar.engine import config
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.target import NEMOTRON, load_model
from sharar.tokenizer import SnapshotTokenizer


@pytest.fixture(scope="module")
def model():
    try:
        snap = NEMOTRON.snapshot_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    gc.collect()
    torch.cuda.empty_cache()
    with config.using(config.current().replace(target="nemotron", kv_format="fp8", w3_decode=False,
                                               w3_checkpoint=None)):
        loaded = load_model(NEMOTRON, snap)
    yield loaded, SnapshotTokenizer(snap)
    del loaded
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("state_format", ["fp32", "fp16_sr"])
def test_mtp_reconstruction_commits_real_batches(model, state_format):
    loaded, tok = model
    with config.using(config.current().replace(target="nemotron", kv_format="fp8", ssm_state=state_format,
            w3_decode=False, w3_checkpoint=None, ssd=False, draft_head="w2", device_accept=True,
            spec_depths="", moe_keep=12, moe_budget=128, moe_budget_per_request=48)):
        ex = Executor(loaded, num_pages=4096, num_slots=8, max_batch=2, max_context=512,
                      spec_config=SpecConfig("mtp", 3, "reconstruct"))
        ex.capture_graphs(log=print)
        assert ex.spec.bytes == verify_bytes(2, 4, "reconstruct", NEMOTRON)
        before = ex.kv_pool.free_count, ex.state_pool.free_count
        for graphs in (False, True):
            ex.use_graphs = graphs
            for batch in (1, 2):
                requests = []
                rounds = ex.spec.rounds
                try:
                    for text in ("List five prime numbers.", "Explain Python lists in one sentence.")[:batch]:
                        ids = tok.encode(f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n"
                                         "<|im_start|>assistant\n<think></think>")
                        requests.append(ex.admit(ids, SamplingParams(max_new_tokens=24, ignore_eos=True)))
                    ex.prefill_batch([(req, req.prefill_len - req.prefill_pos) for req in requests])
                    for _ in range(3):
                        lengths = [req.committed_len for req in requests]
                        ex.decode_step(requests)
                        for req, previous in zip(requests, lengths):
                            assert req.committed_len > previous
                            assert req.committed_len == req.context_len - 1
                            assert req.reservation is None and not req.finished
                        ex.state.check()
                    assert ex.spec.rounds > rounds
                finally:
                    for req in requests:
                        ex.release(req)
                    ex.flush_cache()
                assert (ex.kv_pool.free_count, ex.state_pool.free_count) == before
        ex.ledger.check_floor()
        del ex
    gc.collect()
    torch.cuda.empty_cache()
