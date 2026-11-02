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


@pytest.mark.parametrize("state_format", ["fp32", "fp16_sr", "int8"])
def test_mtp_reconstruction_commits_real_batches(model, state_format):
    loaded, tok = model
    with config.using(config.current().replace(target="nemotron", kv_format="fp8", ssm_state=state_format,
            w3_decode=False, w3_checkpoint=None, ssd=False, draft_head="w2", device_accept=True,
            spec_depths="", moe_keep=12, moe_budget=128, moe_budget_per_request=48)):
        ex = Executor(loaded, num_pages=4096, num_slots=8, max_batch=2, max_context=512,
                      spec_config=SpecConfig("mtp", 3, "reconstruct"))
        ex.capture_graphs(log=print)
        budget = verify_bytes(2, 4, "reconstruct", NEMOTRON)
        # an int8 pool's window ring is smaller than the Mamba-2 replay inputs the budget plans for
        assert ex.spec.bytes <= budget if ex.state_pool.window is not None else ex.spec.bytes == budget
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


def test_an_image_is_read_in_one_chunk_across_chunks_in_a_batch_and_after_a_cache_hit(model):
    """A prompt with an image (`engine/media.py`), its placeholders prefilled in one chunk, in chunks that split the
    image, and in a batched prefill beside a text request: each reads the code written in the image. A second request
    for the same prompt resumes from the cached prefix at a state snapshot inside the image and encodes the image again
    for the placeholders after it."""
    from PIL import Image as PILImage, ImageDraw, ImageFont
    import numpy as np
    from sharar.engine import media
    from sharar.engine.request import Request
    from sharar.model.nemotron import vision as v
    from sharar.weights.safetensors import SnapshotWeights

    loaded, tok = model
    with config.using(config.current().replace(vision=True)):
        w = SnapshotWeights(NEMOTRON.snapshot_dir())
        loaded.vision = loaded.load_vision(w, log=None)
        w.close()
    img = PILImage.new("RGB", (900, 300), "white")
    ImageDraw.Draw(img).text((40, 110), "CODE: BLUE-7429", font=ImageFont.load_default(56), fill="black")
    pixels = v.preprocess(np.array(img))
    n = v.image_tokens(pixels)
    head = tok.encode("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n")
    tail = tok.encode("What is the code in the image? Reply with the code only.<|im_end|>\n<|im_start|>assistant\n"
                      "<think></think>")
    ids = media.placeholder_ids(b"lifecycle", n)
    prompt = head + [v.IMAGE_START, *ids, v.IMAGE_END] + tail

    def request():
        return Request(prompt_ids=list(prompt), params=SamplingParams(max_new_tokens=8),
                       media=(media.Image(start=len(head) + 1, ids=ids, pixels=pixels),))

    def answer(req, ex):
        while not req.finished:
            ex.decode_step([req])
        return tok.decode(req.output_ids)

    try:
        with config.using(config.current().replace(target="nemotron", kv_format="fp8", w3_decode=False,
                                                   w3_checkpoint=None, ssd=False, vision=True)):
            ex = Executor(loaded, num_pages=8192, num_slots=8, max_batch=2, max_context=2048)
            outputs = []
            for chunk in (None, 100, 64):  # one chunk; chunks of 100 and 64 split the image's span
                req = request()
                ex.admit(req.prompt_ids, req.params, req)
                while not req.prefill_done:
                    ex.prefill(req, max_tokens=chunk)
                outputs.append(answer(req, ex))
                ex.release(req)
                ex.flush_cache()
            text_req = ex.admit(tok.encode("<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n<think></think>"),
                                SamplingParams(max_new_tokens=8))
            req = ex.admit(prompt, SamplingParams(max_new_tokens=8), request())
            ex.prefill_batch([(text_req, text_req.prefill_len), (req, 200)])
            ex.prefill_batch([(req, req.prefill_len - req.prefill_pos)])
            outputs.append(answer(req, ex))
            cached = ex.admit(prompt, SamplingParams(max_new_tokens=8), request())  # the same image: a cache hit
            assert len(head) + 1 < cached.prefill_pos < len(prompt), cached.prefill_pos
            while not cached.prefill_done:
                ex.prefill(cached)
            outputs.append(answer(cached, ex))
            for r in (text_req, req, cached):
                ex.release(r)
            ex.flush_cache()
            del ex
    finally:
        loaded.vision = None
        gc.collect()
        torch.cuda.empty_cache()
    print(outputs)
    assert all("BLUE-7429" in o for o in outputs), outputs
