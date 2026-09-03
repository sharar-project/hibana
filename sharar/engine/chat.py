"""Interactive chat on the eager/graph decode loop, with per-turn speed stats."""

from __future__ import annotations

import sys
import time

import torch

from sharar import config
from sharar.api.chat_template import ChatTemplate
from sharar.engine.executor import Executor
from sharar.engine.sampler import SamplingParams
from sharar.model.qwen38 import Qwen38
from sharar.tokenizer import SnapshotTokenizer

IM_END = 248046


def _stream_generate(ex: Executor, tok: SnapshotTokenizer, ids: list[int], params: SamplingParams) -> dict:
    req = ex.admit(ids, params)
    t0 = time.perf_counter()
    ex.prefill(req)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    printed = ""
    t1 = time.perf_counter()
    steps = 0
    while True:
        text = tok.decode(req.output_ids, skip_special_tokens=True)
        if len(text) > len(printed) and not text.endswith("�"):
            sys.stdout.write(text[len(printed):])
            sys.stdout.flush()
            printed = text
        if req.finished:
            break
        ex.decode_step([req])
        steps += 1
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t1
    out = list(req.output_ids)
    cached = req.prefix_len
    ex.release(req)
    return {"output_ids": out, "ttft": ttft, "decode_s": decode_s, "steps": steps, "prompt_tokens": len(ids), "cached": cached}


def chat_cli(max_new_tokens: int, temperature: float, thinking: bool, system: str | None, prompt: str | None, no_graphs: bool) -> int:
    snap = config.snapshot_dir()
    tok = SnapshotTokenizer(snap)
    tmpl = ChatTemplate(snap)
    print(f"loading {config.MODEL_ID} ...", flush=True)
    model = Qwen38(snap, log=None)
    ex = Executor(model, log=print)
    if not no_graphs:
        ex.capture_graphs(log=None)
    print(f"ready ({torch.cuda.memory_allocated() / 2**30:.1f} GiB). thinking={'on' if thinking else 'off'} temperature={temperature} max_new_tokens={max_new_tokens}")
    print("commands: /reset clears history, /exit quits\n", flush=True)
    params = SamplingParams(max_new_tokens=max_new_tokens, temperature=temperature, seed=0 if temperature > 0 else None)
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    def turn(user_text: str) -> None:
        messages.append({"role": "user", "content": user_text})
        rendered = tmpl.render(messages, enable_thinking=thinking)
        ids = tok.encode(rendered)
        sys.stdout.write("assistant> ")
        sys.stdout.flush()
        r = _stream_generate(ex, tok, ids, params)
        reply = tok.decode(r["output_ids"], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": reply})
        gen = len(r["output_ids"])
        tps = (gen - 1) / r["decode_s"] if r["decode_s"] > 0 and gen > 1 else 0.0
        print(f"\n[{r['prompt_tokens']} prompt tokens ({r['cached']} from cache), TTFT {r['ttft'] * 1000:.0f} ms | {gen} generated, {tps:.1f} tok/s]\n", flush=True)

    if prompt is not None:
        turn(prompt)
        return 0
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        line = line.strip()
        if not line:
            continue
        if line == "/exit":
            return 0
        if line == "/reset":
            messages.clear()
            if system:
                messages.append({"role": "system", "content": system})
            print("(history cleared)")
            continue
        turn(line)
