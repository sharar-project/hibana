"""Compare the Sharar forward against the reference oracle outputs.

Two evidence sources under <reference>/:
  dump/<name>.pt   per-layer hidden states from tools/reference/dump_reference.py
  http/<name>.json top-k logprobs at every position and the greedy continuation
                   from tools/reference/http_oracle.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from sharar import config
from sharar.model.qwen38 import Qwen38
from sharar.parity.prompts import read_prompts

# Section 8 of the roadmap. W4A4/FP8 static activation quantization makes the
# 64-layer forward chaotic at the ulp level: the reference disagrees with
# itself at 84-88% top-1 (86-98% at confident positions) when only its prefill
# chunking changes (docs/parity.md). Criteria are therefore:
#   1. each block, fed the reference's own input, matches at cosine >= BLOCK_MIN_COS;
#   2. top-1 agreement at confident positions >= CONFIDENT_MIN_AGREEMENT;
#   3. the next token equals the reference's where the reference's margin is
#      too large for noise to flip it (>= NEXT_TOKEN_SURE_GAP nats; the
#      reference flips 35% of its own 2-3 nat predictions on text1024 and
#      4.6% of those above 5 nats).
# Criterion 2 needs a sample: prompts with fewer than MIN_CONFIDENT_POSITIONS
# confident positions are reported but not judged on it.
BLOCK_MIN_COS = 0.995
PROBE_LAYERS = (0, 3, 31, 47, 58, 59, 63)
CONFIDENT_GAP_NATS = 0.5
CONFIDENT_MIN_AGREEMENT = 0.80
MIN_CONFIDENT_POSITIONS = 20
NEXT_TOKEN_SURE_GAP = 5.0


def _err(mine: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    """(max abs err / ref scale, mean abs err / ref scale, cosine similarity)."""
    ref = ref.to(mine.device, torch.float32)
    mine = mine.float()
    diff = (mine - ref).abs()
    scale = ref.abs().mean().item() or 1.0
    cos = torch.nn.functional.cosine_similarity(mine.flatten(), ref.flatten(), dim=0).item()
    return diff.max().item() / scale, diff.mean().item() / scale, cos


def compare_dump(model: Qwen38, name: str, dump_path: Path, layers_to_show: str) -> dict:
    ref = torch.load(dump_path, map_location="cpu")
    ids = ref["input_ids"].to(model.device)
    started = time.perf_counter()
    mine = model.prefill(ids, capture=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    rows = []
    worst = {"mean_rel": 0.0, "cos": 1.0, "layer": None}
    e_max, e_mean, e_cos = _err(mine["embed"], ref["embed"])
    rows.append(("embed", e_mean, e_cos, None, None))
    n_ref = len(ref["layers"])
    for i in range(min(n_ref, len(mine["layers"]))):
        ref_h, ref_r = ref["layers"][i]
        my_h, my_r = mine["layers"][i]
        stream_ref = ref_h.float() + ref_r.float()
        stream_my = my_h.float() + my_r.float()
        _, s_mean, s_cos = _err(stream_my, stream_ref)
        _, h_mean, h_cos = _err(my_h, ref_h)
        kind = "attn" if model.layers[i].kind == "full_attention" else "gdn"
        rows.append((f"layer{i:02d}:{kind}", s_mean, s_cos, h_mean, h_cos))
        if s_cos < worst["cos"]:
            worst = {"mean_rel": s_mean, "cos": s_cos, "layer": i}
    ref_norm = ref["norm"][0] if isinstance(ref["norm"], (list, tuple)) else ref["norm"]
    _, n_mean, n_cos = _err(mine["norm"], ref_norm)
    rows.append(("final_norm", n_mean, n_cos, None, None))
    my_logits = model.logits(mine["norm"][-1:]).float()[0]
    ref_logits = ref["logits_last"].float().reshape(-1).to(model.device)
    _, l_mean, l_cos = _err(my_logits, ref_logits)
    my_top = my_logits.topk(8).indices.tolist()
    ref_top = ref_logits.topk(8).indices.tolist()
    rows.append(("logits_last", l_mean, l_cos, None, None))
    argmax_ok = my_top[0] == ref_top[0]
    top8_overlap = len(set(my_top) & set(ref_top))
    show = layers_to_show != "none"
    if show:
        wanted = None if layers_to_show == "all" else {int(x) for x in layers_to_show.split(",")}
        print(f"  {'point':<18}{'stream mean-rel':>16}{'stream cos':>12}{'block mean-rel':>16}{'block cos':>11}")
        for r in rows:
            if r[0].startswith("layer") and wanted is not None and int(r[0][5:7]) not in wanted:
                continue
            tail = f"{r[3]:>16.4f}{r[4]:>11.5f}" if r[3] is not None else ""
            print(f"  {r[0]:<18}{r[1]:>16.4f}{r[2]:>12.5f}{tail}")
    print(
        f"  [{name}] T={len(ids)} forward {elapsed:.2f}s  lowest stream cos {worst['cos']:.5f} at layer {worst['layer']} "
        f"(mean rel {worst['mean_rel']:.4f})  final_norm cos {n_cos:.5f}  logits cos {l_cos:.5f}  "
        f"argmax {'OK' if argmax_ok else 'MISMATCH'} (mine {my_top[0]} ref {ref_top[0]}) top8 overlap {top8_overlap}/8"
    )
    # Block probes: feed each probed layer the reference's own input.
    block_min = 1.0
    block_rows = []
    for i in PROBE_LAYERS:
        if i >= n_ref:
            continue
        layer = model.layers[i]
        h_i, r_i = ref["layers"][i]
        if i == 0:
            block_in = ref["embed"].to(model.device)
            h_p = r_p = None
        else:
            h_p, r_p = ref["layers"][i - 1]
            block_in = (h_p.float() + r_p.float()).to(torch.bfloat16).to(model.device)
        positions = torch.arange(len(ids), device=model.device)
        with torch.inference_mode():
            if i == 0:
                h_full, r_full = layer.prefill(block_in, None, positions)
            else:
                h_full, r_full = layer.prefill(h_p.to(model.device), r_p.to(model.device), positions)
        _, _, c = _err(h_full.float() + r_full.float(), h_i.float() + r_i.float())
        block_rows.append((i, c))
        block_min = min(block_min, c)
    blocks = " ".join(f"L{i}:{c:.4f}" for i, c in block_rows)
    print(f"      block cos with reference inputs: {blocks}")
    passed = block_min >= BLOCK_MIN_COS
    return {"name": name, "argmax_ok": argmax_ok, "top8_overlap": top8_overlap, "worst": worst, "block_min_cos": block_min, "passed": passed}


def compare_http(model: Qwen38, name: str, ids: list[int], http_path: Path) -> dict:
    ref = json.loads(http_path.read_text())
    assert ref["prompt_len"] == len(ids)
    x = torch.tensor(ids, dtype=torch.int64, device=model.device)
    started = time.perf_counter()
    out = model.prefill(x)
    arg, top_ids, top_vals = model.argmax_and_topk(out["norm"], k=8)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    # ref top logprobs at position i describe the prediction from position i-1.
    ref_top1 = [None] + [e[0][1] for e in ref["input_top_logprobs"][1:]]
    ref_gap = [None] + [(e[0][0] - e[1][0]) if len(e) > 1 else 9.0 for e in ref["input_top_logprobs"][1:]]
    my_arg = arg.tolist()
    my_lp = torch.log_softmax(top_vals, dim=-1)  # over the top-8 only; gaps are exact
    agree = 0
    mismatches = []
    for pos in range(len(ids) - 1):
        if my_arg[pos] == ref_top1[pos + 1]:
            agree += 1
        else:
            ref_entry = ref["input_top_logprobs"][pos + 1]
            my_gap = (top_vals[pos, 0] - top_vals[pos, 1]).item()
            mismatches.append((pos, my_arg[pos], ref_top1[pos + 1], ref_gap[pos + 1], my_gap, top_ids[pos, :3].tolist(), [e[1] for e in ref_entry[:3]]))
    next_ok = my_arg[-1] == ref["output_ids"][0]
    out_top = ref["output_top_logprobs"][0]
    next_gap = (out_top[0][0] - out_top[1][0]) if len(out_top) > 1 else 9.0
    next_required = next_gap >= NEXT_TOKEN_SURE_GAP
    confident_total = sum(1 for g in ref_gap[1:] if g is not None and g >= CONFIDENT_GAP_NATS)
    confident_miss = sum(1 for m in mismatches if m[3] is not None and m[3] >= CONFIDENT_GAP_NATS)
    confident_agree = confident_total - confident_miss
    conf_rate = confident_agree / max(1, confident_total)
    print(
        f"  [{name}] T={len(ids)} forward+logits {elapsed:.2f}s  top-1 agreement "
        f"{agree}/{len(ids) - 1} ({100 * agree / max(1, len(ids) - 1):.1f}%); "
        f"confident positions (ref gap >= {CONFIDENT_GAP_NATS} nats) {confident_agree}/{confident_total} ({100 * conf_rate:.1f}%); "
        f"next token {'OK' if next_ok else 'MISMATCH'} (mine {my_arg[-1]} ref {ref['output_ids'][0]}, ref gap {next_gap:.2f} nats{'' if next_required else ', within noise'})"
    )
    for m in [m for m in mismatches if m[3] is not None and m[3] >= 2.0][:4]:
        print(f"      confident miss pos {m[0]}: mine top3 {m[5]} (gap {m[4]:.3f})  ref top3 {m[6]} (gap {m[3]:.3f})")
    passed = (next_ok or not next_required) and (
        conf_rate >= CONFIDENT_MIN_AGREEMENT or confident_total < MIN_CONFIDENT_POSITIONS
    )
    return {"name": name, "agree": agree, "total": len(ids) - 1, "next_ok": next_ok, "next_gap": next_gap, "confident_rate": conf_rate, "passed": passed}


def run_parity(reference: str, prompts_sel: str, layers: str) -> int:
    ref_dir = Path(reference)
    prompts_path = ref_dir.parent / "prompts.json"
    prompts = read_prompts(prompts_path)
    wanted = None if prompts_sel == "all" else set(prompts_sel.split(","))
    snap = config.snapshot_dir()
    print(f"loading {config.MODEL_ID}@{config.MODEL_REVISION[:8]} from {snap}")
    model = Qwen38(snap)
    results = []
    all_ok = True
    for name, ids in prompts.items():
        if wanted is not None and name not in wanted:
            continue
        dump = ref_dir / "dump" / f"{name}.pt"
        http = ref_dir / "http" / f"{name}.json"
        if dump.is_file():
            r = compare_dump(model, name, dump, layers)
            results.append(r)
            all_ok &= r["passed"]
        if http.is_file():
            r = compare_http(model, name, ids, http)
            results.append(r)
            all_ok &= r["passed"]
        if not dump.is_file() and not http.is_file():
            print(f"  [{name}] no reference evidence")
        torch.cuda.empty_cache()
    summary_path = ref_dir.parent / "parity_summary.json"
    summary_path.write_text(json.dumps({"passed": all_ok, "results": results}, indent=1, default=str))
    print("PARITY", "PASS" if all_ok else "FAIL", f"(summary: {summary_path})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(run_parity(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "all", "none"))
