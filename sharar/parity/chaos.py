"""Noise-floor measurements for the parity criteria.

1. reference vs reference: the serving container (http/) against the offline
   engine (engine/), both SGLang with different execution paths.
2. Sharar vs Sharar: the same prompt with the embedding perturbed by one bf16
   ulp on 10% of elements, and with the unfused SiLU/quant path.
All reported as top-1 agreement (overall and at confident positions) and final
hidden-state cosine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from sharar import config
from sharar.model.qwen38 import Qwen38
from sharar.parity.prompts import read_prompts

GAP = 0.5


def agreement(top_a: list, top_b: list) -> tuple[float, float]:
    """top_x[i] is the top list for the prediction at position i-1 (index 0 is None)."""
    n = agree = conf_n = conf_agree = 0
    for a, b in zip(top_a[1:], top_b[1:]):
        if a is None or b is None:
            continue
        n += 1
        same = a[0][1] == b[0][1]
        agree += same
        if len(b) > 1 and b[0][0] - b[1][0] >= GAP:
            conf_n += 1
            conf_agree += same
    return agree / max(1, n), conf_agree / max(1, conf_n)


def to_top(arg: torch.Tensor, ids: torch.Tensor, vals: torch.Tensor) -> list:
    lp = torch.log_softmax(vals.float(), dim=-1)
    out = [None]
    for pos in range(ids.shape[0] - 1):
        out.append([[lp[pos, j].item(), ids[pos, j].item()] for j in range(ids.shape[1])])
    return out


def main(reference: str, names: str) -> int:
    ref_dir = Path(reference)
    prompts = read_prompts(ref_dir.parent / "prompts.json")
    for name in names.split(","):
        http = json.loads((ref_dir / "http" / f"{name}.json").read_text())
        eng_path = ref_dir / "engine" / f"{name}.json"
        if eng_path.is_file():
            eng = json.loads(eng_path.read_text())
            a, c = agreement(eng["input_top_logprobs"], http["input_top_logprobs"])
            print(f"[{name}] reference engine vs reference server: top-1 {100 * a:.1f}%  confident {100 * c:.1f}%  next {eng['output_ids'][0]} vs {http['output_ids'][0]}")
    model = Qwen38(config.snapshot_dir(), log=None)
    for name in names.split(","):
        http = json.loads((ref_dir / "http" / f"{name}.json").read_text())
        ids = torch.tensor(prompts[name], device=model.device)
        base = model.prefill(ids)
        top_base = to_top(*model.argmax_and_topk(base["norm"], k=8))
        # perturb the embedding table copy for this run
        saved = model.embed
        noise = (torch.rand_like(saved.float()) < 0.1).to(saved.dtype) * torch.sign(torch.randn_like(saved.float())).to(saved.dtype)
        model.embed = (saved.float() + noise.float() * saved.float().abs() * 2**-8).to(saved.dtype)
        pert = model.prefill(ids)
        model.embed = saved
        top_pert = to_top(*model.argmax_and_topk(pert["norm"], k=8))
        for layer in model.layers:
            layer.mlp.fused_silu_quant = False
        unf = model.prefill(ids)
        for layer in model.layers:
            layer.mlp.fused_silu_quant = True
        top_unf = to_top(*model.argmax_and_topk(unf["norm"], k=8))
        cos_p = torch.nn.functional.cosine_similarity(base["norm"].float().flatten(), pert["norm"].float().flatten(), dim=0).item()
        cos_u = torch.nn.functional.cosine_similarity(base["norm"].float().flatten(), unf["norm"].float().flatten(), dim=0).item()
        a_p, c_p = agreement(top_pert, top_base)
        a_u, c_u = agreement(top_unf, top_base)
        a_r, c_r = agreement(top_base, http["input_top_logprobs"])
        a_pr, c_pr = agreement(top_pert, http["input_top_logprobs"])
        print(f"[{name}] sharar vs sharar+1ulp-noise: final cos {cos_p:.4f} top-1 {100 * a_p:.1f}% confident {100 * c_p:.1f}%")
        print(f"[{name}] sharar vs sharar-unfused-silu: final cos {cos_u:.4f} top-1 {100 * a_u:.1f}% confident {100 * c_u:.1f}%")
        print(f"[{name}] sharar vs reference server: top-1 {100 * a_r:.1f}% confident {100 * c_r:.1f}% | perturbed sharar vs reference: {100 * a_pr:.1f}% / {100 * c_pr:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
