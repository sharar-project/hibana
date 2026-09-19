"""Localize a reference-forward fault: per-layer activation RMS and ablation agreement against stored SGLang logprobs."""
import json, sys, time
from pathlib import Path
import torch
from bench.flashnext_parity import snapshot
from sharar.model.flashnext.reference import FlashNextReference

rows = json.loads(Path(sys.argv[1]).read_text())[:3]  # the short prompts
ref = FlashNextReference(snapshot())


def score(variant):
    ref.ablate = frozenset(variant)
    out_rows = []
    for row in rows:
        ids = row["ids"]
        out = ref.forward(ids, capture=True)
        logp = torch.log_softmax(out["logits"], -1)
        agree = diffs = 0
        n = 0
        for t in range(1, len(ids)):
            lp_s, tok, _ = row["input_token_logprobs"][t]
            if lp_s is None: continue
            s_top = [x[1] for x in row["input_top_logprobs"][t]] if row["input_top_logprobs"][t] else []
            agree += int(s_top and int(logp[t - 1].argmax()) == s_top[0]); diffs += logp[t - 1, tok].item() - lp_s; n += 1
        out_rows.append((row["id"], round(agree / max(1, n), 3), round(diffs / max(1, n), 2), [round(v, 1) for v in out["layers"]]))
    return out_rows


for variant in ([], ["ple"], ["fq"], ["shared"], ["hc"], ["ple", "hc"]):
    for rid, agree, dlp, rms in score(variant):
        print(json.dumps({"ablate": variant, "id": rid, "argmax_agreement": agree, "logprob_diff_mean": dlp, "layer_rms": rms if not variant else rms[:4] + ["..."] + rms[-2:]}), flush=True)
