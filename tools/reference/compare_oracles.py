#!/usr/bin/env python3
"""Top-1 agreement between two oracle directories (http/engine/engine_*), stdlib only."""
import json
import sys
from pathlib import Path

GAP = 0.5
ref_dir, a_name, b_name = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
for pa in sorted((ref_dir / a_name).glob("*.json")):
    pb = ref_dir / b_name / pa.name
    if not pb.is_file():
        continue
    a = json.loads(pa.read_text())
    b = json.loads(pb.read_text())
    n = agree = cn = cagree = 0
    for ta, tb in zip(a["input_top_logprobs"][1:], b["input_top_logprobs"][1:]):
        if ta is None or tb is None:
            continue
        n += 1
        same = ta[0][1] == tb[0][1]
        agree += same
        if len(tb) > 1 and tb[0][0] - tb[1][0] >= GAP:
            cn += 1
            cagree += same
    print(f"[{pa.stem}] {a_name} vs {b_name}: top-1 {100 * agree / max(1, n):.1f}% confident {100 * cagree / max(1, cn):.1f}% ({cagree}/{cn}) next {a['output_ids'][:3]} vs {b['output_ids'][:3]}")
