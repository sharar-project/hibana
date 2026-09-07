"""Fold per-engine ShareGPT summaries into one comparison table (markdown)."""
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
engines = [p.name for p in sorted(root.iterdir()) if (p / "summary.json").exists()]
rows = {}
for e in engines:
    for r in json.loads((root / e / "summary.json").read_text())["rows"]:
        rows[(r["bucket"], r["concurrency"], e)] = r
buckets = ["short", "medium", "long"]
concs = sorted({k[1] for k in rows})
print("| bucket | conc | " + " | ".join(f"{e} tok/s (TTFT p50 s, TPOT ms)" for e in engines) + " |")
print("|---|---:|" + "|".join("---:" for _ in engines) + "|")
for b in buckets:
    for c in concs:
        cells = []
        for e in engines:
            r = rows.get((b, c, e))
            if r is None:
                cells.append("n/a")
            else:
                fail = f" ({r['failed']} failed)" if r.get("failed") else ""
                cells.append(f"{r['output_tok_s']} ({r['ttft_p50_s']}, {r['tpot_mean_ms']}){fail}")
        print(f"| {b} | {c} | " + " | ".join(cells) + " |")
print()
for e in engines:
    s = json.loads((root / e / "summary.json").read_text())
    gen = sum(r["generated_tokens"] for r in s["rows"]); reqs = sum(r["requests"] for r in s["rows"]); fails = sum(r["failed"] for r in s["rows"])
    print(f"{e}: {reqs} requests, {gen} generated tokens, {fails} failures, recorded {s['recorded']}")
