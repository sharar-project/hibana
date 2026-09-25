"""Tabulate bench/h2h.py result files side by side (Markdown).

    python -m bench.h2h_report runs/sharar-27b.jsonl runs/sglang27.jsonl ...
"""
import json
import statistics
import sys
from collections import defaultdict


def med(xs):
    xs = [x for x in xs if x]
    return round(statistics.median(xs), 1) if xs else None


def main(paths):
    rows = [json.loads(line) for p in paths for line in open(p) if line.strip()]
    engines = list(dict.fromkeys(r["engine"] for r in rows))
    table = defaultdict(dict)
    for e in engines:
        rs = [r for r in rows if r["engine"] == e]
        probes = [r["tps"] for r in rs if r["workload"] == "probes" and r.get("median")]
        if probes:
            table["probes (thinking on, greedy median)"][e] = med(probes)
        dec = [r for r in rs if r["workload"] == "decode"]
        for mode in ("greedy", "t07", "t10"):
            m = [r for r in dec if r["mode"] == mode]
            if m:
                table[f"decode {mode}, all prompts"][e] = med(r["tps"] for r in m)
                for cat in sorted({r["cat"] for r in m}):
                    table[f"decode {mode}, {cat}"][e] = med(r["tps"] for r in m if r["cat"] == cat)
                table[f"ttft {mode} (s)"][e] = round(statistics.median(r["ttft_s"] for r in m), 3)
        for r in rs:
            if r["workload"] == "conc":
                table[f"conc {r['streams']} streams {r['mode']}, aggregate tok/s"][e] = r["aggregate_tps"]
            if r["workload"] == "long":
                table[f"cold ttft {r['length'] // 1024}K (s)"][e] = r["cold_ttft_s"]
                table[f"warm ttft {r['length'] // 1024}K (s)"][e] = r["warm_ttft_s"]
                table[f"needle {r['length'] // 1024}K"][e] = "ok" if r["correct"] else "MISS"
    print("| metric | " + " | ".join(engines) + " |")
    print("|---|" + "---:|" * len(engines))
    for k, v in table.items():
        print(f"| {k} | " + " | ".join(str(v.get(e, "")) for e in engines) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
