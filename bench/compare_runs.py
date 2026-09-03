#!/usr/bin/env python3
"""Compare two controlled GB10 benchmark runs and enforce regression gates."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


Cell = tuple[int, int, int]


def read_summary(run_dir: Path) -> tuple[dict[Cell, dict[str, float]], int]:
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[Cell, list[dict[str, str]]] = {}
    failures = 0
    for row in rows:
        key = (
            int(row["input_length"]),
            int(row["output_length"]),
            int(row["concurrency"]),
        )
        grouped.setdefault(key, []).append(row)
        failures += int(row["failed_requests"])
    cells = {}
    for key, cell_rows in grouped.items():
        cells[key] = {
            "throughput": statistics.median(
                float(row["aggregate_output_tokens_per_second"])
                for row in cell_rows
            ),
            "ttft": statistics.median(
                float(row["median_ttft_seconds"])
                for row in cell_rows
                if row["median_ttft_seconds"]
            ),
            "minimum_available": min(
                float(row["minimum_mem_available_gib"]) for row in cell_rows
            ),
        }
    return cells, failures


def read_hashes(run_dir: Path) -> dict[tuple[Any, ...], str]:
    hashes = {}
    with (run_dir / "batches.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            batch = json.loads(line)
            if batch["phase"] != "measured":
                continue
            for request in batch["requests"]:
                key = (
                    int(batch["input_length"]),
                    int(batch["output_length"]),
                    int(batch["concurrency"]),
                    int(batch["repetition"]),
                    request["request_key"],
                )
                hashes[key] = request["response_sha256"]
    return hashes


def quality_passes(run_dir: Path) -> tuple[int, int]:
    cases = json.loads((run_dir / "quality.json").read_text(encoding="utf-8"))
    return sum(bool(case["passed"]) for case in cases), len(cases)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-throughput-regression-percent", type=float, default=3.0)
    parser.add_argument("--max-ttft-regression-percent", type=float, default=15.0)
    parser.add_argument("--minimum-available-gib", type=float, default=16.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline, baseline_failures = read_summary(args.baseline)
    candidate, candidate_failures = read_summary(args.candidate)
    errors: list[str] = []
    if baseline.keys() != candidate.keys():
        errors.append("baseline and candidate workload cells differ")
    if baseline_failures:
        errors.append(f"baseline contains {baseline_failures} failed requests")
    if candidate_failures:
        errors.append(f"candidate contains {candidate_failures} failed requests")

    print(" input output  c   base_tps   cand_tps    delta    ttft_delta  min_avail")
    for key in sorted(baseline.keys() & candidate.keys()):
        base = baseline[key]
        cand = candidate[key]
        throughput_delta = (cand["throughput"] / base["throughput"] - 1) * 100
        ttft_delta = (cand["ttft"] / base["ttft"] - 1) * 100
        print(
            f"{key[0]:6d} {key[1]:6d} {key[2]:2d} "
            f"{base['throughput']:10.3f} {cand['throughput']:10.3f} "
            f"{throughput_delta:+8.2f}% {ttft_delta:+10.2f}% "
            f"{cand['minimum_available']:9.2f} GiB"
        )
        if throughput_delta < -args.max_throughput_regression_percent:
            errors.append(f"{key}: throughput regressed {throughput_delta:.2f}%")
        if ttft_delta > args.max_ttft_regression_percent:
            errors.append(f"{key}: TTFT regressed {ttft_delta:.2f}%")
        if cand["minimum_available"] < args.minimum_available_gib:
            errors.append(
                f"{key}: MemAvailable fell to {cand['minimum_available']:.2f} GiB"
            )

    baseline_hashes = read_hashes(args.baseline)
    candidate_hashes = read_hashes(args.candidate)
    serial_baseline = {key: value for key, value in baseline_hashes.items() if key[2] == 1}
    serial_candidate = {
        key: value for key, value in candidate_hashes.items() if key[2] == 1
    }
    if serial_baseline != serial_candidate:
        errors.append("serial deterministic response hashes differ")
    concurrent_keys = {
        key for key in baseline_hashes.keys() & candidate_hashes.keys() if key[2] > 1
    }
    concurrent_drift = sum(
        baseline_hashes[key] != candidate_hashes[key] for key in concurrent_keys
    )
    print(
        f"Concurrent hash drift: {concurrent_drift}/{len(concurrent_keys)} "
        "(diagnostic only; batch scheduling is not bitwise deterministic)"
    )
    for name, run_dir in (("baseline", args.baseline), ("candidate", args.candidate)):
        passed, total = quality_passes(run_dir)
        if passed != total:
            errors.append(f"{name} quality canaries passed {passed}/{total}")

    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        f"PASS: {len(candidate)} cells, exact serial response hashes, "
        "quality canaries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
