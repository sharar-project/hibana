"""Freeze the family-gate inputs from source: the synthetic task families plus the benchmark prompts.

    python3 -m bench.family_inputs freeze --output $SHARAR_RESULTS/datasets/family-eval-v1/task-inputs.json

`bench/workloads/family_v1.json` holds the family specifications (protocol
`causal-selector-v1`); the three benchmark prompts come from
`sharar/benchmark_workloads.py` at 10,000 input tokens. The output is
byte-identical to the frozen file the 2026-09-10 family evaluations used
(SHA-256 0c0da566...), so a gate run can regenerate its inputs and verify them.
`--extra-code-prompt` appends the tool-free code prompt as `benchmark_code_v2`:
that is the `family-eval-v2` set (22 tasks, SHA-256 fc733283...), the gate's
default until 2026-09-15; `--protocol bench/workloads/family_v3.json --extra-code-prompt`
freezes `family-eval-v3` (101 families plus four benchmark prompts, 105 tasks,
SHA-256 b72bdecd...), the gate's default from 2026-09-15 on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sharar.benchmark_workloads import WORKLOADS, materialize as materialize_benchmark

PROTOCOL = Path(__file__).resolve().parent / "workloads" / "family_v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(protocol: dict, encode, benchmarks: dict[str, list[int]]) -> dict:
    tasks = protocol["tasks"]
    if len({t["id"] for t in tasks}) != len(tasks) or any(t["split"] not in ("train", "validation", "test") for t in tasks):
        raise ValueError("task families must be unique with explicit splits")
    out = []
    for task in tasks:
        head = encode("<|im_start|>user\nEngineering task " + task["id"] + ".\nSpecification:\n" + task["spec"] + "\nHistorical scenario records:\n")
        records = "\n".join(f"Record {i}: scenario={task['id']}; item={i % 97}; arrival_tick={11*i}; "
                            f"completed={i % 7 != 0}; attempt={1 + i % 3}; size_bytes={64 + 17*(i % 59)}; "
                            f"revision={i // 97}; elapsed_ms={13 + (37*i) % 541}." for i in range(800))
        body = encode(records)
        tail = encode("\n[End of historical records. These are synthetic examples, not additional requirements.]\n" + task["question"] +
                      "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        size = protocol["input_tokens"] - len(head) - len(tail)
        if not 0 < size <= len(body):
            raise ValueError("task envelope or records do not fit input budget")
        out.append({**task, "input_ids": head + body[:size] + tail, "output_tokens": protocol["output_tokens"]})
    for case in WORKLOADS:
        out.append({"id": "benchmark_" + case, "split": "benchmark", "category": case,
                    "input_ids": benchmarks[case], "output_tokens": 1000})
    assert all(len(t["input_ids"]) == protocol["input_tokens"] for t in out)
    return {"version": protocol["version"], "tasks": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, default=PROTOCOL)
    freeze.add_argument("--extra-code-prompt", action="store_true", help="append benchmark_code_v2 (the tool-free code prompt)")
    args = ap.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    from sharar import config
    from sharar.tokenizer import SnapshotTokenizer

    snapshot = config.snapshot_dir()
    tok = SnapshotTokenizer(snapshot)
    protocol = json.loads(args.protocol.read_text())
    # benchmark_code stays the original 2026-09 prompt (now `code_v1`) so v1 regenerates byte-identically;
    # the tool-free prompt is the separate benchmark_code_v2 task.
    benchmarks = {case: materialize_benchmark(tok.encode, "code_v1" if case == "code" else case, protocol["input_tokens"]) for case in WORKLOADS}
    data = materialize(protocol, tok.encode, benchmarks)
    if args.extra_code_prompt:
        data["tasks"].append({"id": "benchmark_code_v2", "split": "benchmark", "category": "code_v2",
                              "input_ids": materialize_benchmark(tok.encode, "code_v2", protocol["input_tokens"]), "output_tokens": 1000})
        data["version"] = protocol["version"] + "+code_v2"
    data["protocol_sha256"], data["tokenizer_sha256"] = sha(args.protocol), sha(snapshot / "tokenizer.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{args.output}: {len(data['tasks'])} tasks, sha256 {sha(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
