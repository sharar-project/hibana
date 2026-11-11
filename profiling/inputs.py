"""Frozen long prompts for the in-process profilers: 18 synthetic engineering-task families plus the three
benchmark prompts (code, design, reasoning), 10,000 input tokens each.

`profiling/workloads/family_v1.json` holds the family specifications (protocol `causal-selector-v1`); the benchmark prompts come from `hibana/benchmark_workloads.py`.
`tasks(encode)` builds them with any tokenizer's `encode`, so no generated file has to be kept anywhere:

    from profiling.inputs import tasks
    rows = tasks(SnapshotTokenizer(snapshot).encode)   # [{"id", "split", "category", "input_ids", ...}, ...]
"""
from __future__ import annotations

import json
from pathlib import Path

from hibana.benchmark_workloads import WORKLOADS, materialize as materialize_benchmark

PROTOCOL = Path(__file__).resolve().parent / "workloads" / "family_v1.json"


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


def tasks(encode, protocol_path: Path = PROTOCOL) -> list[dict]:
    """The frozen tasks, tokenized with `encode`; benchmark_code is the tool-call code prompt (`code_v1`)."""
    protocol = json.loads(protocol_path.read_text())
    benchmarks = {case: materialize_benchmark(encode, "code_v1" if case == "code" else case, protocol["input_tokens"]) for case in WORKLOADS}
    return materialize(protocol, encode, benchmarks)["tasks"]
