#!/usr/bin/env python3
"""Reproducible text-only serving benchmark for a single GB10 system.

The benchmark intentionally uses only Python's standard library so it can run
on the DGX Spark host without modifying the SGLang container or environment.
Performance requests use SGLang's native ``/generate`` endpoint with exact
token-id input lengths. Quality canaries use the OpenAI-compatible chat API.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import http.client
import json
import math
import os
import re
from pathlib import Path
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "RadixArk/Qwen3.8-27B-NVFP4"
FILLER_TEXT = (
    "The quick brown fox jumps over the lazy dog while a local inference "
    "server processes a deterministic text benchmark. "
)


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def default_results_root() -> Path:
    state_home = Path(
        os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
    )
    results = Path(
        os.environ.get("SHARAR_RESULTS", state_home / "sharar" / "gb10")
    )
    return results / "controlled"


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def http_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body is not None else "GET",
    )
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.decode("utf-8", errors="replace")


def tokenize(base_url: str, model: str, prompt: str) -> list[int]:
    response = http_json(
        f"{base_url}/v1/tokenize",
        payload={
            "model": model,
            "prompt": prompt,
            "add_special_tokens": False,
        },
    )
    tokens = response.get("tokens") if isinstance(response, dict) else None
    if not isinstance(tokens, list) or not tokens:
        raise RuntimeError(f"Unexpected tokenize response: {response!r}")
    return [int(token) for token in tokens]


def exact_prompt_ids(
    base_url: str,
    model: str,
    target_length: int,
    request_key: str,
    filler_ids: list[int],
) -> list[int]:
    unique_ids = tokenize(
        base_url,
        model,
        f"Unique benchmark request {request_key}. Continue processing text: ",
    )
    if len(unique_ids) >= target_length:
        return unique_ids[:target_length]
    repeats = math.ceil((target_length - len(unique_ids)) / len(filler_ids))
    return (unique_ids + filler_ids * repeats)[:target_length]


def flush_cache(base_url: str) -> None:
    response = http_json(f"{base_url}/flush_cache", payload={}, timeout=30.0)
    if not isinstance(response, str) or "Cache flushed" not in response:
        raise RuntimeError(f"Cache flush did not succeed: {response!r}")


def read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            key, raw = line.split(":", 1)
            fields = raw.strip().split()
            if fields:
                values[key] = int(fields[0])
    return values


def query_power_watts() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        value = result.stdout.strip().splitlines()[0]
        return float(value) if value not in {"", "[N/A]", "N/A"} else None
    except (FileNotFoundError, IndexError, subprocess.SubprocessError, ValueError):
        return None


class ResourceSampler:
    def __init__(self, interval: float, minimum_available_gib: float) -> None:
        self.interval = interval
        self.minimum_available_kib = int(minimum_available_gib * 1024 * 1024)
        self.samples: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.low_memory_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        low_samples = 0
        next_power_sample = 0.0
        last_power: float | None = None
        while not self.stop_event.is_set():
            now = time.time()
            meminfo = read_meminfo()
            if now >= next_power_sample:
                last_power = query_power_watts()
                next_power_sample = now + 1.0
            available = meminfo.get("MemAvailable", 0)
            self.samples.append(
                {
                    "timestamp": now,
                    "mem_available_kib": available,
                    "mem_free_kib": meminfo.get("MemFree", 0),
                    "cached_kib": meminfo.get("Cached", 0),
                    "swap_free_kib": meminfo.get("SwapFree", 0),
                    "power_watts": last_power,
                }
            )
            low_samples = low_samples + 1 if available < self.minimum_available_kib else 0
            if low_samples >= 3:
                self.low_memory_event.set()
            self.stop_event.wait(self.interval)

    def __enter__(self) -> "ResourceSampler":
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)

    def summary(self) -> dict[str, Any]:
        memory = [sample["mem_available_kib"] for sample in self.samples]
        power = [
            sample["power_watts"]
            for sample in self.samples
            if sample["power_watts"] is not None
        ]
        return {
            "sample_count": len(self.samples),
            "minimum_mem_available_gib": min(memory) / 1024 / 1024 if memory else None,
            "maximum_mem_available_gib": max(memory) / 1024 / 1024 if memory else None,
            "mean_power_watts": statistics.fmean(power) if power else None,
            "maximum_power_watts": max(power) if power else None,
            "low_memory_triggered": self.low_memory_event.is_set(),
        }


def stream_generate(
    base_url: str,
    input_ids: list[int],
    output_length: int,
    timeout: float,
    request_key: str,
) -> dict[str, Any]:
    host_and_port = base_url.removeprefix("http://").removeprefix("https://")
    if "/" in host_and_port:
        host_and_port = host_and_port.split("/", 1)[0]
    host, port_text = host_and_port.rsplit(":", 1)
    connection = http.client.HTTPConnection(host, int(port_text), timeout=timeout)
    body = json.dumps(
        {
            "rid": request_key,
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": output_length,
                "temperature": 0,
                "ignore_eos": True,
            },
            "stream": True,
        }
    )
    start = time.perf_counter()
    first_token_at: float | None = None
    last_meta: dict[str, Any] = {}
    final_text = ""
    error: str | None = None
    try:
        connection.request(
            "POST",
            "/generate",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(
                f"HTTP {response.status}: {response.read().decode(errors='replace')}"
            )
        while True:
            raw_line = response.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            meta = chunk.get("meta_info") or {}
            completion_tokens = int(meta.get("completion_tokens") or 0)
            if completion_tokens > 0 and first_token_at is None:
                first_token_at = time.perf_counter()
            last_meta = meta
            final_text = chunk.get("text", final_text)
    except Exception as exc:  # Save request failures in the raw results.
        error = f"{type(exc).__name__}: {exc}"
    finally:
        connection.close()
    end = time.perf_counter()
    completion_tokens = int(last_meta.get("completion_tokens") or 0)
    ttft = first_token_at - start if first_token_at is not None else None
    decode_seconds = end - first_token_at if first_token_at is not None else None
    decode_tps = None
    if decode_seconds is not None and decode_seconds > 0 and completion_tokens > 1:
        decode_tps = (completion_tokens - 1) / decode_seconds
    return {
        "request_key": request_key,
        "success": error is None and completion_tokens == output_length,
        "error": error,
        "prompt_tokens": int(last_meta.get("prompt_tokens") or len(input_ids)),
        "completion_tokens": completion_tokens,
        "ttft_seconds": ttft,
        "e2e_seconds": end - start,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": decode_tps,
        "finish_reason": last_meta.get("finish_reason"),
        "cached_tokens": int(last_meta.get("cached_tokens") or 0),
        "response_character_count": len(final_text),
        "response_sha256": hashlib.sha256(final_text.encode("utf-8")).hexdigest(),
        "spec_accept_rate": last_meta.get("spec_accept_rate"),
        "spec_accept_length": last_meta.get("spec_accept_length"),
        "spec_num_correct_drafts": last_meta.get("spec_num_correct_drafts"),
        "spec_num_proposed_drafts": last_meta.get("spec_num_proposed_drafts"),
        "spec_verify_count": last_meta.get("spec_verify_ct"),
        "server_e2e_seconds": last_meta.get("e2e_latency"),
    }


def run_batch(
    *,
    base_url: str,
    model: str,
    filler_ids: list[int],
    input_length: int,
    output_length: int,
    concurrency: int,
    repetition: int,
    phase: str,
    timeout: float,
    sample_interval: float,
    minimum_available_gib: float,
) -> dict[str, Any]:
    flush_cache(base_url)
    request_ids = [
        f"i{input_length}-o{output_length}-c{concurrency}-{phase}{repetition}-r{index}"
        for index in range(concurrency)
    ]
    prompts = [
        exact_prompt_ids(base_url, model, input_length, key, filler_ids)
        for key in request_ids
    ]
    started = time.perf_counter()
    with ResourceSampler(sample_interval, minimum_available_gib) as sampler:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    stream_generate,
                    base_url,
                    prompt,
                    output_length,
                    timeout,
                    request_id,
                )
                for prompt, request_id in zip(prompts, request_ids)
            ]
            requests = [future.result() for future in as_completed(futures)]
    batch_seconds = time.perf_counter() - started
    if sampler.low_memory_event.is_set():
        raise RuntimeError(
            f"MemAvailable remained below {minimum_available_gib:.1f} GiB; "
            "stopping the benchmark before the next batch"
        )
    successful = [request for request in requests if request["success"]]
    ttfts = [request["ttft_seconds"] for request in successful]
    latencies = [request["e2e_seconds"] for request in successful]
    decode_rates = [
        request["decode_tokens_per_second"]
        for request in successful
        if request["decode_tokens_per_second"] is not None
    ]
    total_completion_tokens = sum(
        request["completion_tokens"] for request in successful
    )
    spec_accept_rates = [
        float(request["spec_accept_rate"])
        for request in successful
        if request["spec_accept_rate"] is not None
    ]
    spec_accept_lengths = [
        float(request["spec_accept_length"])
        for request in successful
        if request["spec_accept_length"] is not None
    ]
    return {
        "phase": phase,
        "input_length": input_length,
        "output_length": output_length,
        "concurrency": concurrency,
        "repetition": repetition,
        "batch_seconds": batch_seconds,
        "successful_requests": len(successful),
        "failed_requests": concurrency - len(successful),
        "aggregate_output_tokens_per_second": total_completion_tokens / batch_seconds,
        "median_ttft_seconds": statistics.median(ttfts) if ttfts else None,
        "p95_ttft_seconds": percentile(ttfts, 0.95),
        "median_e2e_seconds": statistics.median(latencies) if latencies else None,
        "p95_e2e_seconds": percentile(latencies, 0.95),
        "median_request_decode_tokens_per_second": (
            statistics.median(decode_rates) if decode_rates else None
        ),
        "median_spec_accept_rate": (
            statistics.median(spec_accept_rates) if spec_accept_rates else None
        ),
        "median_spec_accept_length": (
            statistics.median(spec_accept_lengths) if spec_accept_lengths else None
        ),
        "resources": sampler.summary(),
        "resource_samples": sampler.samples,
        "requests": requests,
    }


def run_command(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return {
            "command": command,
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


def collect_manifest(
    base_url: str,
    model: str,
    container: str,
    repo: Path,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "benchmark_arguments": arguments,
        "model_endpoint": http_json(f"{base_url}/v1/models"),
        "health": http_json(f"{base_url}/health"),
        "host": {
            "uname": run_command(["uname", "-a"]),
            "lscpu": run_command(["lscpu"]),
            "nvidia_smi": run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,power.draw",
                    "--format=csv,noheader",
                ]
            ),
            "meminfo": read_meminfo(),
        },
        "repository": {
            "path": str(repo),
            "head": run_command(["git", "rev-parse", "HEAD"], cwd=repo),
            "status": run_command(["git", "status", "--short"], cwd=repo),
        },
        "container": run_command(["docker", "inspect", container]),
        "model": model,
    }


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = http_json(
        f"{base_url}/v1/chat/completions",
        payload={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    content = ""
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content") or ""
    return {"elapsed_seconds": elapsed, "content": content, "response": response}


def run_quality_canaries(
    base_url: str,
    model: str,
    timeout: float,
) -> list[dict[str, Any]]:
    needle = "SPARK-NEEDLE-7291"
    filler = "A quiet workshop contains ordinary numbered boxes. " * 600
    cases: list[dict[str, Any]] = [
        {
            "name": "exact_instruction",
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly SPARKLANG_OK and nothing else.",  # keep: frozen baseline hashes depend on this text
                }
            ],
            "max_tokens": 16,
            "check": lambda text: text.strip() == "SPARKLANG_OK",
        },
        {
            "name": "arithmetic",
            "messages": [
                {
                    "role": "user",
                    "content": "What is 17 + 25? Reply with only the integer.",
                }
            ],
            "max_tokens": 16,
            "check": lambda text: text.strip() == "42",
        },
        {
            "name": "json",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        'Return only valid JSON with keys "device" and "model". '
                        'Their values must be "GB10" and "Qwen3.8-27B".'
                    ),
                }
            ],
            "max_tokens": 64,
            "check": lambda text: json.loads(text.strip())
            == {"device": "GB10", "model": "Qwen3.8-27B"},
        },
        {
            "name": "long_context_needle",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        filler
                        + f"The unique access code is {needle}. "
                        + filler
                        + "What is the unique access code? Reply with only the code."
                    ),
                }
            ],
            "max_tokens": 32,
            "check": lambda text: text.strip() == needle,
        },
    ]
    outputs: list[dict[str, Any]] = []
    for case in cases:
        flush_cache(base_url)
        result = chat_completion(
            base_url,
            model,
            case["messages"],
            case["max_tokens"],
            timeout,
        )
        try:
            passed = bool(case["check"](result["content"]))
            check_error = None
        except Exception as exc:
            passed = False
            check_error = f"{type(exc).__name__}: {exc}"
        outputs.append(
            {
                "name": case["name"],
                "passed": passed,
                "check_error": check_error,
                **result,
            }
        )
    return outputs


def write_summary_csv(path: Path, batches: list[dict[str, Any]]) -> None:
    fieldnames = [
        "phase",
        "input_length",
        "output_length",
        "concurrency",
        "repetition",
        "successful_requests",
        "failed_requests",
        "batch_seconds",
        "aggregate_output_tokens_per_second",
        "median_ttft_seconds",
        "p95_ttft_seconds",
        "median_e2e_seconds",
        "p95_e2e_seconds",
        "median_request_decode_tokens_per_second",
        "median_spec_accept_rate",
        "median_spec_accept_length",
        "minimum_mem_available_gib",
        "mean_power_watts",
        "maximum_power_watts",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for batch in batches:
            resources = batch["resources"]
            writer.writerow(
                {
                    key: (
                        resources.get(key)
                        if key
                        in {
                            "minimum_mem_available_gib",
                            "mean_power_watts",
                            "maximum_power_watts",
                        }
                        else batch.get(key)
                    )
                    for key in fieldnames
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--container", default="sparklang-gb10-qwen38-nvfp4")
    parser.add_argument(
        "--label",
        default="gb10-engine-production",
        help="Filesystem-safe variant label included in the result directory",
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_results_root(),
    )
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[128, 2048, 8192])
    parser.add_argument("--output-length", type=int, default=256)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--minimum-available-gib", type=float, default=16.0)
    parser.add_argument("--skip-quality", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.base_url.startswith("https://"):
        raise SystemExit("This local-only benchmark currently expects an http:// endpoint")
    if any(value <= 0 for value in args.input_lengths + args.concurrency):
        raise SystemExit("Input lengths and concurrency values must be positive")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", args.label):
        raise SystemExit("--label must contain only letters, digits, dot, underscore or dash")
    run_dir = args.results_root / f"controlled-27b-{args.label}-{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    manifest = collect_manifest(
        args.base_url,
        args.model,
        args.container,
        args.repo,
        arguments,
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    filler_ids = tokenize(args.base_url, args.model, FILLER_TEXT)
    batches: list[dict[str, Any]] = []
    raw_path = run_dir / "batches.jsonl"
    with raw_path.open("a", encoding="utf-8") as raw_file:
        for input_length in args.input_lengths:
            for concurrency in args.concurrency:
                total_runs = args.warmups + args.repetitions
                for index in range(total_runs):
                    phase = "warmup" if index < args.warmups else "measured"
                    repetition = index if phase == "warmup" else index - args.warmups
                    print(
                        f"[{dt.datetime.now().isoformat(timespec='seconds')}] "
                        f"{phase} input={input_length} output={args.output_length} "
                        f"concurrency={concurrency} repetition={repetition}",
                        flush=True,
                    )
                    batch = run_batch(
                        base_url=args.base_url,
                        model=args.model,
                        filler_ids=filler_ids,
                        input_length=input_length,
                        output_length=args.output_length,
                        concurrency=concurrency,
                        repetition=repetition,
                        phase=phase,
                        timeout=args.timeout,
                        sample_interval=args.sample_interval,
                        minimum_available_gib=args.minimum_available_gib,
                    )
                    raw_file.write(json.dumps(batch, sort_keys=True) + "\n")
                    raw_file.flush()
                    if phase == "measured":
                        batches.append(batch)
                    median_ttft = batch["median_ttft_seconds"]
                    median_ttft_text = (
                        "n/a" if median_ttft is None else f"{median_ttft:.3f}s"
                    )
                    print(
                        f"  aggregate={batch['aggregate_output_tokens_per_second']:.2f} tok/s "
                        f"median_ttft={median_ttft_text} "
                        f"min_available={batch['resources']['minimum_mem_available_gib']:.2f} GiB",
                        flush=True,
                    )
    write_summary_csv(run_dir / "summary.csv", batches)
    benchmark_failed = any(batch["failed_requests"] for batch in batches)
    if not args.skip_quality:
        quality = run_quality_canaries(args.base_url, args.model, args.timeout)
        (run_dir / "quality.json").write_text(
            json.dumps(quality, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        passed = sum(case["passed"] for case in quality)
        print(f"Quality canaries: {passed}/{len(quality)} passed", flush=True)
        benchmark_failed = benchmark_failed or passed != len(quality)
    print(f"Results: {run_dir}", flush=True)
    return 1 if benchmark_failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
        raise SystemExit(f"benchmark failed: {type(exc).__name__}: {exc}") from exc
