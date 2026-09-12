"""Command line entry point for serving, benchmarking and diagnostics."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sharar")
    sub = parser.add_subparsers(dest="command", required=True)
    from sharar.benchmark import add_arguments

    add_arguments(sub.add_parser("benchmark", help="AIPerf workload benchmark with output and decode TPS"))
    p_check = sub.add_parser("check", help="validate host, GPU and pinned snapshot")
    p_check.add_argument("--no-gpu", action="store_true")
    sub.add_parser("describe", help="print the pinned target facts")
    p_par = sub.add_parser("parity", help="compare the Sharar forward with reference dumps")
    p_par.add_argument("--reference", required=True, help="directory written by tools/reference")
    p_par.add_argument("--prompts", default="all")
    p_par.add_argument("--layers", default="all", help="'all' or comma-separated layer ids to dump-compare")
    p_pp = sub.add_parser("parity-prompts", help="write the fixed parity prompt set as token IDs")
    p_pp.add_argument("--out", required=True)
    p_gen = sub.add_parser("generate", help="greedy/sampled generation for one parity prompt (eager decode)")
    p_gen.add_argument("--prompts", required=True, help="prompts.json from parity-prompts")
    p_gen.add_argument("--name", required=True)
    p_gen.add_argument("--max-new-tokens", type=int, default=32)
    p_gen.add_argument("--temperature", type=float, default=0.0)
    p_can = sub.add_parser("canaries", help="decode every parity prompt and compare with the reference continuations")
    p_can.add_argument("--reference", required=True)
    p_can.add_argument("--max-new-tokens", type=int, default=32)
    p_can.add_argument("--batch", type=int, default=1, help="also decode the four canaries as one batch of this size")
    p_can.add_argument("--eager", action="store_true", help="skip CUDA graph capture")
    p_can.add_argument("--spec", type=int, default=0, help="MTP speculative decoding with this many draft tokens")
    p_bd = sub.add_parser("bench-decode", help="decode throughput and determinism with CUDA graphs")
    p_bd.add_argument("--prompts", required=True)
    p_bd.add_argument("--name", default="rand128")
    p_bd.add_argument("--max-new-tokens", type=int, default=64)
    p_bd.add_argument("--batches", default="1,2,4")
    p_bd.add_argument("--spec", type=int, default=0, help="benchmark MTP speculative decoding with k drafts")
    p_chat = sub.add_parser("chat", help="interactive chat with streaming output and speed stats")
    p_chat.add_argument("--max-new-tokens", type=int, default=512)
    p_chat.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p_chat.add_argument("--thinking", action="store_true", help="enable the model's thinking mode")
    p_chat.add_argument("--system", default=None)
    p_chat.add_argument("--prompt", default=None, help="one-shot: answer this and exit")
    p_chat.add_argument("--no-graphs", action="store_true")
    p_chat.add_argument("--spec", type=int, default=0, help="MTP speculative decoding with k draft tokens")
    p_cc = sub.add_parser("cache-check", help="prefix cache: checkpoint, prefix-hit, multi-turn and accounting checks")
    p_cc.add_argument("--prompts", required=True)
    p_cc.add_argument("--max-new-tokens", type=int, default=16)
    p_soak = sub.add_parser("soak", help="cycle long prompts with partial reuse and watch memory")
    p_soak.add_argument("--prompts", required=True)
    p_soak.add_argument("--rounds", type=int, default=40)
    p_soak.add_argument("--prompt-len", type=int, default=8192)
    p_sb = sub.add_parser("sched-bench", help="controlled matrix cells through the scheduler (TTFT and aggregate tok/s)")
    p_sb.add_argument("--cells", default="128,1;128,4;2048,1;2048,4;8192,1;8192,4")
    p_sb.add_argument("--output-len", type=int, default=256)
    p_sb.add_argument("--chunk-size", type=int, default=2048)
    p_sb.add_argument("--prefill-budget", type=int, default=4096)
    p_serve = sub.add_parser("serve", help="OpenAI-compatible HTTP server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=30001)
    p_serve.add_argument("--proposer", choices=("mtp", "ngram", "dspark", "dflash2"), default="dflash2")
    p_serve.add_argument("--verify-state", choices=("snapshots", "reconstruct"), default="reconstruct")
    p_serve.add_argument("--draft-precision", choices=("bf16", "fp8"), default="bf16", help="parallel draft matrix precision")
    p_serve.add_argument("--spec", type=int, default=None, help="draft tokens per round (0 = off; default: SHARAR_SPEC, else the proposer's validated depth)")
    p_serve.add_argument("--draft", default=None, help="trained DFlash2/DSpark checkpoint directory to serve instead of the pinned draft (default: SHARAR_DRAFT)")
    p_serve.add_argument("--traces", default=None, help="directory for opt-in request traces (default: SHARAR_TRACES; unset disables recording)")
    p_serve.add_argument("--traces-max-mib", type=int, default=None, help="prune the oldest trace files above this size (default: SHARAR_TRACES_MAX_MIB or 8192)")
    p_traces = sub.add_parser("traces", help="summarize a request trace directory")
    p_traces.add_argument("--dir", default=None, help="trace directory (default: SHARAR_TRACES)")
    p_train = sub.add_parser("train", help="fine-tune the DFlash2 drafter on request traces (GPU, nothing else served)")
    p_train.add_argument("--traces", default=None, help="trace directory (default: SHARAR_TRACES)")
    p_train.add_argument("--output", required=True, help="run directory: log.jsonl, manifest.json, step-*/ and best/ checkpoints")
    p_train.add_argument("--draft", default=None, help="draft checkpoint directory to start from (default: the pinned publisher draft)")
    p_train.add_argument("--steps", type=int, default=200, help="0 trains until stopped (SIGTERM), then evaluates once more")
    p_train.add_argument("--promote-to", default=None, help="symlink kept pointing at the promoted checkpoint, e.g. /runs/current")
    p_train.add_argument("--continue-from", default=None, help="a 'current' symlink: start from its checkpoint when it resolves (default: the pinned draft)")
    p_train.add_argument("--sequences-per-step", type=int, default=2)
    p_train.add_argument("--anchors", type=int, default=128, help="random anchors per sequence")
    p_train.add_argument("--anchor-chunk", type=int, default=32)
    p_train.add_argument("--block", type=int, default=16, help="block size: serving's k proposals plus the anchor")
    p_train.add_argument("--gamma", type=float, default=7.0, help="position weight decay exp(-(k-1)/gamma)")
    p_train.add_argument("--selector-alpha", type=float, default=1.0)
    p_train.add_argument("--lr", type=float, default=5e-5)
    p_train.add_argument("--warmup-steps", type=int, default=10)
    p_train.add_argument("--max-grad-norm", type=float, default=1.0)
    p_train.add_argument("--max-length", type=int, default=8192)
    p_train.add_argument("--holdout-tokens", type=int, default=20000, help="newest requests held out until they cover this many output tokens")
    p_train.add_argument("--eval-every", type=int, default=50)
    p_train.add_argument("--greedy-only", action="store_true")
    p_train.add_argument("--train-context", action="store_true", help=argparse.SUPPRESS)  # needs raw target features; not recorded
    p_train.add_argument("--seed", type=int, default=0)
    p_prof = sub.add_parser("profile", help="kernel time breakdown for prefill or decode")
    p_prof.add_argument("--prompts", required=True)
    p_prof.add_argument("--name", default="rand2048")
    p_prof.add_argument("--mode", choices=["prefill", "decode", "both"], default="both")
    p_prof.add_argument("--top", type=int, default=30)
    args = parser.parse_args(argv)

    if args.command == "benchmark":
        from sharar.benchmark import benchmark_cli

        return benchmark_cli(args)

    from sharar import config

    if args.command == "check":
        try:
            report = config.check(require_gpu=not args.no_gpu)
        except Exception as exc:  # noqa: BLE001 - report every failure plainly
            print(f"sharar check FAILED: {exc}", file=sys.stderr)
            return 1
        print("sharar check OK")
        for k, v in report.items():
            print(f"  {k}: {v}")
        return 0
    if args.command == "describe":
        facts = {k: v for k, v in vars(config).items() if k.isupper()}
        print(json.dumps(facts, indent=2, default=str))
        return 0
    if args.command == "parity-prompts":
        from pathlib import Path

        from sharar.parity.prompts import write_prompts

        prompts = write_prompts(Path(args.out))
        for name, ids in prompts.items():
            print(f"  {name}: {len(ids)} tokens")
        return 0
    if args.command == "generate":
        from sharar.engine.run import generate_cli

        return generate_cli(args.prompts, args.name, args.max_new_tokens, args.temperature)
    if args.command == "canaries":
        from sharar.engine.run import canaries_cli

        return canaries_cli(args.reference, args.max_new_tokens, args.batch, graphs=not args.eager, spec=args.spec)
    if args.command == "serve":
        import os

        from sharar.proposers.factory import MAX_DEPTHS

        if args.spec is None:
            env = os.environ.get("SHARAR_SPEC")
            args.spec = int(env) if env not in (None, "", "auto") else {"mtp": 3}.get(args.proposer, MAX_DEPTHS[args.proposer])
        if args.draft_precision == 'fp8' and (not args.spec or args.proposer not in ('dspark', 'dflash2')):
            parser.error('--draft-precision fp8 requires an enabled parallel draft')
        traces = args.traces or os.environ.get("SHARAR_TRACES") or None
        draft_dir = args.draft or os.environ.get("SHARAR_DRAFT") or None
        if draft_dir and (not args.spec or args.proposer not in ('dspark', 'dflash2')):
            parser.error('--draft requires an enabled parallel draft')
        max_mib = args.traces_max_mib if args.traces_max_mib is not None else int(os.environ.get("SHARAR_TRACES_MAX_MIB") or 8192)
        if max_mib < 1:
            parser.error("--traces-max-mib must be a positive integer")
        from sharar.api.server import serve
        serve(args.host, args.port, spec=args.spec, proposer=args.proposer, state_strategy=args.verify_state,
              draft_precision=args.draft_precision, traces=traces, traces_max_mib=max_mib, draft_dir=draft_dir)
        return 0
    if args.command == "train":
        import os

        from pathlib import Path

        from sharar.train.trainer import StopRequest, TrainConfig, load_runtime, resolve_draft, run

        traces = args.traces or os.environ.get("SHARAR_TRACES")
        if not traces:
            parser.error("give --traces or set SHARAR_TRACES")
        draft = args.draft
        if draft is None and args.continue_from:
            resolved = resolve_draft(Path(args.continue_from))
            draft = str(resolved) if resolved is not None else None
            print(f"continuing from {draft}" if draft else f"{args.continue_from} does not resolve to a checkpoint; starting from the pinned draft", flush=True)
        cfg = TrainConfig(traces=traces, output=args.output, draft=draft, steps=args.steps, promote_to=args.promote_to,
                          sequences_per_step=args.sequences_per_step, anchors_per_sequence=args.anchors,
                          anchor_chunk=args.anchor_chunk, block=args.block, gamma=args.gamma,
                          selector_alpha=args.selector_alpha, learning_rate=args.lr, warmup_steps=args.warmup_steps,
                          max_grad_norm=args.max_grad_norm, max_length=args.max_length, holdout_tokens=args.holdout_tokens,
                          eval_every=args.eval_every, greedy_only=args.greedy_only,
                          train_context=args.train_context, seed=args.seed)
        draft, pinned = load_runtime(cfg)
        stop = StopRequest()
        stop.install()
        try:
            result = run(cfg, draft, pinned, stop=stop)
        finally:
            stop.restore()
        print(json.dumps(result))
        return 0
    if args.command == "traces":
        import os

        from sharar.engine.traces import summarize

        directory = args.dir or os.environ.get("SHARAR_TRACES")
        if not directory:
            parser.error("give --dir or set SHARAR_TRACES")
        print(json.dumps(summarize(directory), indent=2))
        return 0
    if args.command == "sched-bench":
        from sharar.engine.run import sched_bench_cli

        return sched_bench_cli(args.cells, args.output_len, args.chunk_size, args.prefill_budget)
    if args.command == "cache-check":
        from sharar.engine.run import cache_check_cli

        return cache_check_cli(args.prompts, args.max_new_tokens)
    if args.command == "soak":
        from sharar.engine.run import soak_cli

        return soak_cli(args.prompts, args.rounds, args.prompt_len)
    if args.command == "chat":
        from sharar.engine.chat import chat_cli

        return chat_cli(args.max_new_tokens, args.temperature, args.thinking, args.system, args.prompt, args.no_graphs, args.spec)
    if args.command == "profile":
        from sharar.engine.run import profile_cli

        return profile_cli(args.prompts, args.name, args.mode, top=args.top)
    if args.command == "bench-decode":
        from sharar.engine.run import bench_decode_cli

        return bench_decode_cli(args.prompts, args.name, args.max_new_tokens, args.batches, spec=args.spec)
    if args.command == "parity":
        from sharar.parity.run import run_parity

        return run_parity(args.reference, args.prompts, args.layers)
    return 2
