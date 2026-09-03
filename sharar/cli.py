"""Command line entry point: check, describe, parity (serve arrives in Phase 5)."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sharar")
    sub = parser.add_subparsers(dest="command", required=True)
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
    p_bd = sub.add_parser("bench-decode", help="decode throughput and determinism with CUDA graphs")
    p_bd.add_argument("--prompts", required=True)
    p_bd.add_argument("--name", default="rand128")
    p_bd.add_argument("--max-new-tokens", type=int, default=64)
    p_bd.add_argument("--batches", default="1,2,4")
    p_chat = sub.add_parser("chat", help="interactive chat with streaming output and speed stats")
    p_chat.add_argument("--max-new-tokens", type=int, default=512)
    p_chat.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p_chat.add_argument("--thinking", action="store_true", help="enable the model's thinking mode")
    p_chat.add_argument("--system", default=None)
    p_chat.add_argument("--prompt", default=None, help="one-shot: answer this and exit")
    p_chat.add_argument("--no-graphs", action="store_true")
    p_cc = sub.add_parser("cache-check", help="prefix cache: checkpoint, prefix-hit, multi-turn and accounting checks")
    p_cc.add_argument("--prompts", required=True)
    p_cc.add_argument("--max-new-tokens", type=int, default=16)
    p_soak = sub.add_parser("soak", help="cycle long prompts with partial reuse and watch memory")
    p_soak.add_argument("--prompts", required=True)
    p_soak.add_argument("--rounds", type=int, default=40)
    p_soak.add_argument("--prompt-len", type=int, default=8192)
    p_prof = sub.add_parser("profile", help="kernel time breakdown for prefill or decode")
    p_prof.add_argument("--prompts", required=True)
    p_prof.add_argument("--name", default="rand2048")
    p_prof.add_argument("--mode", choices=["prefill", "decode", "both"], default="both")
    p_prof.add_argument("--top", type=int, default=30)
    args = parser.parse_args(argv)

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

        return canaries_cli(args.reference, args.max_new_tokens, args.batch, graphs=not args.eager)
    if args.command == "cache-check":
        from sharar.engine.run import cache_check_cli

        return cache_check_cli(args.prompts, args.max_new_tokens)
    if args.command == "soak":
        from sharar.engine.run import soak_cli

        return soak_cli(args.prompts, args.rounds, args.prompt_len)
    if args.command == "chat":
        from sharar.engine.chat import chat_cli

        return chat_cli(args.max_new_tokens, args.temperature, args.thinking, args.system, args.prompt, args.no_graphs)
    if args.command == "profile":
        from sharar.engine.run import profile_cli

        return profile_cli(args.prompts, args.name, args.mode, top=args.top)
    if args.command == "bench-decode":
        from sharar.engine.run import bench_decode_cli

        return bench_decode_cli(args.prompts, args.name, args.max_new_tokens, args.batches)
    if args.command == "parity":
        from sharar.parity.run import run_parity

        return run_parity(args.reference, args.prompts, args.layers)
    return 2
