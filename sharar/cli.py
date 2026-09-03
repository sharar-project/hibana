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
    if args.command == "parity":
        from sharar.parity.run import run_parity

        return run_parity(args.reference, args.prompts, args.layers)
    return 2
