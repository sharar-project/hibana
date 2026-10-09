"""Command line entry point for serving and diagnostics (HTTP benchmarks: sharar-eval's `python -m evals.speed`)."""

from __future__ import annotations

import argparse
import json
import os
import sys

from sharar import target as targets
from sharar.distributed import broadcast_pyobj, close_distributed, init_distributed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sharar")
    sub = parser.add_subparsers(dest="command", required=True)
    p_check = sub.add_parser("check", help="validate host, GPU and pinned snapshot")
    p_check.add_argument("--no-gpu", action="store_true")
    sub.add_parser("describe", help="print the pinned target facts")
    p_pp = sub.add_parser("prompts", help="write the fixed prompt set as token IDs (for generate and profile)")
    p_pp.add_argument("--out", required=True)
    p_gen = sub.add_parser("generate", help="greedy/sampled generation for one fixed prompt (eager decode)")
    p_gen.add_argument("--prompts", required=True, help="prompts.json from `sharar prompts`")
    p_gen.add_argument("--name", required=True)
    p_gen.add_argument("--max-new-tokens", type=int, default=32)
    p_gen.add_argument("--temperature", type=float, default=0.0)
    p_chat = sub.add_parser("chat", help="interactive chat with streaming output and speed stats")
    p_chat.add_argument("--max-new-tokens", type=int, default=512)
    p_chat.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p_chat.add_argument("--thinking", action="store_true", help="enable the model's thinking mode")
    p_chat.add_argument("--system", default=None)
    p_chat.add_argument("--prompt", default=None, help="one-shot: answer this and exit")
    p_chat.add_argument("--no-graphs", action="store_true")
    p_chat.add_argument("--spec", type=int, default=0, help="MTP speculative decoding with k draft tokens")
    p_serve = sub.add_parser("serve", help="OpenAI-compatible HTTP server")
    p_serve.add_argument("--target", default=None, help="pinned serving target (default: SHARAR_TARGET, else qwen38)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=30001)
    p_serve.add_argument("--proposer", choices=("mtp", "ngram", "dspark", "dflash2"), default="dflash2")
    p_serve.add_argument("--verify-state", choices=("snapshots", "reconstruct"), default="reconstruct")
    p_serve.add_argument("--draft-precision", choices=("bf16", "fp8"), default="bf16", help="parallel draft matrix precision")
    p_serve.add_argument("--spec", type=int, default=None, help="draft tokens per round (0 = off; default: SHARAR_SPEC, else the proposer's validated depth)")
    p_serve.add_argument("--draft", default=None, help="trained DFlash2/DSpark checkpoint directory to serve instead of the pinned draft (default: SHARAR_DRAFT)")
    p_serve.add_argument("--traces", default=None, help="directory for opt-in request traces (default: SHARAR_TRACES; unset disables recording)")
    p_serve.add_argument("--traces-max-mib", type=int, default=None, help="prune the oldest trace files above this size (default: SHARAR_TRACES_MAX_MIB or 32768)")
    p_serve.add_argument("--context-length", type=int, default=None, help="largest prompt plus output per request, up to 262144 (default: SHARAR_MAX_CONTEXT, else the target profile's 32768)")
    p_serve.add_argument("--max-batch", type=int, default=None, help="most requests decoded together, 1-16 (default: SHARAR_MAX_BATCH, else 4; the throughput profile sets 16)")
    p_serve.add_argument("--spec-depths", default=None, help="verify depth per batch bucket as BUCKET:DEPTH pairs, e.g. 8:1,16:0 (0: plain decode; default: SHARAR_SPEC_DEPTHS, else --spec everywhere)")
    p_serve.add_argument("--accept-default-sampled", action="store_true", default=None, help="also apply the relaxed acceptance default to sampled requests (default: SHARAR_ACCEPT_DEFAULT_SAMPLED=1, else greedy only)")
    p_serve.add_argument("--accept-default-sampled-threshold", default=None, help="with --accept-default-sampled, the K:M sampled requests get instead of --accept-default's (default: SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD)")
    p_serve.add_argument("--accept-default", default=None, help="relaxed acceptance for greedy requests that do not set accept_top_k/accept_margin, as K:M (default: SHARAR_ACCEPT_DEFAULT, else 2:1.0); 'off' keeps every request exact")
    p_serve.add_argument("--tp", type=int, default=1, help="tensor-parallel ranks, one GPU (Spark) each; rank 0 serves HTTP")
    p_serve.add_argument("--tp-rank", type=int, default=0, help="this process's tensor-parallel rank")
    p_serve.add_argument("--tp-address", default=None, help="HOST:PORT on rank 0's host where the ranks meet (required with --tp > 1)")
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
    p_train.add_argument("--anchor-chunk", type=int, default=128)
    p_train.add_argument("--no-compile", action="store_true", help="train the block forward eagerly instead of torch.compile")
    p_train.add_argument("--block", type=int, default=16, help="block size: serving's k proposals plus the anchor")
    p_train.add_argument("--gamma", type=float, default=7.0, help="position weight decay exp(-(k-1)/gamma)")
    p_train.add_argument("--selector-alpha", type=float, default=1.0)
    p_train.add_argument("--kl-alpha", type=float, default=1.0, help="weight of the KL to the target's recorded top-k logits (0 disables)")
    p_train.add_argument("--on-policy", action="store_true", help="anchor blocks at recorded rounds and train the selector on the target's answers to the drafter's own proposals")
    p_train.add_argument("--regression-traces", default=None, help="fixed general trace directory; promotion also requires no regression on its newest slice")
    p_train.add_argument("--regression-tolerance", type=float, default=0.005, help="promotion is refused when the regression set drops below base x (1 - tolerance)")
    p_train.add_argument("--continue-lr", type=float, default=None, help="learning rate used instead of --lr when --continue-from resolves to a checkpoint")
    p_train.add_argument("--train-params", default="attn", choices=("attn", "all", "selector"), help="which draft weights to train (attn: attention, norms, convolutions and selector)")
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
    if args.command == "prompts":
        from pathlib import Path

        from sharar.engine.prompts import write_prompts

        prompts = write_prompts(Path(args.out))
        for name, ids in prompts.items():
            print(f"  {name}: {len(ids)} tokens")
        return 0
    if args.command == "generate":
        from sharar.engine.run import generate_cli

        return generate_cli(args.prompts, args.name, args.max_new_tokens, args.temperature)
    if args.command == "serve":
        from sharar.engine import config as engine_config
        from sharar.proposers.factory import MAX_DEPTHS

        cfg = engine_config.EngineConfig.from_env()
        if args.tp > 1:
            if args.tp_address is None or not 0 <= args.tp_rank < args.tp:
                parser.error("--tp > 1 needs --tp-address and a --tp-rank below --tp")
            if cfg.tp_allreduce not in ("rdma", "nccl"):
                parser.error("SHARAR_TP_ALLREDUCE must be rdma or nccl")
            init_distributed(args.tp_rank, args.tp, args.tp_address, cfg.tp_allreduce)
            cfg = broadcast_pyobj(cfg)  # every rank serves with the primary's environment, not its own
        if args.spec is None:
            env = cfg.spec
            args.spec = int(env) if env not in (None, "", "auto") else {"mtp": 3}.get(args.proposer, MAX_DEPTHS[args.proposer])
        if args.draft_precision == 'fp8' and (not args.spec or args.proposer not in ('dspark', 'dflash2')):
            parser.error('--draft-precision fp8 requires an enabled parallel draft')
        traces = args.traces or cfg.traces
        draft_dir = args.draft or cfg.draft_dir
        if draft_dir and (not args.spec or args.proposer not in ('dspark', 'dflash2')):
            parser.error('--draft requires an enabled parallel draft')
        max_mib = args.traces_max_mib if args.traces_max_mib is not None else cfg.traces_max_mib
        if max_mib < 1:
            parser.error("--traces-max-mib must be a positive integer")
        from sharar.api.validation import parse_accept_default
        try:
            accept_default = parse_accept_default(args.accept_default if args.accept_default is not None else cfg.accept_default)
        except ValueError as exc:
            parser.error(str(exc))
        accept_default_sampled = bool(args.accept_default_sampled if args.accept_default_sampled is not None else cfg.accept_default_sampled)
        sampled_threshold = args.accept_default_sampled_threshold or cfg.accept_default_sampled_threshold
        if accept_default_sampled and sampled_threshold:
            try:
                accept_default_sampled = parse_accept_default(sampled_threshold)
            except ValueError as exc:
                parser.error(str(exc))
            if accept_default_sampled == (1, 0.0):
                accept_default_sampled = False
        if not args.spec:
            accept_default, accept_default_sampled = (1, 0.0), False  # no speculation, nothing to relax
        max_context = args.context_length if args.context_length is not None else cfg.max_context
        from sharar.engine.profile import MAX_CONTEXT
        if max_context is not None and not 2 <= max_context <= MAX_CONTEXT:
            parser.error(f"--context-length must be in [2, {MAX_CONTEXT}]")
        max_batch = args.max_batch if args.max_batch is not None else cfg.max_batch
        from sharar.engine.profile import MAX_BATCH
        if not 1 <= max_batch <= MAX_BATCH:
            parser.error(f"--max-batch must be in [1, {MAX_BATCH}]")
        spec_depths = args.spec_depths if args.spec_depths is not None else cfg.spec_depths
        target = args.target or cfg.target
        if args.tp > 1:
            family = targets.get(target).family
            if family not in ("qwen38", "flashnext"):
                parser.error("tensor parallel serves the 27B and Flash-Next")
            if family == "qwen38" and args.spec and args.proposer == "mtp":
                parser.error("the 27B serves tensor-parallel with the parallel drafts or n-gram speculation, not MTP")
        # One configuration for the process: the environment with the command line applied (#44).
        engine_config.install(cfg.replace(spec=str(args.spec), traces=traces, traces_max_mib=max_mib, draft_dir=draft_dir,
                                          target=target, max_context=max_context, accept_default=f"{accept_default[0]}:{accept_default[1]}",
                                          accept_default_sampled=bool(accept_default_sampled), max_batch=max_batch,
                                          accept_default_sampled_threshold=(f"{accept_default_sampled[0]}:{accept_default_sampled[1]}"
                                                                            if isinstance(accept_default_sampled, tuple) else None),
                                          spec_depths=spec_depths if args.spec else None))
        from sharar.api.server import serve
        serve(args.host, args.port, spec=args.spec, proposer=args.proposer, state_strategy=args.verify_state,
              draft_precision=args.draft_precision, traces=traces, traces_max_mib=max_mib, draft_dir=draft_dir,
              accept_default=accept_default, target=target,
              max_context=max_context, max_batch=max_batch,
              accept_default_sampled=accept_default_sampled)
        if args.tp > 1:
            close_distributed()
        return 0
    if args.command == "train":
        import os

        from pathlib import Path

        from sharar.train.trainer import StopRequest, TrainConfig, load_runtime, resolve_draft, run

        from sharar.engine.config import EngineConfig
        traces = args.traces or EngineConfig.from_env().traces
        if not traces:
            parser.error("give --traces or set SHARAR_TRACES")
        draft = args.draft
        lr = args.lr
        if draft is None and args.continue_from:
            resolved = resolve_draft(Path(args.continue_from))
            draft = str(resolved) if resolved is not None else None
            if draft and args.continue_lr is not None:
                lr = args.continue_lr
            print(f"continuing from {draft} at learning rate {lr}" if draft else f"{args.continue_from} does not resolve to a checkpoint; starting from the pinned draft", flush=True)
        cfg = TrainConfig(traces=traces, output=args.output, draft=draft, steps=args.steps, promote_to=args.promote_to,
                          sequences_per_step=args.sequences_per_step, anchors_per_sequence=args.anchors,
                          anchor_chunk=args.anchor_chunk, compile=not args.no_compile, block=args.block, gamma=args.gamma,
                          selector_alpha=args.selector_alpha, kl_alpha=args.kl_alpha, train_params=args.train_params,
                          on_policy=args.on_policy, regression_traces=args.regression_traces, regression_tolerance=args.regression_tolerance,
                          learning_rate=lr, warmup_steps=args.warmup_steps,
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

        from sharar.engine.config import EngineConfig
        directory = args.dir or EngineConfig.from_env().traces
        if not directory:
            parser.error("give --dir or set SHARAR_TRACES")
        print(json.dumps(summarize(directory), indent=2))
        return 0
    if args.command == "chat":
        from sharar.engine.chat import chat_cli

        return chat_cli(args.max_new_tokens, args.temperature, args.thinking, args.system, args.prompt, args.no_graphs, args.spec)
    if args.command == "profile":
        from sharar.engine.run import profile_cli

        return profile_cli(args.prompts, args.name, args.mode, top=args.top)
    return 2
