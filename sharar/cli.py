"""The `sharar` command: serve a model, talk to it, train its drafter, measure it.

    sharar serve 27b                 # Qwen3.8-27B on 127.0.0.1:30001, in the foreground (Ctrl-C stops it)
    sharar serve flashnext --detach  # Qwen3.8-Flash-Next on :30004 as a background service (sharar stop, sharar logs -f)
    sharar serve 27b --tp 2          # tensor-parallel on two DGX Sparks
    sharar chat                      # chat with whichever target is serving
    sharar launch claude             # Claude Code (or opencode, aider) set up for the serving target
    sharar status | stop | logs
    sharar train                     # train the 27B's drafter on your recorded traffic (Ctrl-C finishes and promotes)
    sharar draft list | use RUN | pinned | reload

Settings: command-line flags, then SHARAR_* environment variables, then the config file (SHARAR_CONFIG_FILE, else
config/gb10.env in the checkout, else ~/.config/sharar/sharar.env), then the target's and the profile's defaults
(sharar/launch/settings.py; docs/operations.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sharar.launch.settings import PROFILE_NAMES, TARGET_CHOICES

PROFILE_HELP = {"exact": "the target's own distribution: exact acceptance, no lossy weight copies (benchmarks)",
                "fast": "the default: every measured speed option, lossy ones included (docs/operations.md#serving-profiles)",
                "throughput": "fast with more requests at once (27B 16, Flash-Next 128) for evals and many agents"}


def add_engine_args(p: argparse.ArgumentParser) -> None:
    """`sharar engine`: the server itself, with every setting explicit (what `sharar serve` and the ranks run)."""
    p.add_argument("--target", default=None, help="pinned serving target (default: SHARAR_TARGET, else qwen38)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30001)
    p.add_argument("--proposer", choices=("mtp", "dflash2"), default="dflash2")
    p.add_argument("--verify-state", choices=("snapshots", "reconstruct"), default="reconstruct")
    p.add_argument("--draft-precision", choices=("bf16", "fp8"), default="bf16", help="parallel draft matrix precision")
    p.add_argument("--spec", type=int, default=None, help="draft tokens per round (0 = off; default: SHARAR_SPEC, else the proposer's validated depth)")
    p.add_argument("--draft", default=None, help="trained DFlash2 checkpoint directory to serve instead of the pinned draft (default: SHARAR_DRAFT)")
    p.add_argument("--traces", default=None, help="directory for opt-in request traces (default: SHARAR_TRACES; unset disables recording)")
    p.add_argument("--traces-max-mib", type=int, default=None, help="prune the oldest trace files above this size (default: SHARAR_TRACES_MAX_MIB or 32768)")
    p.add_argument("--context-length", type=int, default=None, help="largest prompt plus output per request, up to 262144 (default: SHARAR_MAX_CONTEXT, else the target profile's 32768)")
    p.add_argument("--max-batch", type=int, default=None, help="most requests decoded together, 1-128 (default: SHARAR_MAX_BATCH, else 4; the throughput profile sets 16)")
    p.add_argument("--spec-depths", default=None, help="verify depth per batch bucket as BUCKET:DEPTH pairs, e.g. 8:1,16:0 (0: plain decode; default: SHARAR_SPEC_DEPTHS, else --spec everywhere)")
    p.add_argument("--accept-default-sampled", action="store_true", default=None, help="also apply the relaxed acceptance default to sampled requests (default: SHARAR_ACCEPT_DEFAULT_SAMPLED=1, else greedy only)")
    p.add_argument("--accept-default-sampled-threshold", default=None, help="with --accept-default-sampled, the K:M sampled requests get instead of --accept-default's (default: SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD)")
    p.add_argument("--accept-default", default=None, help="relaxed acceptance for greedy requests that do not set accept_top_k/accept_margin, as K:M (default: SHARAR_ACCEPT_DEFAULT, else 2:1.0); 'off' keeps every request exact")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel ranks, one GPU (Spark) each; rank 0 serves HTTP")
    p.add_argument("--tp-rank", type=int, default=0, help="this process's tensor-parallel rank")
    p.add_argument("--tp-address", default=None, help="HOST:PORT on rank 0's host where the ranks meet (required with --tp > 1)")


def add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--traces", default=None, help="trace directory (default: SHARAR_TRACES, else $SHARAR_RESULTS/traces)")
    p.add_argument("--output", default=None, help="run directory (default: $SHARAR_RESULTS/draft-runs/run-<time>): log.jsonl, manifest.json, step-*/ and best/ checkpoints")
    p.add_argument("--draft", default=None, help="draft checkpoint directory to start from (default: the pinned publisher draft)")
    p.add_argument("--steps", type=int, default=0, help="0 trains until stopped (SIGTERM), then evaluates once more")
    p.add_argument("--promote-to", default=None, help="symlink kept pointing at the promoted checkpoint (default: <draft-runs>/current; 'none' promotes nothing)")
    p.add_argument("--continue-from", default=None, help="a 'current' symlink: start from its checkpoint when it resolves (default: <draft-runs>/current; 'none' starts from the pinned draft)")
    p.add_argument("--sequences-per-step", type=int, default=2)
    p.add_argument("--anchors", type=int, default=128, help="random anchors per sequence")
    p.add_argument("--anchor-chunk", type=int, default=128)
    p.add_argument("--no-compile", action="store_true", help="train the block forward eagerly instead of torch.compile")
    p.add_argument("--block", type=int, default=16, help="block size: serving's k proposals plus the anchor")
    p.add_argument("--gamma", type=float, default=7.0, help="position weight decay exp(-(k-1)/gamma)")
    p.add_argument("--selector-alpha", type=float, default=1.0)
    p.add_argument("--kl-alpha", type=float, default=1.0, help="weight of the KL to the target's recorded top-k logits (0 disables)")
    p.add_argument("--on-policy", action="store_true", help="anchor blocks at recorded rounds and train the selector on the target's answers to the drafter's own proposals")
    p.add_argument("--regression-traces", default=None, help="fixed general trace directory; promotion also requires no regression on its newest slice (default: $SHARAR_RESULTS/draft-regression when it exists)")
    p.add_argument("--regression-tolerance", type=float, default=0.005, help="promotion is refused when the regression set drops below base x (1 - tolerance)")
    p.add_argument("--continue-lr", type=float, default=3e-5, help="learning rate used instead of --lr when --continue-from resolves to a checkpoint")
    p.add_argument("--train-params", default="attn", choices=("attn", "all", "selector"), help="which draft weights to train (attn: attention, norms, convolutions and selector)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--holdout-tokens", type=int, default=20000, help="newest requests held out until they cover this many output tokens")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--greedy-only", action="store_true")
    p.add_argument("--train-context", action="store_true", help=argparse.SUPPRESS)  # needs raw target features; not recorded
    p.add_argument("--seed", type=int, default=0)


def run_engine(args, parser) -> int:
    """Serve with explicit settings (the environment holds the rest; `sharar serve` prepares both)."""
    from sharar import target as targets
    from sharar.distributed import broadcast_pyobj, close_distributed, init_distributed
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
    if args.draft_precision == 'fp8' and (not args.spec or args.proposer != 'dflash2'):
        parser.error('--draft-precision fp8 requires an enabled parallel draft')
    traces = args.traces or cfg.traces
    draft_dir = args.draft or cfg.draft_dir
    if draft_dir and (not args.spec or args.proposer != 'dflash2'):
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
            parser.error("the 27B serves tensor-parallel with DFlash2, not MTP")
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


def run_train(args, parser) -> int:
    """Train the drafter with explicit settings (`sharar train` fills the defaults)."""
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


def engine_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sharar engine", description=add_engine_args.__doc__)
    add_engine_args(p)
    return p


def add_target(p: argparse.ArgumentParser, help: str = "27b (Qwen3.8-27B) or flashnext (Qwen3.8-Flash-Next)") -> None:
    p.add_argument("target", nargs="?", choices=TARGET_CHOICES, metavar="TARGET", help=help)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sharar", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("serve", help="serve a model (in the foreground; --detach for a background service)")
    add_target(p, "27b or flashnext (default: SHARAR_TARGET, else 27b)")
    g = p.add_mutually_exclusive_group()
    for name in PROFILE_NAMES:
        g.add_argument(f"--{name}", dest="profile", action="store_const", const=name, help=PROFILE_HELP[name])
    p.add_argument("-d", "--detach", action="store_true", help="run as a user systemd service (sharar stop | status | logs)")
    p.add_argument("--tp", type=int, choices=(1, 2), default=1, help="2: tensor-parallel on two DGX Sparks (docs/tensor-parallel.md)")
    p.add_argument("--port", type=int, default=None, help="default: 30001 for the 27B, 30004 for Flash-Next")
    p.add_argument("--host", default=None, help="default: 127.0.0.1")
    p.add_argument("--context-length", type=int, default=None, help="largest prompt plus output per request (default 262144)")
    p.add_argument("--max-batch", type=int, default=None, help="most requests decoded together, 1-128 (default: the profile's)")
    p.add_argument("--no-traces", action="store_true", help="do not record requests for drafter training")
    p.add_argument("--no-sessions", action="store_true", help="do not store parked sessions on disk")
    d = p.add_mutually_exclusive_group()
    d.add_argument("--pinned", action="store_true", help="serve the publisher's draft even if a trained one was promoted")
    d.add_argument("--draft", default=None, help="serve this trained draft checkpoint directory")
    p.add_argument("--proposer", choices=("mtp", "dflash2"), default=None, help=argparse.SUPPRESS)
    p.add_argument("--spec", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--draft-precision", choices=("bf16", "fp8"), default=None, help=argparse.SUPPRESS)
    p.add_argument("--spec-depths", default=None, help=argparse.SUPPRESS)
    p.add_argument("--accept-default", default=None, help=argparse.SUPPRESS)
    p.add_argument("--service", default=None, help=argparse.SUPPRESS)  # the systemd unit name (tools that serve their own arm)
    p.add_argument("--no-fetch", action="store_true", help=argparse.SUPPRESS)  # the detached child: the parent fetched

    sub.add_parser("stop", help="stop a detached server").add_argument("target", nargs="?", choices=TARGET_CHOICES, metavar="TARGET")
    sub.add_parser("status", help="what is serving, training and the current draft").add_argument("target", nargs="?", choices=TARGET_CHOICES, metavar="TARGET")
    p = sub.add_parser("logs", help="the log of a detached server (or of training)")
    add_target(p)
    p.add_argument("--train", action="store_true")
    p.add_argument("-n", "--lines", type=int, default=50)
    p.add_argument("-f", "--follow", action="store_true")

    p = sub.add_parser("chat", help="chat with the serving model in the terminal")
    add_target(p, "default: whichever target is serving")
    p.add_argument("--think", action="store_true", help="enable the model's thinking mode")
    p.add_argument("--show-thinking", action="store_true", help="print thinking content as it streams")
    p.add_argument("--system", default=None)
    p.add_argument("--max-tokens", type=int, default=8192, help="reply budget, thinking included")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--no-trace", action="store_true", help="do not record this conversation")
    p.add_argument("--timeout", type=float, default=600)

    from sharar.launch.ctl import HARNESSES
    p = sub.add_parser("launch", help="start a coding agent (claude, opencode, aider) set up for the serving model; `show` prints the settings for others")
    p.add_argument("harness", choices=HARNESSES)
    p.add_argument("--target", choices=TARGET_CHOICES, default=None, help="before the agent name; default: the target that is serving")
    p.add_argument("args", nargs=argparse.REMAINDER, help="passed to the agent unchanged")

    p = sub.add_parser("train", help="train the 27B's drafter on recorded traffic (foreground; Ctrl-C finishes and promotes)")
    p.add_argument("action", nargs="?", choices=("start", "stop", "status"), default="start")
    p.add_argument("-d", "--detach", action="store_true", help="run as a user systemd service (sharar train stop finishes it)")
    p.add_argument("--keep-serving", action="store_true", help="train while a server runs (only on a machine with another GPU serving)")
    add_train_args(p)

    p = sub.add_parser("draft", help="trained drafts: list, use RUN, pinned, reload (swaps the running server's draft in place)")
    p.add_argument("action", choices=("list", "use", "pinned", "reload"))
    p.add_argument("run", nargs="?", help="with use: a run name from `sharar draft list`")
    p = sub.add_parser("traces", help="summarize the recorded traces")
    p.add_argument("--dir", default=None, help="default: SHARAR_TRACES, else $SHARAR_RESULTS/traces")
    p = sub.add_parser("verify", help="check a running server: health, pinned model, a short answer")
    add_target(p, "default: whichever target is serving")
    p.add_argument("--url", default=None)

    dev = sub.add_parser("dev", help="engine diagnostics (in-process: they load the model themselves)")
    dsub = dev.add_subparsers(dest="dev_command", required=True, metavar="TOOL")
    p = dsub.add_parser("check", help="validate host, GPU and pinned snapshot")
    p.add_argument("--no-gpu", action="store_true")
    dsub.add_parser("describe", help="print the pinned target facts")
    p = dsub.add_parser("prompts", help="write the fixed prompt set as token IDs (for generate and profile)")
    p.add_argument("--out", required=True)
    p = dsub.add_parser("generate", help="greedy/sampled generation for one fixed prompt (eager decode)")
    p.add_argument("--prompts", required=True, help="prompts.json from `sharar dev prompts`")
    p.add_argument("--name", required=True)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p = dsub.add_parser("repl", help="chat with an in-process engine (no server), with speed stats")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--system", default=None)
    p.add_argument("--prompt", default=None, help="one-shot: answer this and exit")
    p.add_argument("--no-graphs", action="store_true")
    p.add_argument("--spec", type=int, default=0, help="MTP speculative decoding with k draft tokens")
    p = dsub.add_parser("profile", help="kernel time breakdown for prefill or decode")
    p.add_argument("--prompts", required=True)
    p.add_argument("--name", default="rand2048")
    p.add_argument("--mode", choices=["prefill", "decode", "both"], default="both")
    p.add_argument("--top", type=int, default=30)
    for name, module in (("tp-comm", "profiling.tp_comm"), ("tp-draft-head", "profiling.tp_draft_head")):
        p = dsub.add_parser(name, help=f"two Sparks: {module} on both ranks")
        p.add_argument("args", nargs=argparse.REMAINDER)
        p.set_defaults(module=module)

    add_engine_args(sub.add_parser("engine"))  # hidden below: what serve and the ranks run
    sub._choices_actions = [a for a in sub._choices_actions if a.dest != "engine"]  # not listed in --help
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from sharar.launch import ctl, lxcfs, run

    if args.command in ("serve", "train", "engine", "dev"):
        lxcfs.reexec_if_needed()  # before anything imports torch
    if args.command == "serve":
        return run.serve(args)
    if args.command == "engine":
        return run_engine(args, parser)
    if args.command == "stop":
        return run.stop(args)
    if args.command == "status":
        return run.status(args)
    if args.command == "logs":
        return run.logs(args)
    if args.command == "train":
        return run.train(args, parser)
    if args.command in ("chat", "launch", "verify"):
        s = run.serving(args.target)
        if s is None:
            sys.exit("nothing is serving: sharar serve 27b (or flashnext) first")
        if args.command == "chat":
            return ctl.cmd_chat(args, s)
        if args.command == "launch":
            return ctl.cmd_launch(args, s)
        return ctl.verify(args.url or s.url, s)
    if args.command == "draft":
        return ctl.cmd_draft(args, run.resolve("27b"))
    if args.command == "traces":
        from sharar.engine.traces import summarize
        directory = args.dir or run.resolve("27b").traces
        if not directory or not Path(directory).is_dir():
            sys.exit(f"no traces at {directory} (serve with traces on, the default, or give --dir)")
        print(json.dumps(summarize(str(directory)), indent=2))
        return 0
    return dev(args, parser)


def dev(args, parser) -> int:
    from sharar import config

    if args.dev_command == "check":
        try:
            report = config.check(require_gpu=not args.no_gpu)
        except Exception as exc:  # noqa: BLE001 - report every failure plainly
            print(f"sharar check FAILED: {exc}", file=sys.stderr)
            return 1
        print("sharar check OK")
        for k, v in report.items():
            print(f"  {k}: {v}")
        return 0
    if args.dev_command == "describe":
        facts = {k: v for k, v in vars(config).items() if k.isupper()}
        print(json.dumps(facts, indent=2, default=str))
        return 0
    if args.dev_command == "prompts":
        from pathlib import Path

        from sharar.engine.prompts import write_prompts

        for name, ids in write_prompts(Path(args.out)).items():
            print(f"  {name}: {len(ids)} tokens")
        return 0
    if args.dev_command == "generate":
        from sharar.engine.run import generate_cli

        return generate_cli(args.prompts, args.name, args.max_new_tokens, args.temperature)
    if args.dev_command == "repl":
        from sharar.engine.chat import chat_cli

        return chat_cli(args.max_new_tokens, args.temperature, args.thinking, args.system, args.prompt, args.no_graphs, args.spec)
    if args.dev_command == "profile":
        from sharar.engine.run import profile_cli

        return profile_cli(args.prompts, args.name, args.mode, top=args.top)
    if args.dev_command in ("tp-comm", "tp-draft-head"):
        from sharar.launch import tp
        from sharar.launch.settings import resolve

        return tp.diagnostic(args.module, args.args, resolve("27b"))
    return 2
