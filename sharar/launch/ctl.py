"""The client commands: chat, launch an agent, manage trained drafts. Standard library only (no torch)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

from sharar.launch.settings import Settings


def server_info(port: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/get_server_info", timeout=3) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001 - not serving
        return None


def post_json(port: str, route: str, payload: dict, timeout: float = 120) -> dict:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        try:
            detail = json.loads(detail)["error"]["message"]
        except Exception:  # noqa: BLE001 - not the engine's error shape
            pass
        sys.exit(f"{route} failed: {detail}")


def reload_served_draft(s: Settings, run: str | None, dry: bool = False) -> bool:
    """Swap the draft in the running server: a run name, or None for the pinned draft.
    Returns False when nothing is serving."""
    port = str(s.port)
    target = str(s.draft_runs / run / "best") if run else "pinned"
    if dry:
        print(f"+ POST http://127.0.0.1:{port}/reload_draft {{\"draft\": \"{target}\"}}")
        return True
    if server_info(port) is None:
        return False
    reply = post_json(port, "/reload_draft", {"draft": target})
    print(f"serving now uses {reply.get('draft')} ({reply.get('seconds')}s)")
    return True


def current_draft(s: Settings) -> Path | None:
    link = s.draft_runs / "current"
    if not link.is_symlink():
        return None
    target = link.resolve()
    return target if (target / "config.json").is_file() else None


HARNESSES = ("claude", "opencode", "aider", "codex", "show")
ALIASES = {"qwen38": ("qwen3.8-27b", "Qwen3.8-27B"), "flashnext": ("qwen3.8-flash-next", "Qwen3.8-Flash-Next")}
OUTPUT_TOKENS = 32768  # reply budget the harness configs declare (thinking included); the context is the server's


def write_config(path: Path, data: dict, dry: bool) -> None:
    if dry:
        print(f"+ write {path}: {json.dumps(data)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def cmd_launch(args, s: Settings, dry: bool = False) -> int:
    """Start a coding agent against the serving target with its own config (the user's own settings are not touched)."""
    if args.harness == "codex":
        sys.exit("Codex needs the OpenAI Responses API (/v1/responses), which Sharar does not serve")
    url = (os.environ.get("SHARAR_URL") or s.url).rstrip("/")
    try:
        with urllib.request.urlopen(f"{url}/get_server_info", timeout=3) as r:
            info = json.loads(r.read())
    except Exception:  # noqa: BLE001 - not serving
        if not dry:
            sys.exit(f"nothing is serving at {url}: sharar serve 27b (or flashnext) first")
        info = {}
    context = int(info.get("context_length", 262144))
    alias, display = ALIASES[s.target]
    conf = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "sharar"
    henv = dict(os.environ)
    if args.harness == "show":
        print(f"Sharar is serving {display} at {url} (context {context} tokens)")
        print(f"  OpenAI-compatible tools:  base URL {url}/v1, model {alias}, any API key")
        print(f"  Anthropic Messages tools: base URL {url}, any model name and API key (Claude Code: sharar launch claude)")
        print("  Cline, Continue, Zed, Pi and others: an OpenAI-compatible provider with those values, context window "
              f"{context}, max output {OUTPUT_TOKENS}")
        return 0
    if args.harness == "claude":
        # Claude Code only accepts model ids containing "claude"; its own config directory keeps the user's
        # Anthropic login, settings and history untouched (the first launch asks the onboarding questions).
        model = os.environ.get("SHARAR_CLAUDE_MODEL", "claude-sharar-qwen3.8")
        for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE"):
            henv.pop(k, None)
        henv.update(ANTHROPIC_BASE_URL=url, ANTHROPIC_AUTH_TOKEN="sharar", ANTHROPIC_MODEL=model,
                    ANTHROPIC_DEFAULT_SONNET_MODEL=model, ANTHROPIC_DEFAULT_OPUS_MODEL=model, ANTHROPIC_DEFAULT_HAIKU_MODEL=model,
                    CLAUDE_CODE_MAX_CONTEXT_TOKENS=str(context), CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
                    # the model name is not in Claude Code's catalog: without this it assumes a 200K window
                    CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT="1",
                    CLAUDE_CONFIG_DIR=os.environ.get("SHARAR_CLAUDE_CONFIG", str(Path.home() / ".claude-sharar")))
        command = ["claude", *args.args]
    elif args.harness == "opencode":
        # OPENCODE_CONFIG is merged over the user's global config: a `sharar` provider and the default model.
        path = conf / "opencode.json"
        write_config(path, {"$schema": "https://opencode.ai/config.json", "model": f"sharar/{alias}",
                            "provider": {"sharar": {"npm": "@ai-sdk/openai-compatible", "name": "Sharar",
                                                    "options": {"baseURL": f"{url}/v1", "apiKey": "sharar"},
                                                    "models": {alias: {"name": f"{display} (Sharar)",
                                                                       "limit": {"context": context, "output": OUTPUT_TOKENS}}}}}}, dry)
        henv["OPENCODE_CONFIG"] = str(path)
        command = ["opencode", *args.args]
    else:  # aider: an OpenAI-compatible endpoint plus the model's context in a metadata file
        path = conf / "aider.model.metadata.json"
        write_config(path, {f"openai/{alias}": {"max_tokens": OUTPUT_TOKENS, "max_input_tokens": context,
                                                 "max_output_tokens": OUTPUT_TOKENS, "input_cost_per_token": 0,
                                                 "output_cost_per_token": 0, "litellm_provider": "openai", "mode": "chat"}}, dry)
        henv.update(OPENAI_API_BASE=f"{url}/v1", OPENAI_API_KEY="sharar")
        command = ["aider", "--model", f"openai/{alias}", "--model-metadata-file", str(path), *args.args]
    if dry:
        print("+ " + " ".join(command))
        return 0
    if shutil.which(command[0]) is None:
        sys.exit(f"{command[0]} is not installed or not on PATH")
    os.execvpe(command[0], command, henv)  # in the caller's directory, replacing this process


def cmd_draft(args, s: Settings, dry: bool = False) -> int:
    runs = s.draft_runs
    link = runs / "current"
    if args.action == "list":
        current = current_draft(s)
        for run_dir in sorted(runs.glob("run-*")):
            manifest = run_dir / "manifest.json"
            best = run_dir / "best"
            if not manifest.is_file():
                continue
            m = json.loads(manifest.read_text())
            mark = " *" if current and best.resolve() == current else ""
            base = m.get("base_eval", {}).get("tokens_per_round")
            b = m.get("best", {})
            print(f"{run_dir.name}: base {base:.3f} -> best {b.get('tokens_per_round', 0):.3f} at step {b.get('step')}{mark}")
        print("current: " + (str(current) if current else "pinned publisher draft"))
        return 0
    if args.action == "use":
        target = runs / args.run / "best"
        if not (target / "config.json").is_file():
            sys.exit(f"{target} is not a promoted checkpoint")
        if dry:
            print(f"+ ln -sfn {os.path.relpath(target, runs)} {link}")
            return 0
        tmp = runs / ".current.tmp"
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(os.path.relpath(target, runs))
        tmp.replace(link)
        print(f"current -> {target}")
        if not reload_served_draft(s, args.run, dry):
            print("serving is down; it follows current at the next sharar serve")
        return 0
    if args.action == "pinned":
        if link.is_symlink() and not dry:
            link.unlink()
        print("current draft removed")
        if not reload_served_draft(s, None, dry):
            print("serving is down; it uses the pinned publisher draft at the next sharar serve")
        return 0
    if args.action == "reload":
        current = current_draft(s)
        run = current.parent.name if current and current.parent.parent == runs.resolve() else None
        if current and run is None:
            sys.exit(f"current draft {current} is outside {runs}; serve it with sharar serve --draft")
        if not reload_served_draft(s, run, dry):
            sys.exit("serving is not up (sharar serve)")
        return 0
    return 2


def cmd_chat(args, s: Settings) -> int:
    """Streaming multi-turn chat against the running server; /reset, /think, /quit."""
    import time
    url = f"{s.url}/v1/chat/completions"
    if server_info(str(s.port)) is None:
        sys.exit("serving is not up (sharar serve)")
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    think = args.think
    max_tokens = args.max_tokens
    print(f"chatting with Sharar on {url} (thinking {'on' if think else 'off'}, budget {max_tokens} tokens; /reset, /think, /max N, /quit)", flush=True)
    while True:
        try:
            line = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        text = line.strip()
        if not text:
            continue
        if text in ("/quit", "/exit"):
            return 0
        if text == "/reset":
            messages = messages[:1] if messages and messages[0]["role"] == "system" else []
            print("(history cleared)")
            continue
        if text.startswith("/think"):
            think = not think if text == "/think" else text.split()[-1] == "on"
            print(f"(thinking {'on' if think else 'off'})")
            continue
        if text.startswith("/max"):
            try:
                max_tokens = int(text.split()[1])
                print(f"(reply budget {max_tokens} tokens)")
            except (IndexError, ValueError):
                print("usage: /max N")
            continue
        messages.append({"role": "user", "content": text})
        body = {"model": "sharar", "messages": messages, "stream": True, "max_tokens": max_tokens,
                "temperature": args.temperature, "chat_template_kwargs": {"enable_thinking": think},
                "stream_options": {"include_usage": True}}
        if args.no_trace:
            body["trace"] = False
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        first = None
        reply, reasoning, usage, finish = [], [], None, None
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                print("sharar> ", end="", flush=True)
                for raw in response:
                    raw = raw.decode().strip()
                    if not raw.startswith("data:"):
                        continue
                    data = raw[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        finish = choice.get("finish_reason") or finish
                        delta = choice.get("delta") or {}
                        if delta.get("reasoning_content"):
                            if first is None:
                                first = time.perf_counter()
                            reasoning.append(delta["reasoning_content"])
                            if args.show_thinking:
                                print(delta["reasoning_content"], end="", flush=True)
                        if delta.get("content"):
                            if first is None:
                                first = time.perf_counter()
                            reply.append(delta["content"])
                            print(delta["content"], end="", flush=True)
        except KeyboardInterrupt:
            print("\n(interrupted)")
            messages.pop()
            continue
        except urllib.error.URLError as exc:
            print(f"\nrequest failed: {exc}")
            messages.pop()
            continue
        elapsed = time.perf_counter() - started
        answer = "".join(reply)
        messages.append({"role": "assistant", "content": answer})
        completion = (usage or {}).get("completion_tokens")
        ttft = f"{first - started:.2f}s to first token, " if first else ""
        rate = f"{completion / max(elapsed - (first - started if first else 0), 1e-6):.1f} tok/s" if completion else ""
        print(f"\n[{completion or '?'} tokens, {ttft}{rate}{', thinking hidden' if reasoning and not args.show_thinking else ''}]")
        if finish == "length":
            what = "while still thinking; no answer was produced" if reasoning and not answer else "before the reply finished"
            print(f"(stopped at the {max_tokens}-token budget {what}: /max N raises it, /think off skips thinking)")


def verify(url: str, s: Settings) -> int:
    """A running server: health, the pinned model and context, the model list, and a short exact answer."""
    if s.target == "flashnext":
        from sharar.model.flashnext import config as c
    else:
        from sharar.model import qwen38_config as c

    def get(route):
        with urllib.request.urlopen(f"{url}{route}", timeout=30) as r:
            return json.loads(r.read()) if route != "/health" else r.status

    try:
        get("/health")
        info = get("/get_server_info")
        expected = {"engine": "sharar", "model_path": c.MODEL_ID, "revision": c.MODEL_REVISION,
                    "kv_cache_dtype": "float8_e4m3fn", "mamba_ssm_dtype": "float32"}
        wrong = {k: (info.get(k), v) for k, v in expected.items() if info.get(k) != v}
        if wrong:
            sys.exit(f"sharar verify: {url} serves something else: {wrong}")
        models = get("/v1/models")["data"]
        if [m["id"] for m in models] != [c.MODEL_ID]:
            sys.exit(f"sharar verify: model list is {models}")
        body = {"model": c.MODEL_ID, "messages": [{"role": "user", "content": "Reply with exactly SHARAR_OK and nothing else."}],
                "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, "trace": False}
        request = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as r:
            answer = json.loads(r.read())["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        sys.exit(f"sharar verify: {url}: {exc}")
    if answer != "SHARAR_OK":
        sys.exit(f"sharar verify: the canary answered {answer!r}")
    print(f"Sharar READY at {url} ({c.MODEL_ID}, context {info.get('context_length')}, {info.get('tp_size', 1)} Spark(s))")
    return 0
