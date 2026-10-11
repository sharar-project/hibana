"""The `sharar` command (sharar/cli.py, sharar/launch/): what each subcommand starts, without a GPU, systemd or a server."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sharar import cli
from sharar.launch import ctl, preflight, run, service

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def home(monkeypatch, tmp_path):
    """No config file, results under tmp_path, nothing serving, no systemd."""
    for key in [k for k in os.environ if k.startswith("SHARAR_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("SHARAR_CONFIG_FILE", str(tmp_path / "none.env"))
    monkeypatch.setenv("SHARAR_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("SHARAR_HF_CACHE", str(tmp_path / "hf"))
    monkeypatch.setattr(ctl, "server_info", lambda port: None)
    monkeypatch.setattr(service, "running", lambda unit: False)
    monkeypatch.setattr(preflight, "check_host", lambda: {"gpu": "GB10", "memory": "128 GiB"})
    monkeypatch.setattr(preflight, "ensure_snapshots", lambda s, log=print, fetch=True: [])
    monkeypatch.setattr(preflight, "ensure_w3", lambda s, log=print, fetch=True: tmp_path / "w3" if s.get("SHARAR_W3_DECODE") == "1" else None)
    from sharar.launch import lxcfs
    monkeypatch.setattr(lxcfs, "reexec_if_needed", lambda: None)
    saved = dict(os.environ)  # serve and train install the engine's settings into this process's environment
    yield tmp_path
    os.environ.clear()
    os.environ.update(saved)


def test_importing_the_cli_does_not_import_torch():
    out = subprocess.run([sys.executable, "-c", "import sys, sharar.cli, sharar.launch.run; print('torch' in sys.modules)"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.stdout.strip() == "False"


def test_serve_runs_the_engine_in_the_foreground(home, monkeypatch):
    started = []
    monkeypatch.setattr(run.guard, "start_thread", lambda **kw: started.append("guard"))
    monkeypatch.setattr(cli, "run_engine", lambda args, parser: started.append((args, dict(os.environ))) or 0)
    monkeypatch.setenv("SHARAR_DRAFT", "pinned")  # set by the caller; the engine gets the resolved draft, not this
    assert cli.main(["serve", "flashnext", "--exact", "--no-traces", "--port", "30500"]) == 0
    assert started[0] == "guard"
    args, env = started[1]
    assert (args.target, args.port, args.proposer, args.spec, args.accept_default) == ("flashnext", 30500, "mtp", 4, "off")
    assert args.traces is None and args.context_length == 262144
    assert env["SHARAR_PROFILE"] == "exact" and env["SHARAR_W3_DECODE"] == "0" and env["HF_HUB_OFFLINE"] == "1"
    assert "SHARAR_DRAFT" not in env and "/usr/local/cuda/bin" in env["PATH"]
    assert (home / "results" / "sessions").is_dir()


def test_serve_refuses_a_busy_port_and_bad_settings(home, monkeypatch):
    with pytest.raises(SystemExit, match="MAX_BATCH"):
        cli.main(["serve", "27b", "--max-batch", "500"])
    monkeypatch.setattr(ctl, "server_info", lambda port: {"target": "qwen38"})
    with pytest.raises(SystemExit, match="already serves"):
        cli.main(["serve", "27b"])


def test_detach_starts_the_same_command_as_a_service(home, monkeypatch):
    started = []
    monkeypatch.setattr(service, "start", lambda unit, argv, log, description, stop_seconds=120: started.append((unit, argv, log)))
    monkeypatch.setattr(service, "follow_startup", lambda unit, log, url: 0)
    monkeypatch.setattr(sys, "argv", ["sharar", "serve", "27b", "--detach", "--throughput"])
    assert cli.main(["serve", "27b", "--detach", "--throughput"]) == 0
    unit, argv, log = started[0]
    assert unit == "sharar-qwen38" and argv == ["serve", "27b", "--throughput", "--no-fetch"]
    assert log == home / "results" / "logs" / "sharar-qwen38.log"


def test_service_start_composes_systemd_run(home, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(service, "systemd_available", lambda: True)
    monkeypatch.setattr(service.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    log = tmp_path / "logs" / "unit.log"
    log.parent.mkdir()
    log.write_text("old run")
    service.start("sharar-x", ["serve", "27b"], log, "Sharar test")
    cmd = calls[-1]
    assert cmd[:2] == ["systemd-run", "--user"] and cmd[cmd.index("--unit") + 1] == "sharar-x"
    assert cmd[-5:] == [sys.executable, "-m", "sharar", "serve", "27b"]
    assert f"StandardOutput=append:{log}" in cmd and f"SHARAR_RESULTS={home / 'results'}" in cmd
    assert (tmp_path / "logs" / "unit.log.1").read_text() == "old run"


def test_stop_status_and_logs_follow_what_is_serving(home, monkeypatch, capsys):
    stopped = []
    monkeypatch.setattr(service, "stop", lambda unit: stopped.append(unit) or unit == "sharar-flashnext")
    assert cli.main(["stop"]) == 0 and stopped == ["sharar-qwen38", "sharar-flashnext"]
    assert "stopped sharar-flashnext" in capsys.readouterr().out
    monkeypatch.setattr(ctl, "server_info", lambda port: {"config": {"profile": "fast"}, "tp_size": 1, "context_length": 262144,
                                                           "max_running_requests": 4} if port == "30004" else None)
    assert run.serving(None).target == "flashnext" and run.serving("27b").target == "qwen38"
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "flashnext: serving on http://127.0.0.1:30004 from the foreground, profile fast" in out and "qwen38:" not in out
    with pytest.raises(SystemExit, match="no log"):
        cli.main(["logs"])
    log = home / "results" / "logs" / "sharar-flashnext.log"
    log.parent.mkdir(parents=True)
    log.write_text("x\n")
    tails = []
    monkeypatch.setattr(run.subprocess, "call", lambda cmd: tails.append(cmd) or 0)
    assert cli.main(["logs", "-f"]) == 0 and tails[-1] == ["tail", "-n", "50", "-F", str(log)]


def test_train_fills_the_run_from_the_settings(home, monkeypatch):
    traces = home / "results" / "traces"
    traces.mkdir(parents=True)
    ran = []
    monkeypatch.setattr(run.guard, "start_thread", lambda **kw: None)
    monkeypatch.setattr(cli, "run_train", lambda args, parser: ran.append(args) or 0)
    assert cli.main(["train", "--steps", "10"]) == 0
    args = ran[-1]
    runs = home / "results" / "draft-runs"
    assert args.traces == str(traces) and args.output.startswith(str(runs / "run-")) and args.steps == 10
    assert args.promote_to == args.continue_from == str(runs / "current") and args.lr == 1e-4 and args.continue_lr == 3e-5
    assert cli.main(["train", "--promote-to", "none", "--continue-from", "none"]) == 0
    assert ran[-1].promote_to is None and ran[-1].continue_from is None
    monkeypatch.setattr(ctl, "server_info", lambda port: {"target": "qwen38"} if port == "30001" else None)
    with pytest.raises(SystemExit, match="foreground"):
        cli.main(["train"])


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self.body).encode()


def test_launch_sets_up_each_agent_for_the_serving_target(home, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(ctl, "server_info", lambda port: {"target": "x"})
    monkeypatch.setattr(ctl.urllib.request, "urlopen", lambda url, timeout: FakeResponse({"context_length": 131072}))
    monkeypatch.setattr(ctl.shutil, "which", lambda name: f"/usr/bin/{name}")
    launched = []
    monkeypatch.setattr(ctl.os, "execvpe", lambda file, command, env: launched.append((command, env)))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "real-key")
    cli.main(["launch", "claude", "-p", "fix it"])
    command, env = launched[-1]
    assert command == ["claude", "-p", "fix it"] and env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:30001"
    assert "ANTHROPIC_API_KEY" not in env and env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072" and "claude" in env["ANTHROPIC_MODEL"]
    cli.main(["launch", "--target", "flashnext", "opencode"])
    command, env = launched[-1]
    config = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
    assert command == ["opencode"] and config["model"] == "sharar/qwen3.8-flash-next"
    provider = config["provider"]["sharar"]
    assert provider["options"]["baseURL"] == "http://127.0.0.1:30004/v1" and provider["models"]["qwen3.8-flash-next"]["limit"]["context"] == 131072
    cli.main(["launch", "aider", "--yes"])
    command, env = launched[-1]
    assert command[:3] == ["aider", "--model", "openai/qwen3.8-27b"] and command[-1] == "--yes" and env["OPENAI_API_BASE"] == "http://127.0.0.1:30001/v1"
    with pytest.raises(SystemExit, match="Responses API"):
        cli.main(["launch", "codex"])


def test_launch_and_chat_need_a_server(home):
    with pytest.raises(SystemExit, match="nothing is serving"):
        cli.main(["chat"])


def test_draft_list_use_pinned_and_reload(home, monkeypatch, capsys):
    runs = home / "results" / "draft-runs"
    for name, best in (("run-1", 5.1), ("run-2", 5.4)):
        d = runs / name / "best"
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
        (runs / name / "manifest.json").write_text(json.dumps({"base_eval": {"tokens_per_round": 5.0}, "best": {"step": 40, "tokens_per_round": best}}))
    assert cli.main(["draft", "use", "run-2"]) == 0
    assert os.readlink(runs / "current") == "run-2/best" and "serving is down" in capsys.readouterr().out
    assert cli.main(["draft", "list"]) == 0
    out = capsys.readouterr().out
    assert "run-2: base 5.000 -> best 5.400 at step 40 *" in out and "run-1: base 5.000 -> best 5.100" in out
    with pytest.raises(SystemExit):
        cli.main(["draft", "use", "run-9"])
    posted = []
    monkeypatch.setattr(ctl, "server_info", lambda port: {"speculative": {}})
    monkeypatch.setattr(ctl, "post_json", lambda port, route, payload, timeout=120: posted.append((port, route, payload)) or {"draft": payload["draft"], "seconds": 1.5})
    assert cli.main(["draft", "use", "run-1"]) == 0
    assert posted[-1] == ("30001", "/reload_draft", {"draft": str(runs / "run-1" / "best")})
    assert cli.main(["draft", "pinned"]) == 0 and posted[-1][2] == {"draft": "pinned"} and not (runs / "current").exists()


def test_no_launcher_scripts_left():
    for name in ("sharar-ctl", "sharar-tp", "run-sharar.sh", "run-sharar-train.sh"):
        assert not (ROOT / name).exists()
    assert sorted(p.name for p in (ROOT / "tools").glob("*.sh")) == ["check.sh"]
