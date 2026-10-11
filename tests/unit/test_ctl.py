"""The host-side control command composes the launcher calls without torch, systemd or a server."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def ctl(monkeypatch, tmp_path):
    from importlib.machinery import SourceFileLoader
    spec = importlib.util.spec_from_file_location("sharar_ctl", ROOT / "sharar-ctl", loader=SourceFileLoader("sharar_ctl", str(ROOT / "sharar-ctl")))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "gb10.env"
    config.write_text(f"SHARAR_HF_CACHE=/cache\nSHARAR_RESULTS={tmp_path / 'results'}\n")
    monkeypatch.setenv("SHARAR_CONFIG_FILE", str(config))
    monkeypatch.setenv("SHARAR_RESULTS", str(tmp_path / "results"))  # the environment outranks the config file
    for name in ("SHARAR_TRACES", "SHARAR_DRAFT", "SHARAR_DRAFT_RUNS", "SHARAR_DRAFT_PRECISION"):
        monkeypatch.delenv(name, raising=False)
    calls = []
    monkeypatch.setattr(module, "run", lambda command, env, dry: calls.append((command, env)) or 0)
    monkeypatch.setattr(module, "service_running", lambda name: False)
    monkeypatch.setattr(module, "follow_startup", lambda env: 0)
    monkeypatch.setattr(module, "server_info", lambda port: None)  # never touch a real server
    return module, calls, tmp_path / "results"


def test_serve_records_traces_by_default_and_honors_overrides(ctl):
    module, calls, results = ctl
    assert module.main(["serve"]) == 0
    command, env = calls[-1]
    assert command == [str(ROOT / "run-sharar.sh")] and env["SHARAR_TRACES"] == str(results / "traces") and "SHARAR_DRAFT" not in env
    assert module.main(["serve", "--no-traces", "--pinned", "--fp8"]) == 0
    _, env = calls[-1]
    assert "SHARAR_TRACES" not in env and env["SHARAR_DRAFT"] == "pinned" and env["SHARAR_DRAFT_PRECISION"] == "fp8"
    assert module.main(["serve", "--draft", "/ckpt"]) == 0 and calls[-1][1]["SHARAR_DRAFT"] == "/ckpt"
    assert module.main(["serve", "--throughput"]) == 0 and calls[-1][1]["SHARAR_PROFILE"] == "throughput"
    assert module.main(["stop"]) == 0 and calls[-1][0] == [str(ROOT / "run-sharar.sh"), "--stop"]
    assert module.main(["serve", "--tp", "2"]) == 0
    command, env = calls[-1]
    assert command == [str(ROOT / "sharar-tp"), "serve"] and env["SHARAR_TRACES"] == str(results / "traces")
    assert module.main(["serve", "--tp", "2", "--no-traces"]) == 0 and "SHARAR_TRACES" not in calls[-1][1]


def test_train_commands_wrap_the_training_launcher(ctl):
    module, calls, _ = ctl
    script = str(ROOT / "run-sharar-train.sh")
    assert module.main(["train"]) == 0 and calls[-1][0] == [script, "start"]
    assert module.main(["train", "stop"]) == 0 and calls[-1][0] == [script, "stop"]
    assert module.main(["train", "stop", "--no-serve"]) == 0 and calls[-1][0] == [script, "stop", "--no-serve"]
    assert module.main(["train", "status"]) == 0 and calls[-1][0] == [script, "status"]


def test_draft_list_use_and_pinned(ctl, capsys):
    module, calls, results = ctl
    runs = results / "draft-runs"
    for name, best in (("run-1", 5.1), ("run-2", 5.4)):
        d = runs / name / "best"
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
        (runs / name / "manifest.json").write_text(json.dumps({"base_eval": {"tokens_per_round": 5.0}, "best": {"step": 40, "tokens_per_round": best}}))
    assert module.main(["draft", "use", "run-2"]) == 0
    assert os.readlink(runs / "current") == "run-2/best"
    assert module.main(["draft", "list"]) == 0
    out = capsys.readouterr().out
    assert "run-2: base 5.000 -> best 5.400 at step 40 *" in out and "run-1: base 5.000 -> best 5.100" in out
    with pytest.raises(SystemExit):
        module.main(["draft", "use", "run-9"])
    assert module.main(["draft", "pinned"]) == 0 and not (runs / "current").exists()
    assert module.main(["status"]) == 0
    assert "current draft: pinned publisher draft" in capsys.readouterr().out


def test_setup_builds_the_python_environment_and_traces_and_logs_use_it(ctl, monkeypatch):
    module, calls, results = ctl
    assert module.main(["setup"]) == 0 and calls[-1][0] == ["bash", str(ROOT / "tools" / "setup-venv.sh")]
    assert module.main(["traces"]) == 0
    command, env = calls[-1]
    assert command == [str(ROOT / ".venv" / "bin" / "python"), "-m", "sharar", "traces", "--dir", str(results / "traces")]
    assert env["PYTHONPATH"] == str(ROOT)
    monkeypatch.setenv("SHARAR_VENV", "/opt/venv")
    assert module.main(["traces"]) == 0 and calls[-1][0][0] == "/opt/venv/bin/python"
    assert module.main(["logs", "-f"]) == 0 and calls[-1][0] == ["tail", "-n", "50", "-F", str(results / "logs" / "sharar-qwen38.log")]
    assert module.main(["logs", "--train"]) == 0 and calls[-1][0][-1] == str(results / "logs" / "sharar-train.log")
    for launcher in ("run-sharar.sh", "run-sharar-train.sh"):  # no container runtime left in the launchers
        assert "docker" not in (ROOT / launcher).read_text()


def test_follow_startup_prints_milestones_until_serving(ctl, monkeypatch, capsys, tmp_path):
    from importlib.machinery import SourceFileLoader
    spec = importlib.util.spec_from_file_location("sharar_ctl2", ROOT / "sharar-ctl", loader=SourceFileLoader("sharar_ctl2", str(ROOT / "sharar-ctl")))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    log = tmp_path / "logs" / "svc.log"
    log.parent.mkdir()
    log.write_text("loading qwen38\nnoise\n  weights loaded in 30 s\nserving on http://127.0.0.1:30001\n")
    monkeypatch.setattr(module, "service_running", lambda name: True)
    env = {"SHARAR_SERVICE": "svc", "SHARAR_RESULTS": str(tmp_path), "SHARAR_PORT": "30001"}
    assert module.follow_startup(env) == 0
    out = capsys.readouterr().out
    assert "loading qwen38" in out and "weights loaded" in out and "noise" not in out and "ready:" in out
    log.write_text("loading qwen38\nTraceback (most recent call last):\nRuntimeError: boom\n")
    assert module.follow_startup(env) == 1 and "boom" in capsys.readouterr().out


def test_serve_names_a_target_and_gets_its_settings(ctl):
    module, calls, _ = ctl
    assert module.main(["serve", "flashnext"]) == 0
    _, env = calls[-1]
    assert (env["SHARAR_TARGET"], env["SHARAR_PORT"], env["SHARAR_SERVICE"], env["SHARAR_PROPOSER"]) == ("flashnext", "30004", "sharar-flashnext", "mtp")
    assert module.main(["serve", "27b"]) == 0
    _, env = calls[-1]
    assert (env["SHARAR_TARGET"], env["SHARAR_PORT"], env["SHARAR_PROPOSER"]) == ("qwen38", "30001", "dflash2")
    assert module.main(["serve"]) == 0 and calls[-1][1]["SHARAR_TARGET"] == "qwen38"  # no target named: the 27B
    with pytest.raises(SystemExit):
        module.main(["serve", "llama"])


def test_commands_without_a_target_follow_the_one_serving(ctl, monkeypatch):
    module, calls, _ = ctl
    monkeypatch.setattr(module, "service_running", lambda name: name == "sharar-flashnext")
    assert module.main(["stop"]) == 0 and calls[-1][1]["SHARAR_TARGET"] == "flashnext"
    assert module.main(["stop", "27b"]) == 0 and calls[-1][1]["SHARAR_TARGET"] == "qwen38"


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self.body).encode()


def test_launch_sets_up_each_agent_for_the_serving_target(ctl, monkeypatch, tmp_path):
    module, _, _ = ctl
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda url, timeout: FakeResponse({"context_length": 131072}))
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/usr/bin/{name}")
    launched = []
    monkeypatch.setattr(module.os, "execvpe", lambda file, command, env: launched.append((command, env)))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "real-key")
    module.main(["launch", "claude", "-p", "fix it"])
    command, env = launched[-1]
    assert command == ["claude", "-p", "fix it"] and env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:30001"
    assert "ANTHROPIC_API_KEY" not in env and env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072" and "claude" in env["ANTHROPIC_MODEL"]
    module.main(["launch", "--target", "flashnext", "opencode"])
    command, env = launched[-1]
    config = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
    assert command == ["opencode"] and config["model"] == "sharar/qwen3.8-flash-next"
    provider = config["provider"]["sharar"]
    assert provider["options"]["baseURL"] == "http://127.0.0.1:30004/v1" and provider["models"]["qwen3.8-flash-next"]["limit"]["context"] == 131072
    module.main(["launch", "aider", "--yes"])
    command, env = launched[-1]
    assert command[:3] == ["aider", "--model", "openai/qwen3.8-27b"] and command[-1] == "--yes" and env["OPENAI_API_BASE"] == "http://127.0.0.1:30001/v1"
    assert json.loads(Path(command[4]).read_text())["openai/qwen3.8-27b"]["max_input_tokens"] == 131072
    with pytest.raises(SystemExit, match="Responses API"):
        module.main(["launch", "codex"])


def test_dry_run_prints_instead_of_running(ctl, capsys, monkeypatch):
    module, calls, _ = ctl
    monkeypatch.setattr(module, "run", lambda command, env, dry: (print("+ " + " ".join(command)) or 0) if dry else 1)
    assert module.main(["--dry-run", "traces"]) == 0
    assert "-m sharar traces --dir " in capsys.readouterr().out


def test_draft_use_pinned_and_reload_swap_the_running_server(ctl, capsys, monkeypatch):
    module, calls, results = ctl
    runs = results / "draft-runs"
    best = runs / "run-3" / "best"
    best.mkdir(parents=True)
    (best / "config.json").write_text("{}")
    posted = []
    monkeypatch.setattr(module, "server_info", lambda port: None)
    assert module.main(["draft", "use", "run-3"]) == 0
    assert "serving is down" in capsys.readouterr().out and not posted
    monkeypatch.setattr(module, "server_info", lambda port: {"speculative": {}})
    monkeypatch.setattr(module, "post_json", lambda port, route, payload, timeout=120: posted.append((port, route, payload)) or {"draft": payload["draft"], "seconds": 1.5})
    assert module.main(["draft", "use", "run-3"]) == 0
    assert posted[-1] == ("30001", "/reload_draft", {"draft": str(best)})
    assert f"serving now uses {best}" in capsys.readouterr().out
    assert module.main(["draft", "reload"]) == 0 and posted[-1][2] == {"draft": str(best)}
    assert module.main(["draft", "pinned"]) == 0 and posted[-1][2] == {"draft": "pinned"}
    assert module.main(["draft", "reload"]) == 0 and posted[-1][2] == {"draft": "pinned"}  # no current link: the pinned draft
    assert module.main(["--dry-run", "draft", "reload"]) == 0 and "POST" in capsys.readouterr().out and len(posted) == 4
