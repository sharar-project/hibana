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
    monkeypatch.setattr(preflight, "check_host", lambda floor_kib=None: {"gpu": "GB10", "memory": "128 GiB"})
    monkeypatch.setattr(preflight, "ensure_snapshots", lambda s, log=print, fetch=True: [])
    monkeypatch.setattr(preflight, "ensure_w3", lambda s, log=print, fetch=True: tmp_path / "w3" if s.get("SHARAR_W3_DECODE") == "1" else None)
    from sharar.launch import lxcfs
    monkeypatch.setattr(lxcfs, "reexec_if_needed", lambda: None)
    saved = dict(os.environ)  # serve installs the engine's settings into this process's environment
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
    assert cli.main(["serve", "flashnext", "--exact", "--port", "30500"]) == 0
    assert started[0] == "guard"
    args, env = started[1]
    assert (args.target, args.port, args.proposer, args.spec, args.accept_default) == ("flashnext", 30500, "mtp", 4, "off")
    assert args.traces is None and args.context_length == 262144  # traces are opt-in
    assert env["SHARAR_PROFILE"] == "exact" and env["SHARAR_W3_DECODE"] == "0" and env["HF_HUB_OFFLINE"] == "1"
    assert "SHARAR_DRAFT" not in env and "/usr/local/cuda/bin" in env["PATH"]
    assert not (home / "results" / "traces").exists()
    assert cli.main(["serve", "flashnext", "--traces", "--port", "30500"]) == 0
    args, _ = started[-1]
    assert args.traces == str(home / "results" / "traces") and (home / "results" / "traces").is_dir()
    assert env["SHARAR_SSD"] == "1" and env["SHARAR_SSD_DIR"] == str(home / "results" / "ssd" / "flashnext")  # on by default


def test_engine_by_hand_serves_what_serve_would(home):
    from sharar.engine.config import EngineConfig

    args = cli.engine_parser().parse_args(["--target", "flashnext"])
    cfg = EngineConfig.from_env(cli.resolved_engine_env(args))
    assert (args.port, args.proposer, args.verify_state, args.draft_precision) == (30004, "mtp", "reconstruct", "bf16")
    assert cfg.profile == "fast" and cfg.accept_default == "3:2.0" and cfg.accept_soft == 0.5 and cfg.device_accept and (cfg.moe_keep, cfg.moe_budget, cfg.moe_budget_per_request) == (4, 64, 12)
    assert cfg.qsa_prefill_backend == "auto" and cfg.max_context == 262144 and cfg.spec == "8"
    assert cfg.ssd and cfg.ssd_dir == str(home / "results" / "ssd" / "flashnext") and cfg.w3_decode
    args = cli.engine_parser().parse_args(["--port", "31000"])
    cfg = EngineConfig.from_env(cli.resolved_engine_env(args))
    assert (args.port, args.proposer, args.draft_precision) == (31000, "dflash2", "fp8") and cfg.target == "qwen38"


@pytest.mark.parametrize("profile", ["--exact", "--fast", "--throughput"])
@pytest.mark.parametrize("target", ["27b", "flashnext"])
def test_engine_resolution_keeps_what_serve_prepared(home, monkeypatch, target, profile):
    from sharar.engine.config import EngineConfig

    started = []
    monkeypatch.setattr(run.guard, "start_thread", lambda **kw: None)
    monkeypatch.setattr(cli, "run_engine", lambda args, parser: started.append((args, dict(os.environ))) or 0)
    assert cli.main(["serve", target, profile]) == 0
    args, env = started[0]
    before = vars(args).copy()
    os.environ.clear()  # the engine process's environment (the `home` fixture restores the test's)
    os.environ.update(env)
    assert EngineConfig.from_env(cli.resolved_engine_env(args)) == EngineConfig.from_env(env)
    assert vars(args) == before


def test_cache_reports_the_gpu_cache_and_the_ssd_tier(home, monkeypatch, capsys):
    ssd = {"prefixes": 4, "used_gib": 1.5, "budget_gib": 200.0, "restores": 2, "restored_tokens": 9000}
    info = {"kv_pages": 1000, "config": {"ssd": True, "ssd_dir": "/tmp/sharar-ssd"},
            "scheduler": {"cache": {"cached_tokens": 600, "hits": 3, "misses": 1, "evictions": 2, "ssd": ssd}}}
    monkeypatch.setattr(ctl, "server_info", lambda port: info if port == "30001" else None)
    assert cli.main(["cache"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("qwen38: serving") and "600 of 1,000 KV tokens cached" in out
    assert "4 prefixes, 1.5 of 200 GiB in /tmp/sharar-ssd; 2 restores (9,000 tokens)" in out
    info["scheduler"]["cache"].update(locks={"total": 3, "request": 2, "restore": 1}, snapshots={"spare": 4, "not_spare": 7},
                                      evicted={"slots": {"spare": 5, "not_spare": 1}, "kv": {"not_spare": 2}},
                                      pruned=5, full_reprefills=1, branching_resumes=2,
                                      draft_resumes={"hits": 9, "fallbacks": 0})
    assert cli.main(["cache"]) == 0
    assert ("locks: 3 (2 requests, 1 SSD restores); snapshots: 7 + 4 spare, 8 evicted; 5 dead-KV nodes pruned, "
            "1 full re-prefills, 2 resumes at a branching point; drafter resumes 9 hits, 0 fallbacks") in capsys.readouterr().out
    del info["scheduler"]["cache"]["ssd"]
    assert cli.main(["cache"]) == 0 and "on in the settings but not running" in capsys.readouterr().out


def test_cache_without_a_server_counts_the_store_on_disk(home, monkeypatch, capsys, tmp_path):
    epoch = tmp_path / "ssd" / "v2" / "e3"
    (epoch / "ab").mkdir(parents=True)
    (epoch.parent / "EPOCH").write_text("3")
    (epoch.parent / "e2" / "cd").mkdir(parents=True)
    for name in ("e3/ab/1.kv", "e3/ab/2.kv", "e3/ab/3.tail", "e3/ab/4.kv.tmp", "e2/cd/5.tail"):  # e2: a flushed epoch
        (epoch.parent / name).write_bytes(b"\0" * 4096)
    monkeypatch.setenv("SHARAR_SSD_DIR", str(tmp_path / "ssd"))
    assert cli.main(["cache"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("nothing is serving; SSD tier on") and f"{tmp_path / 'ssd'}: 1 prefixes, 2 blocks" in out


def test_two_spark_serve_stays_the_user_for_ssh(home, monkeypatch):
    """The CPU-list namespace makes the process root, and ssh would read root's keys: on two Sparks only the ranks
    restart themselves in it, never the launcher that mirrors the checkout over ssh."""
    from sharar.launch import lxcfs
    wrapped = []
    monkeypatch.setattr(lxcfs, "reexec_if_needed", lambda: wrapped.append(True))
    monkeypatch.setattr(run, "serve", lambda args: 0)
    assert cli.main(["serve", "27b", "--tp", "2"]) == 0 and wrapped == []
    assert cli.main(["serve", "27b"]) == 0 and wrapped == [True]


def test_serve_needs_a_named_model(home, monkeypatch):
    """Serving takes the GPU and may download a model: never one nobody named, but SHARAR_TARGET or the config file names one."""
    started = []
    monkeypatch.setattr(run.guard, "start_thread", lambda **kw: None)
    monkeypatch.setattr(cli, "run_engine", lambda args, parser: started.append(os.environ["SHARAR_TARGET"]) or 0)
    with pytest.raises(SystemExit, match="name a model: 27b, flashnext, nemotron or bolt"):
        cli.main(["serve"])
    assert started == []
    (home / "none.env").write_text("SHARAR_TARGET=bolt\n")
    assert cli.main(["serve"]) == 0 and started == ["bolt"]
    os.environ["SHARAR_TARGET"] = "flashnext"  # not monkeypatch: it would restore the bolt serve installed, after `home`
    assert cli.main(["serve"]) == 0 and started[-1] == "flashnext"


def test_nothing_serving_names_every_target(home, capsys):
    with pytest.raises(SystemExit, match="sharar serve 27b, flashnext, nemotron or bolt first"):
        cli.main(["chat"])
    assert run.status(cli.build_parser().parse_args(["status"])) == 0
    assert "sharar serve 27b, flashnext, nemotron or bolt" in capsys.readouterr().out


def test_serve_refuses_a_busy_port_and_bad_settings(home, monkeypatch):
    with pytest.raises(SystemExit, match="MAX_BATCH"):
        cli.main(["serve", "27b", "--max-batch", "500"])
    monkeypatch.setattr(ctl, "server_info", lambda port: {"target": "qwen38"})
    with pytest.raises(SystemExit, match="already serves"):
        cli.main(["serve", "27b"])


def test_detach_starts_the_same_command_as_a_service(home, monkeypatch):
    started = []
    monkeypatch.setattr(service, "start", lambda unit, argv, log, description: started.append((unit, argv, log)))
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
    assert cli.main(["stop"]) == 0 and stopped == ["sharar-qwen38", "sharar-flashnext", "sharar-nemotron", "sharar-bolt"]
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
    assert "ANTHROPIC_API_KEY" not in env and env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"
    assert env["ANTHROPIC_MODEL"] == "claude-sharar-qwen3.8-27b"  # contains "claude" and names the serving model
    cli.main(["launch", "--target", "nemotron", "claude"])
    assert launched[-1][1]["ANTHROPIC_MODEL"] == "claude-sharar-nemotron-3.5-super"
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


def test_rank0_gets_the_resolved_settings_not_the_callers(monkeypatch):
    """SHARAR_TRACES=off in the caller's environment must not reach rank 0, which would read it as a trace directory."""
    from sharar.launch import tp
    started = []

    class FakePopen:
        def __init__(self, cmd, env=None, **kw):
            started.append((cmd, env))
            self.stdout = iter(())

    monkeypatch.setattr(tp.subprocess, "Popen", FakePopen)
    monkeypatch.setenv("SHARAR_TRACES", "off")
    monkeypatch.setenv("SHARAR_SPEC", "9")
    monkeypatch.setenv("HOME_MARKER", "kept")
    tp.Pair("peer-host", {"SHARAR_SPEC": "4", "SHARAR_TARGET": "qwen38"}, "sharar", ["engine"]).start()
    (_, rank1_env), (_, rank0_env) = started
    assert "SHARAR_TRACES" not in rank0_env and rank0_env["SHARAR_SPEC"] == "4" and rank0_env["HOME_MARKER"] == "kept"
