"""The launcher's per-target settings: one resolver (tools/launcher-config.sh) and checkpoints pinned as the engine pins them."""
import os
import re
import subprocess
from pathlib import Path

from sharar.model.flashnext import config as flashnext
from sharar.target import FLASHNEXT, QWEN38

ROOT = Path(__file__).resolve().parents[2]


def resolve(*args, **env):
    base = {"PATH": os.environ["PATH"], "HOME": "/home/user", "SHARAR_CONFIG_FILE": "/nonexistent"}
    out = subprocess.run(["bash", str(ROOT / "tools" / "launcher-config.sh"), "print", *args], env={**base, **env},
                         capture_output=True, text=True, check=True).stdout
    return dict(line.split("=", 1) for line in out.splitlines())


def test_without_a_config_file_paths_default_to_the_user_caches():
    env = resolve("flashnext")
    assert env["SHARAR_TARGET"] == "flashnext" and env["SHARAR_PORT"] == "30004" and env["SHARAR_PROPOSER"] == "mtp"
    assert env["SHARAR_HF_CACHE"] == env["SHARAR_FLASHNEXT_CACHE"] == "/home/user/.cache/huggingface"
    assert env["SHARAR_RESULTS"] == "/home/user/.local/share/sharar"
    assert resolve("27b")["SHARAR_TARGET"] == "qwen38" and resolve()["SHARAR_TARGET"] == "qwen38"


def test_the_config_file_and_environment_outrank_the_target_settings(tmp_path):
    conf = tmp_path / "gb10.env"
    conf.write_text("SHARAR_PORT=30100\nSHARAR_HF_CACHE=/models\n")
    env = resolve("flashnext", SHARAR_CONFIG_FILE=str(conf), SHARAR_SPEC="2")
    assert env["SHARAR_PORT"] == "30100" and env["SHARAR_SPEC"] == "2" and env["SHARAR_FLASHNEXT_CACHE"] == "/models"


def test_target_settings_carry_no_host_paths_or_image():
    for f in (ROOT / "config" / "targets").glob("*.env"):
        keys = [line.split("=", 1)[0] for line in f.read_text().splitlines() if line and not line.startswith("#")]
        assert not {"SHARAR_IMAGE", "SHARAR_HF_CACHE", "SHARAR_FLASHNEXT_CACHE", "SHARAR_RESULTS"} & set(keys), f.name


def test_launcher_pins_the_engines_checkpoints():
    text = (ROOT / "run-sharar.sh").read_text()
    pins = dict(re.findall(r'MODEL_REPO="([^"]+)"; MODEL_REVISION="([0-9a-f]{40})"', text))
    assert pins == {QWEN38.model_id: QWEN38.model_revision, FLASHNEXT.model_id: FLASHNEXT.model_revision}
    assert flashnext.MODEL_REVISION == FLASHNEXT.model_revision
