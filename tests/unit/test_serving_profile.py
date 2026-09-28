"""Serving-profile defaults the launcher resolves (tools/serving-profile.sh, run-sharar.sh)."""
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
KNOBS = ("P_ACCEPT", "P_ACCEPT_SAMPLED", "P_W3", "P_W3_DRAFT", "P_FN_W3", "P_DEVICE_ACCEPT")


def resolve(profile: str, target: str = "", explicit: str | None = None) -> dict:
    """The profile's knob defaults, and the device-acceptance value run-sharar.sh passes the container."""
    script = (f'source "{ROOT}/tools/serving-profile.sh"; serving_profile {profile} "{target}" || exit 3; '
              + "".join(f'echo {k}=${k}; ' for k in KNOBS)
              + 'echo DEVICE_ACCEPT="${SHARAR_DEVICE_ACCEPT:-$P_DEVICE_ACCEPT}"')
    env = {"PATH": "/usr/bin:/bin"} | ({"SHARAR_DEVICE_ACCEPT": explicit} if explicit is not None else {})
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    if out.returncode:
        return {"returncode": out.returncode}
    return dict(line.split("=", 1) for line in out.stdout.split())


@pytest.mark.parametrize("profile, target, device_accept", [
    ("fast", "flashnext", "1"), ("fast", "", "0"), ("fast", "qwen38", "0"),
    ("balanced", "flashnext", "0"), ("exact", "flashnext", "0"), ("exact", "", "0")])
def test_device_acceptance_defaults_on_only_for_fast_flashnext(profile, target, device_accept):
    assert resolve(profile, target)["DEVICE_ACCEPT"] == device_accept


@pytest.mark.parametrize("explicit", ["0", "1"])
def test_an_explicit_device_acceptance_setting_wins(explicit):
    assert resolve("fast", "flashnext", explicit)["DEVICE_ACCEPT"] == explicit
    assert resolve("exact", "", explicit)["DEVICE_ACCEPT"] == explicit


def test_profile_knobs_are_unchanged():
    assert {k: resolve("exact")[k] for k in KNOBS[:5]} == dict(zip(KNOBS, ("off", "0", "0", "0", "off")))
    assert {k: resolve("balanced")[k] for k in KNOBS[:5]} == dict(zip(KNOBS, ("2:1.0", "0", "1", "1", "off")))
    assert {k: resolve("fast")[k] for k in KNOBS[:5]} == dict(zip(KNOBS, ("3:2.0", "1", "1", "1", "auto")))


def test_unknown_profile_is_rejected():
    assert resolve("turbo") == {"returncode": 3}


def test_launcher_uses_the_shared_profile_table():
    text = (ROOT / "run-sharar.sh").read_text()
    assert 'source "$SCRIPT_DIR/tools/serving-profile.sh"' in text
    assert 'DEVICE_ACCEPT="${SHARAR_DEVICE_ACCEPT:-$P_DEVICE_ACCEPT}"' in text
    assert 'args+=(--env "SHARAR_DEVICE_ACCEPT=$DEVICE_ACCEPT")' in text
