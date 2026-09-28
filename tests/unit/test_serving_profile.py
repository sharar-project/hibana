"""Serving-profile defaults the launcher resolves (tools/serving-profile.sh, run-sharar.sh)."""
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
KNOBS = ("P_ACCEPT", "P_ACCEPT_SAMPLED", "P_W3", "P_W3_DRAFT", "P_FN_W3", "P_DEVICE_ACCEPT", "P_MAX_BATCH", "P_SPEC_DEPTHS")


def resolve(profile: str, target: str = "", explicit: str | None = None) -> dict:
    """The profile's knob defaults, and the device-acceptance value run-sharar.sh passes the container."""
    script = (f'source "{ROOT}/tools/serving-profile.sh"; serving_profile {profile} "{target}" || exit 3; '
              + "".join(f'echo {k}=\"${k}\"; ' for k in KNOBS)
              + 'echo DEVICE_ACCEPT="${SHARAR_DEVICE_ACCEPT:-$P_DEVICE_ACCEPT}"')
    env = {"PATH": "/usr/bin:/bin"} | ({"SHARAR_DEVICE_ACCEPT": explicit} if explicit is not None else {})
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    if out.returncode:
        return {"returncode": out.returncode}
    return dict(line.split("=", 1) for line in out.stdout.splitlines())


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


@pytest.mark.parametrize("target, depths", [("flashnext", "16:3"), ("", "8:7,16:3"), ("qwen38", "8:7,16:3")])
def test_throughput_is_fast_with_sixteen_requests_and_verifies_within_64_rows(target, depths):
    # #59: every verification stays within the 64 rows the 3-bit copies serve (Flash-Next 16 x 4, the 27B
    # 8 x 8 and 16 x 4), so a full batch runs fast's weights.
    fast, wide = resolve("fast", target), resolve("throughput", target)
    assert {k: wide[k] for k in KNOBS[:6]} == {k: fast[k] for k in KNOBS[:6]}
    assert wide["P_MAX_BATCH"] == "16" and wide["P_SPEC_DEPTHS"] == depths
    k = 4 if target == "flashnext" else 15
    for bucket, depth in (item.split(":") for item in depths.split(",")):
        assert int(bucket) * (int(depth) + 1) <= 64 and int(depth) <= k


@pytest.mark.parametrize("profile", ["exact", "balanced", "fast"])
def test_other_profiles_keep_batch_four_and_one_depth(profile):
    for target in ("", "flashnext"):
        knobs = resolve(profile, target)
        assert knobs["P_MAX_BATCH"] == "4" and knobs["P_SPEC_DEPTHS"] == ""


def test_launcher_forwards_batch_and_depths():
    text = (ROOT / "run-sharar.sh").read_text()
    assert 'MAX_BATCH="${SHARAR_MAX_BATCH:-$P_MAX_BATCH}"' in text
    assert 'SPEC_DEPTHS="${SHARAR_SPEC_DEPTHS:-$P_SPEC_DEPTHS}"' in text
    assert 'serve_args+=(--max-batch "$MAX_BATCH")' in text
