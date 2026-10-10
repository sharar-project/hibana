"""Serving-profile defaults the launcher resolves (tools/serving-profile.sh, run-sharar.sh)."""
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
KNOBS = ("P_ACCEPT", "P_ACCEPT_SAMPLED", "P_W3", "P_DEVICE_ACCEPT", "P_MAX_BATCH", "P_SPEC_DEPTHS", "P_ACCEPT_SAMPLED_THRESHOLD")


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
    assert {k: resolve("exact")[k] for k in KNOBS[:3]} == dict(zip(KNOBS, ("off", "0", "0")))
    assert {k: resolve("balanced")[k] for k in KNOBS[:3]} == dict(zip(KNOBS, ("2:1.0", "0", "1")))
    assert {k: resolve("fast")[k] for k in KNOBS[:3]} == dict(zip(KNOBS, ("3:2.0", "1", "1")))
    assert {k: resolve("fast", "flashnext")[k] for k in KNOBS[:3]} == dict(zip(KNOBS, ("3:2.0", "1", "1")))


@pytest.mark.parametrize("profile, target, threshold", [
    ("fast", "", "2:1.0"), ("fast", "qwen38", "2:1.0"), ("throughput", "", "2:1.0"), ("fast", "flashnext", ""),
    ("throughput", "flashnext", ""), ("balanced", "", ""), ("exact", "", "")])
def test_the_27b_samples_at_a_tighter_threshold_under_fast(profile, target, threshold):
    # #11: at 3:2.0 the 27B's sampled thinking looped to max_tokens in Claude Code sessions, at 2:1.0 it did not
    assert resolve(profile, target)["P_ACCEPT_SAMPLED_THRESHOLD"] == threshold


@pytest.mark.parametrize("profile, target, w3", [
    ("exact", "", "0"), ("exact", "flashnext", "0"), ("balanced", "", "1"), ("balanced", "flashnext", "0"),
    ("fast", "", "1"), ("fast", "flashnext", "1"), ("throughput", "", "1"), ("throughput", "flashnext", "1")])
def test_the_profile_alone_turns_the_3bit_copies_on(profile, target, w3):
    # #72: one switch for both targets; Flash-Next's 3-bit trunk is fast's (and throughput's), not balanced's
    assert resolve(profile, target)["P_W3"] == w3


def test_unknown_profile_is_rejected():
    assert resolve("turbo") == {"returncode": 3}


def test_launcher_uses_the_shared_profile_table():
    text = (ROOT / "run-sharar.sh").read_text()
    assert 'source "$SCRIPT_DIR/tools/serving-profile.sh"' in text
    assert 'DEVICE_ACCEPT="${SHARAR_DEVICE_ACCEPT:-$P_DEVICE_ACCEPT}"' in text
    assert 'args+=(--env "SHARAR_DEVICE_ACCEPT=$DEVICE_ACCEPT")' in text
    assert 'ACCEPT_DEFAULT_SAMPLED_THRESHOLD="${SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD:-$P_ACCEPT_SAMPLED_THRESHOLD}"' in text
    assert '--accept-default-sampled-threshold "$ACCEPT_DEFAULT_SAMPLED_THRESHOLD"' in text


@pytest.mark.parametrize("target, depths", [("flashnext", "16:3"), ("", ""), ("qwen38", "")])
def test_throughput_is_fast_with_sixteen_requests(target, depths):
    # #59: Flash-Next verifies within the 64 rows its 3-bit trunk copies serve at 16 requests (16 x 4);
    # the 27B keeps its full depth at every batch (operator decision, GSM8K x1.74).
    fast, wide = resolve("fast", target), resolve("throughput", target)
    assert {k: wide[k] for k in KNOBS[:4]} == {k: fast[k] for k in KNOBS[:4]}
    # Flash-Next decodes up to 128 requests on one Spark with its 3-bit experts (607 tok/s against 230 at 16)
    assert wide["P_MAX_BATCH"] == ("128" if target == "flashnext" else "16") and wide["P_SPEC_DEPTHS"] == depths
    for bucket, depth in (item.split(":") for item in filter(None, depths.split(","))):
        assert int(bucket) * (int(depth) + 1) <= 64 and int(depth) <= 4


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
