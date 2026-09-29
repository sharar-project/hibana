"""The lossy-change gate's host logic (tools/lossy-gate.py): profile resolution, statistics, verdict rules."""
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _gate():
    loader = SourceFileLoader("lossy_gate_tool", str(REPO / "tools" / "lossy-gate.py"))
    spec = importlib.util.spec_from_loader("lossy_gate_tool", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


gate = _gate()


def test_profiles_come_from_the_launcher_table():
    """The gate reads tools/serving-profile.sh (as run-sharar.sh does), per target."""
    assert gate.profile_defaults("exact", "flashnext")["SHARAR_ACCEPT_DEFAULT"] == "off"
    assert gate.profile_defaults("fast", "flashnext")["SHARAR_DEVICE_ACCEPT"] == "1"
    assert gate.profile_defaults("fast", "qwen38")["SHARAR_DEVICE_ACCEPT"] == "0"
    wide = gate.profile_defaults("throughput", "flashnext")
    assert wide["SHARAR_MAX_BATCH"] == "16" and wide["SHARAR_SPEC_DEPTHS"] == "16:3"
    for name in gate.PROFILE_NAMES:
        assert set(gate.profile_defaults(name, "qwen38")) == set(gate.PROFILE_SETTINGS.values())
    with pytest.raises(ValueError, match="unknown serving profile"):
        gate.profile_defaults("turbo", "qwen38")


def test_batch_settings_stay_out_of_the_in_process_arm():
    env = gate.resolve({"SHARAR_TARGET": "flashnext", "SHARAR_DRAFT": "pinned", "SHARAR_FLASHNEXT_CACHE": "/fn"}, "throughput", {})
    plan = gate.arm_plan(env)
    assert plan["env"]["SHARAR_DEVICE_ACCEPT"] == "1"
    assert "SHARAR_MAX_BATCH" not in plan["env"] and "SHARAR_SPEC_DEPTHS" not in plan["env"]


def test_resolve_explicit_settings_win_over_the_profile(tmp_path):
    config = {"SHARAR_TARGET": "flashnext", "SHARAR_W3_DECODE": "0", "SHARAR_RESULTS": str(tmp_path), "SHARAR_DRAFT": "pinned"}
    fast = gate.resolve(config, "fast", {})
    assert fast["SHARAR_W3_DECODE"] == "0"  # the config file sets it, as config/flashnext.env does
    assert fast["SHARAR_ACCEPT_DEFAULT"] == "3:2.0"
    assert fast["SHARAR_FLASHNEXT_W3_CHECKPOINT"] == "off"  # auto without flashnext-current
    ck = tmp_path / "w3-checkpoints" / "flashnext-current"
    ck.mkdir(parents=True)
    (ck / "manifest.json").write_text("{}")
    assert gate.resolve(config, "fast", {})["SHARAR_FLASHNEXT_W3_CHECKPOINT"] == str(ck.resolve())
    near = gate.resolve(config, "exact", {"SHARAR_FLASHNEXT_TRUNK": "w3"})
    plan = gate.arm_plan({**near, "SHARAR_FLASHNEXT_CACHE": "/fn"})
    assert plan["env"]["SHARAR_FLASHNEXT_TRUNK"] == "w3" and plan["accept"] == "1:0.0"
    assert "SHARAR_FLASHNEXT_W3_CHECKPOINT" not in plan["env"]


def test_overrides_reach_the_in_process_arm():
    env = gate.resolve({"SHARAR_DRAFT": "pinned"}, "exact", {"SHARAR_TRUNK_KEEP_BF16": "mlp", "SHARAR_PORT": "1"})
    plan = gate.arm_plan(env)
    assert plan["env"]["SHARAR_TRUNK_KEEP_BF16"] == "mlp"
    assert "SHARAR_PORT" not in plan["env"]
    assert plan["proposer"] == "dflash2" and plan["spec"] == 15 and plan["draft_precision"] == "fp8"


def test_accept_rule_matches_serve():
    assert gate.accept_rule("off") == "1:0.0"
    assert gate.accept_rule("3:2") == "3:2.0"


def test_paired_accuracy_and_mcnemar():
    base = [True] * 90 + [False] * 10
    cand = [True] * 80 + [False] * 10 + [True] * 2 + [False] * 8
    r = gate.paired_acc(base, cand)
    assert (r["wins"], r["losses"]) == (2, 10)
    assert r["delta_pp"] == pytest.approx(-8.0)
    assert r["ci95_pp"][1] < 0
    assert r["p"] == pytest.approx(0.0386, abs=1e-3)
    assert gate.mcnemar_p(0, 0) == 1.0


def test_ci_mean_resamples_groups():
    values = [0.0] * 50 + [1.0] * 50
    m, lo, hi = gate.ci_mean(values, [i // 50 for i in range(100)])
    assert m == 0.5 and lo == 0.0 and hi == 1.0  # two groups: every resample is 0, 0.5 or 1


def _l1(excess, delta):
    d = {"kl_excess": excess, "argmax_delta": delta, "within_floor": excess[1] <= 0 and delta[2] >= 0,
         "kl": 0.1, "kl_floor": 0.1, "argmax": 0.9, "argmax_floor": 0.9, "n": 10}
    return {"code": d, "all": d}


def _l2(ratio, lo, hi, tpr=(0.99, 1.01)):
    return {"decode_tps": {"n": 28, "ratio": ratio, "ci95": [lo, hi]}, "tok_per_round": {"n": 28, "ratio": 1.0, "ci95": list(tpr)},
            "within_floor": tpr[0] <= 1 <= tpr[1]}


def _l3(wins, losses, n=700):
    base = [True] * (n - 50) + [False] * 50
    cand = list(base)
    for i in range(losses):
        cand[i] = False
    for i in range(wins):
        cand[n - 50 + i] = True
    r = gate.paired_acc(base, cand)
    return {"tasks": {"gsm8k": r}, "pooled": r}


def test_verdict_rules():
    same = _l1((-0.01, -0.02, 0.0), (0.0, -0.001, 0.002))
    assert gate.verdict(same, _l2(1.0, 0.99, 1.01), None, None)[0] == "exact"
    lossy = _l1((0.05, 0.04, 0.06), (-0.02, -0.03, -0.01))
    assert gate.verdict(lossy, _l2(0.99, 0.97, 1.0), None, None)[0] == "rejected"  # no speed gain
    assert gate.verdict(lossy, _l2(1.03, 1.01, 1.05), None, None)[0] == "rejected"  # within the speed noise
    assert gate.verdict(lossy, _l2(1.2, 1.1, 1.3), None, None)[0] == "undecided"
    assert gate.verdict(lossy, _l2(1.2, 1.1, 1.3), _l3(2, 14), "screen")[0] == "rejected"
    assert gate.verdict(lossy, _l2(1.2, 1.1, 1.3), _l3(6, 8), "screen")[0] == "fast"
    assert gate.verdict(lossy, _l2(1.2, 1.1, 1.3), _l3(6, 8, n=3000), "full")[0] == "balanced"
    assert gate.screen_is_close(_l3(6, 8))
    assert not gate.screen_is_close(_l3(2, 14))


def test_shards_cover_the_sets_and_reuse_the_screen():
    screen = {gate.shard_path(Path("x"), s) for s in gate.shards(gate.SCREEN)}
    full = {gate.shard_path(Path("x"), s) for s in gate.shards(gate.FULL)}
    assert sum(gate.shard_items(s) for s in gate.shards(gate.FULL)) == 1319 + 164 + 500 + 2750
    assert {p for p in screen if "gsm8k" in p.name or "humaneval" in p.name} <= full
