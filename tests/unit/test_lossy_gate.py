"""The lossy-change gate's host logic (tools/lossy_gate.py): profile resolution, statistics, verdict rules."""
import importlib.util
import math
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _gate():
    loader = SourceFileLoader("lossy_gate_tool", str(REPO / "tools" / "lossy_gate.py"))
    spec = importlib.util.spec_from_loader("lossy_gate_tool", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


gate = _gate()


@pytest.fixture
def clean(monkeypatch, tmp_path):
    """The gate resolves arms from the config file alone: point it at an empty one and a results directory."""
    for key in [k for k in gate.os.environ if k.startswith("HIBANA_")]:
        monkeypatch.delenv(key)
    conf = tmp_path / "gb10.env"
    conf.write_text(f"HIBANA_RESULTS={tmp_path}\nHIBANA_FLASHNEXT_CACHE=/fn\n")
    monkeypatch.setenv("HIBANA_CONFIG_FILE", str(conf))
    return tmp_path


def test_arms_resolve_as_sharar_serve_does(clean):
    """The gate and `sharar serve` share one resolver (hibana/launch/settings.py)."""
    exact = gate.resolve("flashnext", "exact", {})
    assert exact.get("HIBANA_ACCEPT_DEFAULT") == "off" and gate.arm_plan(exact)["accept"] == "1:0.0"
    assert gate.resolve("flashnext", "fast", {}).get("HIBANA_DEVICE_ACCEPT") == "1"
    assert gate.resolve("qwen38", "fast", {}).get("HIBANA_DEVICE_ACCEPT") == "0"
    with pytest.raises(ValueError, match="unknown serving profile"):
        gate.resolve("qwen38", "turbo", {})


def test_batch_settings_stay_out_of_the_in_process_arm(clean):
    plan = gate.arm_plan(gate.resolve("flashnext", "throughput", {}))
    assert plan["env"]["HIBANA_DEVICE_ACCEPT"] == "1"
    assert "HIBANA_MAX_BATCH" not in plan["env"] and "HIBANA_SPEC_DEPTHS" not in plan["env"] and "HIBANA_SSD_DIR" not in plan["env"]


def test_the_3bit_checkpoint_is_what_current_holds(clean):
    fast = gate.arm_plan(gate.resolve("flashnext", "fast", {}))
    # without flashnext-current: Flash-Next's copies are calibrated only, so the trunk stays FP8 (the gate does not fetch)
    assert fast["env"]["HIBANA_W3_DECODE"] == "0" and "HIBANA_W3_CHECKPOINT" not in fast["env"]
    ck = clean / "w3-checkpoints" / "flashnext-current"
    ck.mkdir(parents=True)
    (ck / "manifest.json").write_text("{}")
    fast = gate.arm_plan(gate.resolve("flashnext", "fast", {}))
    assert fast["env"]["HIBANA_W3_DECODE"] == "1" and fast["env"]["HIBANA_W3_CHECKPOINT"] == str(ck.resolve())
    assert "HIBANA_W3_CHECKPOINT" not in gate.arm_plan(gate.resolve("flashnext", "fast", {"HIBANA_W3_DECODE": "0"}))["env"]
    exact = gate.arm_plan(gate.resolve("flashnext", "exact", {}))
    assert exact["env"]["HIBANA_W3_DECODE"] == "0" and "HIBANA_W3_CHECKPOINT" not in exact["env"] and exact["accept"] == "1:0.0"


def test_overrides_reach_the_in_process_arm(clean):
    plan = gate.arm_plan(gate.resolve("qwen38", "exact", {"HIBANA_TRUNK_KEEP_BF16": "mlp", "HIBANA_DRAFT": "pinned"}))
    assert plan["env"]["HIBANA_TRUNK_KEEP_BF16"] == "mlp"
    assert plan["proposer"] == "dflash2" and plan["spec"] == 15 and plan["draft_precision"] == "fp8" and plan["draft"] is None


def test_sampled_rule_reaches_the_in_process_arm_as_serve_records_it(clean):
    fast = gate.arm_plan(gate.resolve("flashnext", "fast", {}))
    assert fast["env"]["HIBANA_ACCEPT_SOFT"] == "0.5"
    off = gate.arm_plan(gate.resolve("qwen38", "fast", {"HIBANA_DRAFT": "pinned"}))
    assert off["env"]["HIBANA_ACCEPT_SOFT"] == "0"
    nospec = gate.arm_plan(gate.resolve("qwen38", "fast", {"HIBANA_SPEC": "0"}))
    assert nospec["env"]["HIBANA_ACCEPT_SOFT"] == "0" and nospec["accept"] == "1:0.0"


def test_served_arms_are_sharar_serve_in_the_foreground(clean):
    cmd = gate.served_command("flashnext", "fast", 30100)
    assert cmd[2:] == ["-m", "hibana", "serve", "flashnext", "--fast", "--port", "30100", "--no-fetch"]
    assert "HIBANA_CONFIG_FILE" in gate.served_env({}) and gate.served_env({"HIBANA_SSD": "1"})["HIBANA_SSD"] == "0"
    assert gate.served_env({"HIBANA_TRACES": "/x"})["HIBANA_TRACES"] == "off"


def test_accept_rule_matches_serve():
    assert gate.accept_rule("off") == "1:0.0"
    assert gate.accept_rule("3:2") == "3:2.0"


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


def _paired(wins, losses, n, kind="binary"):
    """A task row as sharar-eval's `evals.suite compare --json` writes it (McNemar, normal interval of the paired difference)."""
    d = (wins - losses) / n
    se = math.sqrt(max(0.0, (wins + losses) / n - d * d) / n)
    k = wins + losses
    p = min(1.0, sum(math.comb(k, i) for i in range(k + 1) if abs(i - k / 2) >= abs(wins - k / 2)) / 2 ** k) if k else 1.0
    return {"kind": kind, "n": n, "delta": 100 * d, "ci95": [100 * (d - 1.96 * se), 100 * (d + 1.96 * se)],
            "wins": wins, "losses": losses, "p": p}


def _l3(wins, losses, n=700):
    r = _paired(wins, losses, n)
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
    v, why = gate.verdict(lossy, _l2(1.2, 1.1, 1.3), _l3(6, 8, n=3000), "full")
    assert v == "fast" and "non-inferior at 1.0 pp" in why[0]
    assert gate.screen_is_close(_l3(6, 8))
    assert not gate.screen_is_close(_l3(2, 14))


def test_added_batteries_reject_on_pass_fail_tasks_and_ruler_only():
    lossy, fast = _l1((0.05, 0.04, 0.06), (-0.02, -0.03, -0.01)), _l2(1.2, 1.1, 1.3)
    wmt = {**_paired(300, 700, 1000, "score"), "delta": -1.0}  # a significant chrF++ loss decides nothing
    assert gate.verdict(lossy, fast, _l3(6, 8), "screen", {"hard": {"tasks": {"wmt": wmt}}})[0] == "fast"
    hard = {"tasks": {"mmlupro": _paired(43, 72, 1400), "wmt": wmt}}
    v, why = gate.verdict(lossy, fast, _l3(6, 8), "screen", {"hard": hard})
    assert v == "rejected" and "mmlupro" in why[0] and "wmt" not in why[0]
    ruler = {"tasks": {"ruler2-32k": _paired(10, 12, 120, "score"), "ruler2-128k": _paired(8, 30, 120, "score")}}
    v, why = gate.verdict(lossy, fast, _l3(6, 8), "screen", {"long": ruler})
    assert v == "rejected" and "ruler2-128k" in why[0] and "ruler2-32k" not in why[0]
    assert gate.BATTERIES["all"] == ["hard", "long"] and gate.STAGES["long"][0] == "long"
    assert gate.STAGES["agentic"] == ("agentic", ["tbench"])


def test_stages_reuse_the_screen_shards():
    """The screen and full stages are sharar-eval suites; full's GSM8K starts with the screen's 300 items."""
    assert gate.STAGES["screen"][0] == "quick" and gate.STAGES["full"][0] == "full"
    assert {"gsm8k", "humaneval"} <= set(gate.STAGES["screen"][1]) & set(gate.STAGES["full"][1])
