"""`sharar serve nemotron`'s settings (sharar/launch/settings.py): the target's port and drafter, its profiles, the
pinned snapshot the preflight checks."""
import pytest

from sharar.launch import preflight
from sharar.launch import settings as launch
from sharar.model.nemotron import config as nemotron


@pytest.fixture
def env(tmp_path):
    return {"HOME": str(tmp_path), "SHARAR_CONFIG_FILE": str(tmp_path / "none.env")}


def resolve(env, profile=None, **kw):
    return launch.resolve("nemotron", profile, env=env, **kw)


@pytest.mark.parametrize("name", ["nemotron", "nemotron-super", "Nemotron"])
def test_names_port_service_and_drafter(env, name):
    s = launch.resolve(name, env=env)
    assert (s.target, s.port, s.service, s.proposer, s.get("SHARAR_VERIFY_STATE")) == (
        "nemotron", 30005, "sharar-nemotron", "mtp", "reconstruct")
    assert s.model_cache == s.hf_cache and s.ssd == s.results / "ssd" / "nemotron"
    assert s.get("SHARAR_DRAFT_PRECISION") == "bf16" and not s.parallel_draft


def test_exact_is_the_target_alone(env):
    s = resolve(env, "exact")
    assert (s.spec, s.get("SHARAR_ACCEPT_DEFAULT"), s.get("SHARAR_ACCEPT_DEFAULT_SAMPLED"), s.get("SHARAR_DEVICE_ACCEPT"),
            s.get("SHARAR_MOE_KEEP"), s.get("SHARAR_SPEC_DEPTHS"), s.get("SHARAR_MAX_BATCH")) == (3, "off", "0", "0", "", "", "4")
    args = s.engine_args()
    assert args[args.index("--spec") + 1] == "3" and args[args.index("--accept-default") + 1] == "off"


def test_exact_serves_the_checkpoint_alone(env):
    s = resolve(env, "exact")
    assert s.get("SHARAR_W3_DECODE") == "0" and preflight.ensure_w3(s, fetch=False) is None and not s.warnings


@pytest.mark.parametrize("profile", ["fast", "throughput"])
def test_the_3bit_profiles_need_the_calibrated_copies(env, tmp_path, profile):
    s = resolve(env, profile)
    assert s.get("SHARAR_W3_DECODE") == "1"
    with pytest.raises(preflight.PreflightError, match="calibrated 3-bit copies"):
        preflight.ensure_w3(s, fetch=False)  # no checkpoint: no fallback
    current = s.w3_root / "nemotron-current"
    current.mkdir(parents=True)
    (current / "manifest.json").write_text("{}")
    s = resolve(env, profile)
    assert preflight.ensure_w3(s, fetch=False) == current.resolve() and not s.warnings
    assert s.engine_env(current)["SHARAR_W3_DECODE"] == "1"


def test_fast_and_throughput_share_experts_and_halve_the_state(env):
    fast = resolve(env, "fast")
    assert (fast.get("SHARAR_MOE_KEEP"), fast.get("SHARAR_MOE_BUDGET"), fast.get("SHARAR_SSM_STATE")) == ("12", "48:128", "fp16_sr")
    tput, exact = resolve(env, "throughput"), resolve(env, "exact")
    assert (tput.get("SHARAR_MOE_KEEP"), tput.get("SHARAR_MOE_BUDGET"), tput.get("SHARAR_SSM_STATE")) == ("12", "48:128", "fp16_sr")
    assert exact.get("SHARAR_MOE_KEEP", "0") == "0" and not exact.get("SHARAR_MOE_BUDGET")
    assert exact.get("SHARAR_SSM_STATE", "fp32") == "fp32"


def test_fast_and_throughput(env):
    fast, tput = resolve(env, "fast"), resolve(env, "throughput")
    for s in (fast, tput):
        assert s.get("SHARAR_ACCEPT_DEFAULT") == "3:2.0" and s.get("SHARAR_ACCEPT_DEFAULT_SAMPLED") == "1"
        assert "--accept-default-sampled" in s.engine_args()
    assert fast.get("SHARAR_MAX_BATCH") == "4" and tput.get("SHARAR_MAX_BATCH") == "16"


def test_an_explicit_setting_wins(env):
    s = resolve({**env, "SHARAR_SPEC": "4", "SHARAR_ACCEPT_DEFAULT": "2:1.0"}, "fast")
    assert s.spec == 4 and s.get("SHARAR_ACCEPT_DEFAULT") == "2:1.0"


def test_the_mamba_state_takes_every_format(env):
    """FP16 (Mamba-2's) and the 8- and 4-bit windows; the GDN targets refuse FP16."""
    for state in ("fp32", "fp16_sr", "int8", "int4"):
        assert resolve(env, "exact", overrides={"SHARAR_SSM_STATE": state}).get("SHARAR_SSM_STATE") == state
    with pytest.raises(ValueError, match="SHARAR_SSM_STATE must be"):
        launch.resolve("27b", "exact", env=env, overrides={"SHARAR_SSM_STATE": "fp16_sr"})


def test_one_spark_only(env):
    with pytest.raises(ValueError, match="one Spark"):
        resolve(env, tp=2)


def test_preflight_checks_the_pinned_snapshot(env, tmp_path):
    s = resolve(env)
    assert preflight.pinned_snapshots(s) == [(s.hf_cache, nemotron.MODEL_ID, nemotron.MODEL_REVISION)]
    assert preflight.ensure_snapshots(s, fetch=False) == [f"{nemotron.MODEL_ID}@{nemotron.MODEL_REVISION[:7]}"]
    assert launch.model_config("nemotron") is nemotron


@pytest.mark.parametrize("kv", ["nvfp4", "sim-fp4", "sim-k8v4", "sim-bf16"])
def test_fp8_kv_only(env, kv):
    """The attention's fused K/V writer serves the FP8 pool: the NVFP4 cache and the accuracy screens are refused."""
    with pytest.raises(ValueError, match="FP8 KV cache only"):
        resolve({**env, "SHARAR_KV_FORMAT": kv}, "fast")
    assert resolve(env, "fast").get("SHARAR_KV_FORMAT", "fp8") == "fp8"
