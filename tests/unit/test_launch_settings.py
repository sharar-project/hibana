"""`sharar serve`'s settings (sharar/launch/settings.py): sources, profiles, validation, what the engine receives."""
import pytest

from sharar.launch import settings as launch
from sharar.model import qwen38_config
from sharar.model.flashnext import config as flashnext
from sharar.target import FLASHNEXT, QWEN38


@pytest.fixture
def env(tmp_path):
    """A clean environment: no config file, a fresh home."""
    return {"HOME": str(tmp_path), "SHARAR_CONFIG_FILE": str(tmp_path / "none.env")}


def resolve(env, target=None, profile=None, overrides=None, **kw):
    return launch.resolve(target, profile, overrides, env=env, **kw)


def test_without_a_config_file_paths_default_to_the_user_caches(env, tmp_path, monkeypatch):
    monkeypatch.setattr(launch.Path, "home", lambda: tmp_path)
    s = resolve(env, "flashnext")
    assert (s.target, s.port, s.proposer, s.service) == ("flashnext", 30004, "mtp", "sharar-flashnext")
    assert s.hf_cache == s.model_cache == tmp_path / ".cache" / "huggingface"
    assert s.results == tmp_path / ".local" / "share" / "sharar"
    assert resolve(env, "27b").target == "qwen38" and resolve(env).target == "qwen38"
    assert s.max_context == 262144 and s.traces == s.results / "traces" and s.sessions == s.results / "sessions"
    with pytest.raises(ValueError, match="unknown target"):
        resolve(env, "llama")


def test_command_line_then_environment_then_file_then_target(env, tmp_path):
    conf = tmp_path / "gb10.env"
    conf.write_text("SHARAR_PORT=30100\nSHARAR_HF_CACHE=~/models\nSHARAR_SPEC=3\n# SHARAR_SPEC=9\n")
    env["SHARAR_CONFIG_FILE"] = str(conf)
    s = resolve({**env, "SHARAR_SPEC": "2"}, "flashnext")
    assert s.port == 30100 and s.spec == 2 and s.model_cache == tmp_path / "models"  # ~ expanded with HOME
    assert resolve({**env, "SHARAR_SPEC": "2"}, "flashnext", overrides={"SHARAR_SPEC": "1"}).spec == 1


@pytest.mark.parametrize("profile, target, device_accept", [
    ("fast", "flashnext", "1"), ("fast", "27b", "0"), ("balanced", "flashnext", "0"), ("exact", "flashnext", "0"), ("exact", "27b", "0")])
def test_device_acceptance_defaults_on_only_for_fast_flashnext(env, profile, target, device_accept):
    assert resolve(env, target, profile).get("SHARAR_DEVICE_ACCEPT") == device_accept


@pytest.mark.parametrize("explicit", ["0", "1"])
def test_an_explicit_setting_wins_over_the_profile(env, explicit):
    assert resolve({**env, "SHARAR_DEVICE_ACCEPT": explicit}, "flashnext", "fast").get("SHARAR_DEVICE_ACCEPT") == explicit
    assert resolve({**env, "SHARAR_DEVICE_ACCEPT": explicit}, "27b", "exact").get("SHARAR_DEVICE_ACCEPT") == explicit


def test_profile_knobs():
    pick = lambda p, t: {k: launch.profile_defaults(p, t)[k] for k in ("SHARAR_ACCEPT_DEFAULT", "SHARAR_ACCEPT_DEFAULT_SAMPLED", "SHARAR_W3_DECODE")}  # noqa: E731
    assert pick("exact", "qwen38") == {"SHARAR_ACCEPT_DEFAULT": "off", "SHARAR_ACCEPT_DEFAULT_SAMPLED": "0", "SHARAR_W3_DECODE": "0"}
    assert pick("balanced", "qwen38") == {"SHARAR_ACCEPT_DEFAULT": "2:1.0", "SHARAR_ACCEPT_DEFAULT_SAMPLED": "0", "SHARAR_W3_DECODE": "1"}
    assert pick("fast", "qwen38") == pick("fast", "flashnext") == {"SHARAR_ACCEPT_DEFAULT": "3:2.0", "SHARAR_ACCEPT_DEFAULT_SAMPLED": "1", "SHARAR_W3_DECODE": "1"}
    with pytest.raises(ValueError, match="unknown serving profile"):
        launch.profile_defaults("turbo", "qwen38")


@pytest.mark.parametrize("profile, target, threshold", [
    ("fast", "qwen38", "2:1.0"), ("throughput", "qwen38", "2:1.0"), ("fast", "flashnext", None), ("throughput", "flashnext", None),
    ("balanced", "qwen38", None), ("exact", "qwen38", None)])
def test_the_27b_samples_at_a_tighter_threshold_under_fast(profile, target, threshold):
    # #11: at 3:2.0 the 27B's sampled thinking looped to max_tokens in Claude Code sessions, at 2:1.0 it did not
    assert launch.profile_defaults(profile, target).get("SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD") == threshold


@pytest.mark.parametrize("profile, target, w3", [
    ("exact", "qwen38", "0"), ("exact", "flashnext", "0"), ("balanced", "qwen38", "1"), ("balanced", "flashnext", "0"),
    ("fast", "qwen38", "1"), ("fast", "flashnext", "1"), ("throughput", "qwen38", "1"), ("throughput", "flashnext", "1")])
def test_the_profile_alone_turns_the_3bit_copies_on(profile, target, w3):
    # #72: one switch for both targets; Flash-Next's 3-bit trunk is fast's (and throughput's), not balanced's
    assert launch.profile_defaults(profile, target)["SHARAR_W3_DECODE"] == w3


@pytest.mark.parametrize("target, experts, batch, depths", [
    ("flashnext", "w3", "128", "16:3"), ("flashnext", "nvfp4", "64", "16:3"), ("qwen38", "w3", "16", None)])
def test_throughput_decodes_more_requests(target, experts, batch, depths):
    # #59: Flash-Next verifies within the 64 rows its 3-bit copies serve; the 27B keeps its full depth (GSM8K x1.74)
    knobs = launch.profile_defaults("throughput", target, experts)
    assert knobs["SHARAR_MAX_BATCH"] == batch and knobs.get("SHARAR_SPEC_DEPTHS") == depths
    for profile in ("exact", "balanced", "fast"):
        assert launch.profile_defaults(profile, target)["SHARAR_MAX_BATCH"] == "4"


@pytest.mark.parametrize("override, message", [
    ({"SHARAR_MAX_BATCH": "129"}, "MAX_BATCH"), ({"SHARAR_SPEC_DEPTHS": "8"}, "SPEC_DEPTHS"),
    ({"SHARAR_DRAFT_PRECISION": "fp4"}, "DRAFT_PRECISION"), ({"SHARAR_PROPOSER": "dspark"}, "SHARAR_PROPOSER"),
    ({"SHARAR_PROPOSER": "ngram"}, "SHARAR_PROPOSER"),
    ({"SHARAR_PROPOSER": "mtp"}, "FP8 draft precision"), ({"SHARAR_FLASHNEXT_EXPERTS": "fp4"}, "EXPERTS")])
def test_bad_settings_are_refused(env, override, message):
    with pytest.raises(ValueError, match=message):
        resolve(env, "27b", overrides=override)


def test_throughput_flashnext_batch_passes_validation(env):
    # the launcher once refused SHARAR_MAX_BATCH above 16, so `serve flashnext --throughput` (128) failed
    assert resolve(env, "flashnext", "throughput").get("SHARAR_MAX_BATCH") == "128"


def test_the_engine_gets_explicit_flags_and_its_environment(env, tmp_path):
    s = resolve({**env, "SHARAR_DRAFT": "pinned"}, "27b", "fast")
    args = s.engine_args()
    assert args[args.index("--port") + 1] == "30001" and args[args.index("--spec") + 1] == "15"
    assert args[args.index("--context-length") + 1] == "262144" and args[args.index("--draft-precision") + 1] == "fp8"
    assert args[args.index("--accept-default") + 1] == "3:2.0" and args[args.index("--accept-default-sampled-threshold") + 1] == "2:1.0"
    assert "--draft" not in args and args[args.index("--traces") + 1] == str(s.traces)
    e = s.engine_env(tmp_path / "w3")
    assert e["SHARAR_W3_CHECKPOINT"] == str(tmp_path / "w3") and e["SHARAR_SESSION_DIR"] == str(s.sessions)
    assert "SHARAR_DRAFT" not in e and "SHARAR_TRACES" not in e  # the flags carry them
    assert resolve(env, "27b", traces=False).traces is None and "--traces" not in resolve(env, "27b", traces=False).engine_args()
    assert resolve(env, "27b", overrides={"SHARAR_SESSION_DIR": "off"}).sessions is None
    assert resolve(env, "27b", tp=2).sessions.name == "sessions-tp" and "--tp" in resolve(env, "27b", tp=2).engine_args(tp=2)


def test_flashnext_without_a_checkpoint_serves_its_fp8_trunk(env):
    e = resolve(env, "flashnext", "fast").engine_env(None)
    assert e["SHARAR_W3_DECODE"] == "0" and e["SHARAR_FLASHNEXT_CACHE"] and e["SHARAR_DRAFT_HEAD"] == "w3"


@pytest.mark.parametrize("policy", ["auto", "cute", "triton"])
def test_gdn_prefill_backend_reaches_engine_through_new_launcher(env, policy):
    from sharar.engine.config import EngineConfig
    resolved = resolve({**env, "SHARAR_GDN_PREFILL": policy}, "27b")
    engine_env = resolved.engine_env(None)
    assert engine_env["SHARAR_GDN_PREFILL"] == policy
    assert EngineConfig.from_env(engine_env).gdn_prefill == policy


def test_a_promoted_draft_is_served_only_when_it_matches_the_proposer(env, tmp_path):
    runs = tmp_path / "results" / "draft-runs"
    best = runs / "run-1" / "best"
    best.mkdir(parents=True)
    (best / "model.safetensors").write_bytes(b"")
    (best / "config.json").write_text('{"architectures": ["DFlash2DraftModel"]}')
    (runs / "current").symlink_to("run-1/best")
    env = {**env, "SHARAR_RESULTS": str(tmp_path / "results")}
    assert resolve(env, "27b").draft_dir == best.resolve()
    assert resolve(env, "27b", overrides={"SHARAR_DRAFT": "pinned"}).draft_dir is None
    assert resolve(env, "27b", overrides={"SHARAR_PROPOSER": "mtp", "SHARAR_DRAFT_PRECISION": "bf16"}).draft_dir is None
    assert resolve(env, "flashnext").draft_dir is None
    assert resolve(env, "27b").follows_current and not resolve(env, "flashnext").follows_current
    assert not resolve(env, "27b", overrides={"SHARAR_DRAFT": "pinned"}).follows_current
    assert launch.promoted_draft(runs, "dflash2") == best.resolve() and launch.promoted_draft(runs, "mtp") is None


def test_tensor_parallel_limits(env):
    with pytest.raises(ValueError, match="not MTP"):
        resolve(env, "27b", overrides={"SHARAR_PROPOSER": "mtp", "SHARAR_DRAFT_PRECISION": "bf16"}, tp=2)
    with pytest.raises(ValueError, match="DRAFT_VOCAB"):
        resolve(env, "flashnext", overrides={"SHARAR_DRAFT_VOCAB": "65536"}, tp=2)


def test_the_launcher_pins_are_the_engines():
    assert (qwen38_config.MODEL_ID, qwen38_config.MODEL_REVISION) == (QWEN38.model_id, QWEN38.model_revision)
    assert (flashnext.MODEL_ID, flashnext.MODEL_REVISION) == (FLASHNEXT.model_id, FLASHNEXT.model_revision)
    for target in launch.TARGET_SETTINGS.values():  # settings, never host paths
        assert not {"SHARAR_HF_CACHE", "SHARAR_FLASHNEXT_CACHE", "SHARAR_RESULTS"} & set(target)


def test_config_file_parsing(tmp_path, monkeypatch):
    monkeypatch.setenv("SOMEWHERE", "/data")
    conf = tmp_path / "c.env"
    conf.write_text('export SHARAR_A="x y"\nSHARAR_B=$SOMEWHERE/b  # note\nnot a setting\nSHARAR_A=z\n')
    assert launch.read_config_file(conf) == {"SHARAR_A": "z", "SHARAR_B": "/data/b"}
