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
    assert s.max_context == 262144 and s.traces is None and s.ssd == s.results / "ssd" / "flashnext"  # the SSD tier is on
    assert resolve({**env, "SHARAR_SSD": "0"}, "flashnext").ssd is None
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
    ("fast", "flashnext", "1"), ("fast", "27b", "0"), ("exact", "flashnext", "0"), ("exact", "27b", "0")])
def test_device_acceptance_defaults_on_only_for_fast_flashnext(env, profile, target, device_accept):
    assert resolve(env, target, profile).get("SHARAR_DEVICE_ACCEPT") == device_accept


@pytest.mark.parametrize("explicit", ["0", "1"])
def test_an_explicit_setting_wins_over_the_profile(env, explicit):
    assert resolve({**env, "SHARAR_DEVICE_ACCEPT": explicit}, "flashnext", "fast").get("SHARAR_DEVICE_ACCEPT") == explicit
    assert resolve({**env, "SHARAR_DEVICE_ACCEPT": explicit}, "27b", "exact").get("SHARAR_DEVICE_ACCEPT") == explicit


def test_profile_knobs():
    pick = lambda p, t: {k: launch.profile_defaults(p, t)[k] for k in ("SHARAR_ACCEPT_DEFAULT", "SHARAR_ACCEPT_DEFAULT_SAMPLED", "SHARAR_W3_DECODE")}  # noqa: E731
    assert pick("exact", "qwen38") == {"SHARAR_ACCEPT_DEFAULT": "off", "SHARAR_ACCEPT_DEFAULT_SAMPLED": "0", "SHARAR_W3_DECODE": "0"}
    assert pick("fast", "qwen38") == pick("fast", "flashnext") == {"SHARAR_ACCEPT_DEFAULT": "3:2.0", "SHARAR_ACCEPT_DEFAULT_SAMPLED": "1", "SHARAR_W3_DECODE": "1"}
    with pytest.raises(ValueError, match="unknown serving profile"):
        launch.profile_defaults("turbo", "qwen38")
    with pytest.raises(ValueError, match="choose from exact, fast, throughput"):
        launch.profile_defaults("balanced", "qwen38")


@pytest.mark.parametrize("profile, target, threshold", [
    ("fast", "qwen38", "2:1.0"), ("throughput", "qwen38", "2:1.0"), ("fast", "flashnext", None), ("throughput", "flashnext", None),
    ("exact", "qwen38", None)])
def test_the_27b_samples_at_a_tighter_threshold_under_fast(profile, target, threshold):
    # at 3:2.0 the 27B's sampled thinking looped to max_tokens in Claude Code sessions, at 2:1.0 it did not
    assert launch.profile_defaults(profile, target).get("SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD") == threshold


@pytest.mark.parametrize("profile, target, w3", [
    ("exact", "qwen38", "0"), ("exact", "flashnext", "0"),
    ("fast", "qwen38", "1"), ("fast", "flashnext", "1"), ("throughput", "qwen38", "1"), ("throughput", "flashnext", "1")])
def test_the_profile_alone_turns_the_3bit_copies_on(profile, target, w3):
    # one switch for both targets
    assert launch.profile_defaults(profile, target)["SHARAR_W3_DECODE"] == w3


@pytest.mark.parametrize("profile, target, backend", [
    ("exact", "flashnext", None), ("fast", "flashnext", "auto"), ("throughput", "flashnext", "auto"),
    ("fast", "qwen38", None), ("throughput", "qwen38", None)])
def test_the_lossy_profiles_take_the_optimized_qsa_prefill(profile, target, backend):
    # exact keeps the `original` QSA prefill (the engine's own default); an explicit setting still wins
    assert launch.profile_defaults(profile, target).get("SHARAR_QSA_PREFILL_BACKEND") == backend


def test_an_explicit_qsa_prefill_backend_wins_over_the_profile(env):
    assert resolve({**env, "SHARAR_QSA_PREFILL_BACKEND": "original"}, "flashnext", "fast").get("SHARAR_QSA_PREFILL_BACKEND") == "original"
    assert resolve(env, "flashnext", "fast").get("SHARAR_QSA_PREFILL_BACKEND") == "auto"


@pytest.mark.parametrize("target, experts, batch, depths", [
    ("flashnext", "w3", "128", "16:3,32:3,64:3,128:3"), ("flashnext", "nvfp4", "64", "16:3"), ("qwen38", "w3", "16", None)])
def test_throughput_decodes_more_requests(target, experts, batch, depths):
    # Flash-Next verifies within the 64 rows its 3-bit copies serve; the 27B keeps its full depth (GSM8K x1.74);
    # with 3-bit experts on one Spark, every bucket from 16 requests verifies 3 drafts
    knobs = launch.profile_defaults("throughput", target, experts)
    assert knobs["SHARAR_MAX_BATCH"] == batch and knobs.get("SHARAR_SPEC_DEPTHS") == depths
    for profile in ("exact", "fast"):
        assert launch.profile_defaults(profile, target)["SHARAR_MAX_BATCH"] == "4"


def test_expert_selection_is_on_in_fast_and_throughput(env):
    """Flash-Next's budgeted selection (keep 4, 12 experts a request up to 64) in fast and throughput, on one Spark or
    two and with either expert format; exact and the 27B route in full."""
    on = {"SHARAR_MOE_KEEP": "4", "SHARAR_MOE_BUDGET": "12:64"}
    pick = lambda *args, **kw: {k: launch.profile_defaults(*args, **kw).get(k) for k in on}  # noqa: E731
    for profile in ("fast", "throughput"):
        assert pick(profile, "flashnext") == on
        assert pick(profile, "flashnext", "nvfp4") == on  # the NVFP4 experts repeat a kept expert at weight 0
        assert pick(profile, "flashnext", tp=2) == on
        assert pick(profile, "flashnext", "nvfp4", tp=2) == on
        assert pick(profile, "qwen38") == {k: None for k in on}  # the 27B has no routed experts
    assert pick("exact", "flashnext") == {k: None for k in on}
    assert resolve(env, "flashnext", "fast").get("SHARAR_MOE_KEEP") == "4"
    assert resolve({**env, "SHARAR_MOE_KEEP": "0"}, "flashnext", "fast").get("SHARAR_MOE_KEEP") == "0"
    assert resolve(env, "flashnext", "throughput", tp=2).get("SHARAR_MOE_BUDGET") == "12:64"


def test_adaptive_depth_prices_a_drafted_token_per_spark_count(env):
    # on two Sparks the one-Spark price (0.14) drafted chat and prose too deep; 0.2 measured best
    assert launch.profile_defaults("fast", "flashnext").get("SHARAR_SPEC_STEP_COST") == "0.14"
    assert launch.profile_defaults("fast", "flashnext", tp=2).get("SHARAR_SPEC_STEP_COST") == "0.2"
    assert resolve(env, "flashnext", "fast", tp=2).get("SHARAR_SPEC_STEP_COST") == "0.2"
    assert resolve({**env, "SHARAR_SPEC_STEP_COST": "0.3"}, "flashnext", "fast", tp=2).get("SHARAR_SPEC_STEP_COST") == "0.3"
    assert "SHARAR_SPEC_STEP_COST" not in launch.profile_defaults("throughput", "flashnext", tp=2)
    assert launch.profile_defaults("throughput", "flashnext", tp=2)["SHARAR_SPEC_DEPTHS"] == "16:3"  # measured one Spark


@pytest.mark.parametrize("profile, target, state", [
    ("exact", "qwen38", None), ("fast", "qwen38", None), ("throughput", "qwen38", "int8"),
    ("exact", "flashnext", None), ("fast", "flashnext", None), ("throughput", "flashnext", "int8")])
def test_the_8bit_state_under_throughput(profile, target, state):
    # +8-30% decode on the 27B at 4-16 requests, +13-25% on Flash-Next at 16-64; at one request nothing, and
    # the 27B's prefill is slower, so `fast` and `exact` keep the FP32 state
    assert launch.profile_defaults(profile, target).get("SHARAR_SSM_STATE") == state


def test_the_8bit_state_default_yields(env):
    assert resolve(env, "27b", "throughput").get("SHARAR_SSM_STATE") == "int8"
    assert resolve({**env, "SHARAR_SSM_STATE": "fp32"}, "27b", "throughput").get("SHARAR_SSM_STATE") == "fp32"
    assert resolve(env, "27b", "throughput", overrides={"SHARAR_SSM_STATE": "int4"}).get("SHARAR_SSM_STATE") == "int4"
    assert resolve({**env, "SHARAR_SSM_STATE": "int8"}, "27b", "fast").get("SHARAR_SSM_STATE") == "int8"
    # snapshot accepted states cannot commit a quantized state: the profile's int8 steps aside
    assert resolve({**env, "SHARAR_VERIFY_STATE": "snapshots"}, "27b", "throughput").get("SHARAR_SSM_STATE", "fp32") == "fp32"


@pytest.mark.parametrize("override, message", [
    ({"SHARAR_MAX_BATCH": "129"}, "MAX_BATCH"), ({"SHARAR_SPEC_DEPTHS": "8"}, "SPEC_DEPTHS"),
    ({"SHARAR_DRAFT_PRECISION": "fp4"}, "DRAFT_PRECISION"), ({"SHARAR_PROPOSER": "dspark"}, "SHARAR_PROPOSER"),
    ({"SHARAR_PROPOSER": "ngram"}, "SHARAR_PROPOSER"),
    ({"SHARAR_PROPOSER": "mtp", "SHARAR_DRAFT_PRECISION": "fp8"}, "FP8 draft precision"), ({"SHARAR_FLASHNEXT_EXPERTS": "fp4"}, "EXPERTS")])
def test_bad_settings_are_refused(env, override, message):
    with pytest.raises(ValueError, match=message):
        resolve(env, "27b", overrides=override)


def test_throughput_flashnext_batch_passes_validation(env):
    # the launcher once refused SHARAR_MAX_BATCH above 16, so `serve flashnext --throughput` (128) failed
    assert resolve(env, "flashnext", "throughput").get("SHARAR_MAX_BATCH") == "128"


def test_the_engine_gets_explicit_flags_and_its_environment(env, tmp_path):
    s = resolve({**env, "SHARAR_DRAFT": "pinned"}, "27b", "fast", traces=True)
    args = s.engine_args()
    assert args[args.index("--port") + 1] == "30001" and args[args.index("--spec") + 1] == "15"
    assert args[args.index("--context-length") + 1] == "262144" and args[args.index("--draft-precision") + 1] == "fp8"
    assert args[args.index("--accept-default") + 1] == "3:2.0" and args[args.index("--accept-default-sampled-threshold") + 1] == "2:1.0"
    assert "--draft" not in args and args[args.index("--traces") + 1] == str(s.traces)
    assert "--accept-soft" not in args  # unset: the forced accept
    soft = resolve({**env, "SHARAR_DRAFT": "pinned", "SHARAR_ACCEPT_SOFT": "0.25"}, "27b", "fast").engine_args()
    assert soft[soft.index("--accept-soft") + 1] == "0.25"
    e = s.engine_env(tmp_path / "w3")
    assert e["SHARAR_W3_CHECKPOINT"] == str(tmp_path / "w3") and e["SHARAR_SSD_DIR"] == str(s.ssd)
    assert "SHARAR_DRAFT" not in e and "SHARAR_TRACES" not in e  # the flags carry them
    assert resolve(env, "27b").traces is None and "--traces" not in resolve(env, "27b").engine_args()
    assert "--tp" in resolve(env, "27b", tp=2).engine_args(tp=2)


def test_the_ssd_tier_settings_are_checked(env, tmp_path):
    """The SSD tier stores under the results directory, one store per target, on every Spark, or under SHARAR_SSD_DIR
    when set; malformed SHARAR_SSD, SHARAR_SSD_BLOCK_TOKENS and SHARAR_SSD_GIB values are rejected."""
    s = resolve(env, "27b", overrides={"SHARAR_SSD": "1", "SHARAR_SSD_GIB": "50", "SHARAR_SSD_BLOCK_TOKENS": "1024"})
    e = s.engine_env(None)
    assert s.ssd == s.results / "ssd" / "qwen38" and e["SHARAR_SSD_DIR"] == str(s.ssd)
    assert resolve(env, "flashnext", overrides={"SHARAR_SSD": "1"}).ssd == s.results / "ssd" / "flashnext"
    assert e["SHARAR_SSD"] == "1" and e["SHARAR_SSD_GIB"] == "50" and e["SHARAR_SSD_BLOCK_TOKENS"] == "1024"
    assert resolve(env, "27b", overrides={"SHARAR_SSD": "1", "SHARAR_SSD_DIR": str(tmp_path / "nvme")}).ssd == tmp_path / "nvme"
    for override, message in (({"SHARAR_SSD": "2"}, "SHARAR_SSD must be"), ({"SHARAR_SSD_BLOCK_TOKENS": "300"}, "divide 2048"),
                              ({"SHARAR_SSD_GIB": "lots"}, "number of GiB")):
        with pytest.raises(ValueError, match=message):
            resolve(env, "27b", overrides=override)
    two = resolve(env, "27b", overrides={"SHARAR_SSD": "1"}, tp=2)  # each rank stores on its own Spark, same path
    assert two.engine_env(None)["SHARAR_SSD_DIR"] == str(s.ssd) and "--tp" in two.engine_args(tp=2)


def test_flashnext_without_a_checkpoint_serves_its_fp8_trunk(env):
    e = resolve(env, "flashnext", "fast").engine_env(None)
    assert e["SHARAR_W3_DECODE"] == "0" and e["SHARAR_FLASHNEXT_CACHE"] and e["SHARAR_DRAFT_HEAD"] == "w2"


@pytest.mark.parametrize("policy", ["auto", "cute", "triton"])
def test_gdn_prefill_backend_reaches_engine_through_the_launcher(env, policy):
    from sharar.engine.config import EngineConfig
    resolved = resolve({**env, "SHARAR_GDN_PREFILL": policy}, "27b")
    engine_env = resolved.engine_env(None)
    assert engine_env["SHARAR_GDN_PREFILL"] == policy
    assert EngineConfig.from_env(engine_env).gdn_prefill == policy


@pytest.mark.parametrize("policy", ["original", "auto", "optimized"])
def test_qsa_prefill_backend_reaches_engine(env, policy):
    from sharar.engine.config import EngineConfig
    resolved = resolve({**env, "SHARAR_QSA_PREFILL_BACKEND": policy}, "flashnext")
    engine_env = resolved.engine_env(None)
    assert engine_env["SHARAR_QSA_PREFILL_BACKEND"] == policy
    assert EngineConfig.from_env(engine_env).qsa_prefill_backend == policy


def test_traces_are_opt_in(env, tmp_path):
    env = {**env, "SHARAR_RESULTS": str(tmp_path / "results")}
    assert resolve(env, "27b").traces is None
    assert resolve(env, "27b", traces=True).traces == tmp_path / "results" / "traces"
    assert resolve({**env, "SHARAR_TRACES": str(tmp_path / "t")}, "27b").traces == tmp_path / "t"
    assert resolve({**env, "SHARAR_TRACES": "off"}, "27b", traces=True).traces is None


def test_a_draft_directory_is_served_only_by_dflash2(env, tmp_path):
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "model.safetensors").write_bytes(b"")
    (draft / "config.json").write_text('{"architectures": ["DFlash2DraftModel"]}')
    assert resolve(env, "27b").draft_dir is None  # the pinned publisher draft
    assert resolve(env, "27b", overrides={"SHARAR_DRAFT": str(draft)}).draft_dir == draft.resolve()
    with pytest.raises(ValueError, match="DFlash2"):
        resolve(env, "flashnext", overrides={"SHARAR_DRAFT": str(draft)})


def test_tensor_parallel_limits(env):
    with pytest.raises(ValueError, match="not MTP"):
        resolve(env, "27b", overrides={"SHARAR_PROPOSER": "mtp", "SHARAR_DRAFT_PRECISION": "bf16"}, tp=2)
    with pytest.raises(ValueError, match="DFlash2 on one Spark"):
        resolve(env, "flashnext", overrides={"SHARAR_PROPOSER": "dflash2"}, tp=2)
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


def test_the_27b_fp8_draft_applies_when_its_draft_runs(env):
    assert resolve(env, "27b").get("SHARAR_DRAFT_PRECISION") == "fp8"
    for override in ({"SHARAR_SPEC": "0"}, {"SHARAR_PROPOSER": "mtp"}):
        assert resolve(env, "27b", overrides=override).get("SHARAR_DRAFT_PRECISION") == "bf16"


@pytest.mark.parametrize("profile", launch.PROFILE_NAMES)
def test_flashnext_serves_dflash2_on_request(env, profile):
    # FP8 as the 27B's (with the 3-bit copies where the profile decodes 3-bit), at depths that keep MTP's guaranteed
    # experts a verified token in `fast` and MTP's own in `throughput`; MTP stays the default with its own depths
    s = resolve(env, "flashnext", profile, overrides={"SHARAR_PROPOSER": "dflash2"})
    assert s.parallel_draft and s.draft_dir is None
    assert (s.get("SHARAR_DRAFT_PRECISION"), s.get("SHARAR_SPEC_STEP_COST")) == ("fp8", "")
    assert s.get("SHARAR_ACCEPT_DEFAULT") == {"exact": "off", "fast": "2:1.0", "throughput": "2:1.0"}[profile]  # MTP keeps 3:2.0
    depths = {"exact": ("7", ""), "fast": ("9", "2:7,4:7"), "throughput": ("4", "16:3,32:3,64:3,128:3")}
    assert (s.get("SHARAR_SPEC"), s.get("SHARAR_SPEC_DEPTHS")) == depths[profile]
    capped = resolve(env, "flashnext", profile, overrides={"SHARAR_PROPOSER": "dflash2", "SHARAR_SPEC": "5"})
    assert capped.get("SHARAR_SPEC_DEPTHS") == {"exact": "", "fast": "2:5,4:5", "throughput": "16:3,32:3,64:3,128:3"}[profile]
    if profile == "throughput":  # MTP's depths, also with NVFP4 experts (up to 64 requests)
        assert resolve(env, "flashnext", profile, overrides={"SHARAR_PROPOSER": "dflash2", "SHARAR_FLASHNEXT_EXPERTS": "nvfp4"}).get(
            "SHARAR_SPEC_DEPTHS") == "16:3"
    mtp = resolve(env, "flashnext", profile)
    assert (mtp.proposer, mtp.get("SHARAR_DRAFT_PRECISION")) == ("mtp", "bf16") and not mtp.parallel_draft
    assert mtp.get("SHARAR_ACCEPT_DEFAULT") == ("off" if profile == "exact" else "3:2.0")
    assert resolve({**env, "SHARAR_DRAFT_PRECISION": ""}, "flashnext", profile).get("SHARAR_DRAFT_PRECISION") == "bf16"  # empty is unset
    assert mtp.get("SHARAR_SPEC") == ("8" if profile == "fast" else "4")


def test_moe_budget_setting():
    from sharar.engine.config import EngineConfig
    assert (EngineConfig.from_env({"SHARAR_MOE_BUDGET": "64"}).moe_budget, EngineConfig.from_env({"SHARAR_MOE_BUDGET": "64"}).moe_budget_per_request) == (64, 0)
    cfg = EngineConfig.from_env({"SHARAR_MOE_BUDGET": "12:64"})
    assert (cfg.moe_budget, cfg.moe_budget_per_request) == (64, 12)
    with pytest.raises(ValueError, match="SHARAR_MOE_BUDGET"):
        EngineConfig.from_env({"SHARAR_MOE_BUDGET": "a:b"})
    assert EngineConfig.from_env({}).moe_budget == 0


def test_profile_depths_stay_within_a_lower_batch_limit(env):
    # a limit below 128 starts (128 is no captured bucket then), and a limit that is not a bucket verifies what the
    # bucket it would pad to does
    cases = {("throughput", "64"): {"mtp": "16:3,32:3,64:3", "dflash2": "16:3,32:3,64:3"},
             ("throughput", "100"): {"mtp": "16:3,32:3,64:3,100:3", "dflash2": "16:3,32:3,64:3,100:3"},
             ("throughput", "12"): {"mtp": "12:3", "dflash2": "12:3"},
             ("fast", "3"): {"mtp": "2:4,3:4", "dflash2": "2:7,3:7"}}
    for (profile, batch), want in cases.items():
        for proposer, depths in want.items():
            s = resolve(env, "flashnext", profile, overrides={"SHARAR_PROPOSER": proposer, "SHARAR_MAX_BATCH": batch})
            assert s.get("SHARAR_SPEC_DEPTHS") == depths, (profile, batch, proposer)



def test_preflight_fetches_the_targets_own_drafter(env):
    from sharar.launch.preflight import pinned_snapshots
    from sharar.model import qwen38_config
    from sharar.model.flashnext import config as flashnext_config
    for target, config in (("27b", qwen38_config), ("flashnext", flashnext_config)):
        s = resolve(env, target, "fast", overrides={"SHARAR_PROPOSER": "dflash2"})
        assert pinned_snapshots(s)[-1][1:] == config.DRAFT_PINS["dflash2"], target
    assert len(pinned_snapshots(resolve(env, "flashnext", "fast"))) == 1  # MTP: the checkpoint's own head


def test_both_targets_keep_nvfp4_kv_in_fast_and_throughput_on_one_spark(env):
    for target in ("27b", "flashnext"):
        for profile, want in (("exact", ""), ("fast", "nvfp4"), ("throughput", "nvfp4")):
            assert resolve(env, target, profile).get("SHARAR_KV_FORMAT") == want, (target, profile)
        assert resolve(env, target, "fast", tp=2).get("SHARAR_KV_FORMAT") == "", target
    assert resolve(env, "27b", "fast", overrides={"SHARAR_KV_FORMAT": "fp8"}).get("SHARAR_KV_FORMAT") == "fp8"
    for target, overrides, kw, match in (("flashnext", {"SHARAR_KV_FORMAT": "nvfp4"}, {"tp": 2}, "two Sparks"),
                                         ("27b", {"SHARAR_KV_FORMAT": "nvfp4"}, {"tp": 2}, "two Sparks"),
                                         ("27b", {"SHARAR_KV_FORMAT": "fp4"}, {}, "must be fp8 or nvfp4")):
        with pytest.raises(ValueError, match=match):
            resolve(env, target, "fast", overrides=overrides, **kw)


def test_host_floor_setting(env, tmp_path):
    from sharar.launch import guard
    default = resolve(env, "flashnext").values
    assert guard.floor_kib(default) == 6 * 2**20  # the GB10 default
    assert guard.guard_kib(default) == 3 * 2**20  # the guard acts at half the floor
    conf = tmp_path / "floor.env"
    conf.write_text("SHARAR_HOST_FLOOR_GIB=12\n")
    values = resolve({**env, "SHARAR_CONFIG_FILE": str(conf)}, "flashnext").values
    assert guard.floor_kib(values) == 12 * 2**20 and guard.guard_kib(values) == 6 * 2**20  # a config file's
    for bad in ("4G", "-8"):
        with pytest.raises(ValueError, match="SHARAR_HOST_FLOOR_GIB"):
            resolve(env, "flashnext", overrides={"SHARAR_HOST_FLOOR_GIB": bad})
    with pytest.raises(ValueError, match="SHARAR_MIN_AVAILABLE_KIB is retired"):  # main's guard setting, e.g. its 16 GiB
        resolve(env, "flashnext", overrides={"SHARAR_MIN_AVAILABLE_KIB": str(16 * 2**20)})
