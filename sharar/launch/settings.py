"""What `sharar serve` serves: one target's settings, resolved from every source in one place.

Order, first wins: command-line choices (`resolve`'s `target`, `profile` and `overrides`), the process environment
(`SHARAR_*`), the user's config file (`SHARAR_CONFIG_FILE`, else `config/gb10.env` in the checkout, else
`~/.config/sharar/sharar.env`; `KEY=VALUE` lines), the target's settings (`TARGET_SETTINGS`), the serving profile's
defaults (`profile_defaults`), then the built-in defaults. `resolve` returns the engine's environment and its `sharar engine` arguments; nothing here
imports torch, so the launcher stays fast and testable on a CPU.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path


CHECKOUT = Path(__file__).resolve().parents[2]  # the repository root when sharar is installed editable

# Canonical target names and the names users type.
ALIASES = {"27b": "qwen38", "27B": "qwen38", "qwen38": "qwen38", "qwen3.8-27b": "qwen38",
           "flashnext": "flashnext", "flash-next": "flashnext", "Flash-Next": "flashnext",
           "nemotron": "nemotron", "nemotron-super": "nemotron", "Nemotron": "nemotron",
           "bolt": "bolt", "nemotron-bolt": "bolt"}
TARGET_CHOICES = ("27b", "flashnext", "nemotron", "bolt", "qwen38", "flash-next", "nemotron-super", "nemotron-bolt")
SERVED = ("27b", "flashnext", "nemotron", "bolt")  # one name per target, in the order `status`, `stop` and the clients look
SERVE_HINT = "sharar serve " + ", ".join(SERVED[:-1]) + " or " + SERVED[-1]  # what to run when nothing is serving

# What each target serves when nothing else says so (port, service, drafter: the SHARAR_* keys), and what the launcher
# knows of it: `recurrence` (`Target.recurrence`, the formats of SSM_STATES; default gdn), `w3_missing` (what the
# 3-bit profiles serve without a calibrated checkpoint), `w3_calibrated_only` (no nearest-rounded copies then),
# `w3_required` (no fallback: the 3-bit profiles refuse to start without one) and `fp8_kv_only` (its attention writes
# the FP8 pool only).
TARGET_SETTINGS = {
    "qwen38": {"SHARAR_SERVICE": "sharar-qwen38", "SHARAR_PORT": "30001", "SHARAR_PROPOSER": "dflash2", "SHARAR_SPEC": "15",
               "SHARAR_VERIFY_STATE": "reconstruct", "SHARAR_DRAFT_PRECISION": "fp8",
               "w3_missing": "nearest-rounded copies (about 2 GSM8K points and 8% tokens per round below calibrated)"},
    "flashnext": {"SHARAR_SERVICE": "sharar-flashnext", "SHARAR_PORT": "30004", "SHARAR_PROPOSER": "mtp",
                  "SHARAR_VERIFY_STATE": "reconstruct", "SHARAR_DRAFT_PRECISION": "fp8", "SHARAR_DRAFT": "pinned",
                  "SHARAR_DRAFT_HEAD": "w2", "SHARAR_FLASHNEXT_TRUNK": "fp8", "w3_calibrated_only": True,
                  "w3_missing": "serving the FP8 trunk, about 24% slower single-stream than {profile} with it"},
    # its own MTP head (the depth is the profile's), one Spark only; the draft steps read a 2-bit copy of the FP8
    # lm_head, its top 16 rescored by the FP8 rows (lossless: acceptance checks every draft against the target)
    # Both Nemotron checkpoints serve their vision encoder (image input; SHARAR_VISION=0 leaves its 1.7 GiB to the pools)
    "nemotron": {"SHARAR_SERVICE": "sharar-nemotron", "SHARAR_PORT": "30005", "SHARAR_PROPOSER": "mtp",
                 "SHARAR_VERIFY_STATE": "reconstruct", "SHARAR_DRAFT_HEAD": "w2", "SHARAR_VISION": "1", "recurrence": "mamba2",
                 "w3_calibrated_only": True, "w3_required": True, "fp8_kv_only": True},
    "bolt": {"SHARAR_SERVICE": "sharar-bolt", "SHARAR_PORT": "30006", "SHARAR_PROPOSER": "dflash2",
             "SHARAR_DRAFT": "pinned", "SHARAR_DRAFT_PRECISION": "bf16", "SHARAR_VERIFY_STATE": "reconstruct",
             "SHARAR_VISION": "1", "recurrence": "mamba2", "w3_calibrated_only": True,
             "w3_missing": "serving the checkpoint encodings, about 25% slower single-stream than {profile} with them"},
}
# The recurrent-state formats each recurrence stores (SHARAR_SSM_STATE, cache/ssm_format.py): the per-window integer
# formats on both, Mamba-2's FP16 with stochastic rounding.
SSM_STATES = {"gdn": ("fp32", "int8", "int4"), "mamba2": ("fp32", "fp16_sr", "int8", "int4")}

# The serving profiles: the default of every speed/fidelity knob a setting does not fix.
#   exact     the target's own distribution: exact acceptance, no lossy weight copies (benchmarks)
#   fast      (default) everything measured to be worth it: relaxed top-3 within 2.0 logits for greedy requests,
#             relaxed s = 0.5 for sampled ones (no token more than e^0.5 times as often as the target's law gives it),
#             the calibrated 3-bit copies, device-resident acceptance on Flash-Next,
#             and Flash-Next's draft depth per request (up to 8 for one request, 4 for two or more; the others
#             verify 4 drafts), and Flash-Next's budgeted expert selection (a decode or verify forward reads at most
#             12 experts a request, 16 for a request alone, 64 in all, chosen by router probability across its tokens;
#             each token keeps its top 4, fewer in wide forwards; one Spark: 1 / 4 requests +2 / +8% over keep 5 + reuse
#             5), and Flash-Next's
#             optimized QSA prefill kernels where eligible (SHARAR_QSA_PREFILL_BACKEND=auto: RULER v2 32K equal)
#   throughput  fast with more requests decoded together: the 27B 16 (full depth; batches of nine or more verify on the
#             original weights), Flash-Next 128 with its 3-bit experts (64 with NVFP4
#             experts), three drafts at sixteen requests so their verification stays within the 64 rows Flash-Next's
#             3-bit copies serve (four above: wider batches exceed 64 rows at any depth, and three measured no faster there)
PROFILE_NAMES = ("exact", "fast", "throughput")


def profile_defaults(profile: str, target: str, flashnext_experts: str = "w3", tp: int = 1,
                     proposer: str | None = None) -> dict[str, str]:
    """The profile's value for each knob it sets, as SHARAR_* settings, for one Spark or `tp` of them and the drafter
    `proposer` (None: the target's own)."""
    if target == "bolt":
        if profile not in PROFILE_NAMES:
            raise ValueError(f"unknown serving profile {profile!r}")
        # exact: the checkpoint encodings, all nine routed experts, FP32 state and exact verification of LiLiCorr's
        # seven drafts. Routed experts on b12x's fused MoE (BF16 activations, as vLLM serves them).
        p = {"SHARAR_SPEC": "7", "SHARAR_W3_DECODE": "0", "SHARAR_ACCEPT_DEFAULT": "off",
             "SHARAR_ACCEPT_SOFT": "0", "SHARAR_DEVICE_ACCEPT": "0", "SHARAR_MAX_BATCH": "4",
             "SHARAR_SSM_STATE": "fp32", "SHARAR_KV_FORMAT": "fp8", "SHARAR_SSD": "1",
             "SHARAR_MOE_KEEP": "0", "SHARAR_MOE_BUDGET": "0", "SHARAR_BOLT_EXPERTS": "b12x"}
        if profile == "exact":
            return p
        # fast, throughput: the calibrated 3-bit copies of the trunk and vocabulary head (weights/bolt_w3_calibrate.py)
        # and the int8 Mamba state; paired on GSM8K / GPQA against exact +0.1 / +1.5 points. Over the 22 h2h prompts on
        # one GB10, mean / four-stream tok/s 86.7 / 169.8 (exact) -> 111.0 / 187.0 (+W3) -> 114.2 / 213.7 (+int8).
        # Expert sharing as Super's fast (model/nemotron/moe.py route_budget): each token keeps its top 6 experts and a
        # decode or verify call reads at most 20 a request and 64 in all (114.2 / 213.7 -> 121.7 / 228.9). Sampled
        # with thinking on, fast against exact on GPQA: 75.8% against 76.3% (p 1.0). Keep 4 at 16:48 (130.7 / 251.6)
        # thought 1.23x longer and cost 6.6 points (p 0.011), mostly answers cut off at the 32K cap.
        # Verification stays exact: relaxed top-2 within 1.0 cost fast GPQA 5.6 points sampled with thinking on
        # (p 0.052), and greedy top-3 within 2.0 cost 8.6 (p 0.02).
        p.update(SHARAR_W3_DECODE="1", SHARAR_SSM_STATE="int8",
                 SHARAR_MOE_KEEP="6", SHARAR_MOE_BUDGET="20:64")
        return p
    if target == "nemotron":
        return nemotron_defaults(profile)
    flashnext = target == "flashnext"
    if profile == "exact":
        p = {"SHARAR_ACCEPT_DEFAULT": "off", "SHARAR_ACCEPT_SOFT": "0", "SHARAR_W3_DECODE": "0", "SHARAR_DEVICE_ACCEPT": "0"}
    elif profile in ("fast", "throughput"):
        # sampled requests: Flash-Next s = 0.5, the 27B exact. Sampled with thinking on, 4 requests, one Spark: s = 0.5
        # decodes Flash-Next (sampled MTP drafts) at 216 tok/s against 183 exact, the 27B (DFlash2) at 195 against 165.
        # On GPQA-D thinking at a 64K cap (Flash-Next, 99 items, paired) s = 0.5 scored 81.8% against 84.8% exact
        # (p 0.51) with 18% longer answers, so the faster decode buys little on long reasoning; the 27B stays exact.
        # Accepting outright every draft within the greedy K:M rule (257 and 196 tok/s) cost long reasoning its
        # accuracy: those tokens came out at the drafter's probability.
        p = {"SHARAR_ACCEPT_DEFAULT": "3:2.0", "SHARAR_ACCEPT_SOFT": "0.5" if flashnext else "0", "SHARAR_W3_DECODE": "1",
             "SHARAR_DEVICE_ACCEPT": "1" if flashnext else "0"}
    else:
        raise ValueError(f"unknown serving profile {profile!r} (choose from {', '.join(PROFILE_NAMES)})")
    if profile != "exact" and (proposer or ("mtp" if flashnext else "dflash2")) == "mtp":
        # sampled MTP drafts (proposers/mtp.draft_laws): a sampled request draws each draft from its own law over the
        # draft head's 16 rescored candidates, and the sampled rule tests the draft against that law, which raises
        # acceptance without changing what the rule emits (operations.md#sampled-mtp-drafts)
        p["SHARAR_MTP_SAMPLE"] = "1"
    p["SHARAR_MAX_BATCH"] = "4"
    if flashnext and profile != "exact":
        # budgeted expert selection in decode and verify forwards: at most 12 experts a request (16 for a request alone),
        # 64 in all, chosen by router probability across the forward's tokens; each token keeps its top 4, fewer in wide
        # forwards (2 from 64 tokens). One Spark, 3-bit experts, against keep 5 + reuse 5 in fast and keep 2 + budget 64
        # in throughput: fast 1 / 4 requests +2 / +8% tok/s, GSM8K-300 one request at a time -1.3 pp (not significant),
        # quick suite at 4 requests +0.7 pp; throughput 16 / 64 streams +4 / +1%, RULER v2 32K +2.5. Two Sparks, before
        # the faster route kernels and the 16-expert floor: fast -6.5% at one request (NVFP4 experts -5%), throughput
        # -2 to +9%; NVFP4 experts on one Spark unmeasured.
        p.update(SHARAR_MOE_KEEP="4", SHARAR_MOE_BUDGET="12:64")
        # the optimized QSA prefill kernels where eligible (SM121, one or two Sparks, through 32K): RULER v2 32K 94.1
        # against 94.0 (120 items, 2 better / 2 worse), prefill 3-4% faster at 8K and 32K
        p["SHARAR_QSA_PREFILL_BACKEND"] = "auto"
    if flashnext:
        p["SHARAR_SPEC"] = "4"
        if proposer == "dflash2":
            # One forward drafts the block. Under budgeted expert selection a verify round reads its cap's experts at
            # any depth (one request: 32-35 ms from depth 3 to 15, one Spark, 3-bit experts), so a deeper round guarantees
            # each verified token fewer of its own: the depths keep MTP's floor of 3 a token (one request 9, two and four
            # 7), where one request at depth 10 or more, or two to four at 8 or more, would keep 2. `exact` reads every
            # expert; its depth is unmeasured.
            if profile == "exact":
                p["SHARAR_SPEC"] = "7"
            else:
                # top-2 within 1.0: at Flash-Next's top-3 within 2.0, DFlash2 lost 3.3 points of GSM8K-300 one request
                # at a time against MTP with its 3-bit draft head (-6.2 to -0.4); at 2:1.0 it ties (-0.3, -2.9 to +2.2)
                # for 5% fewer tokens a round there
                p["SHARAR_ACCEPT_DEFAULT"] = "2:1.0"
                if profile == "fast":
                    p.update(SHARAR_SPEC="9", SHARAR_SPEC_DEPTHS="2:7,4:7")
        elif profile == "fast":  # adaptive depth: h2h greedy median 92.2 -> 101.3 tok/s, 4 streams within noise
            # two Sparks: 0.14 drafted chat and prose too deep (-12%, -7% against depth 4); 0.2 keeps them within noise.
            # One Spark, two to four requests: caps 7 and 6, which a batch reaches when one request accepts well (fixed
            # depth, 512 greedy tokens: two requests 184.0 tok/s at 7 against 171.1 at 4, four 238.7 at 6 against 230.5)
            p.update(SHARAR_SPEC="8", SHARAR_SPEC_STEP_COST="0.14" if tp == 1 else "0.2",
                     SHARAR_SPEC_DEPTHS="2:7,4:6" if tp == 1 else "2:4,4:4")
    if profile == "throughput" and target == "qwen38" and (proposer or "dflash2") == "dflash2":
        # adaptive verification (engine/depth.RowBudget): DFlash2's calibrated draft scores and the measured verify
        # costs decide what each round verifies, a depth per request in packed graphs (one Spark against fixed 15,
        # greedy / T 0.7: 8 requests x1.14 / x1.09, 16 x1.29 / x1.21; two Sparks +9 / +18% greedy; quick suite equal).
        # `fast` keeps the fixed depth: at 1 and 4 requests it measured x0.96-1.05, for a host wait every round and
        # eleven packed graphs a bucket; `exact` keeps it so its runs repeat. Flash-Next keeps its depth tables: under
        # its expert budget verify rows cost little, so verifying fewer saved nothing
        p["SHARAR_SPEC_POLICY"] = "budget"
    if profile != "exact" and tp == 1:
        # NVFP4 KV (cache/kv_pool.py): 4.5 bits a value where FP8 takes 8, so the same memory holds more tokens (the
        # 27B 16/9, Flash-Next ~1.4 beside its BF16 index keys): long contexts and concurrent requests, more cached
        # prefixes, fewer bytes read back from the SSD. The 27B: teacher-forced at 32K-96K its divergence from BF16
        # KV is 1.6x FP8's; rounds within 2% of FP8 at 10K and 6% faster at 64K with 4 requests (verify_attn.cu).
        p["SHARAR_KV_FORMAT"] = "nvfp4"
    if profile == "throughput":
        p["SHARAR_MAX_BATCH"] = ("128" if flashnext_experts == "w3" else "64") if flashnext else "16"
        if flashnext:
            # at 32-128 requests a verified row costs ~0.5 ms (one Spark, 3-bit experts) and depth 3 beat 4 in steady
            # state (32: 523 vs 516 tok/s, 64: 795 vs 783, 128: 1003 vs 974); NVFP4 experts and two Sparks are unmeasured
            # DFlash2 drafts as MTP here: from 16 requests depth 5 decoded 9-10% slower than 3 (one Spark, 3-bit
            # experts, 8K and 32K prompts), its verification then past the 64 rows the 3-bit copies serve; and the
            # verify buffers hold every request slot at the deepest depth (128 requests: 1.7 GiB at 4, 3.4 at 9)
            p["SHARAR_SPEC_DEPTHS"] = "16:3,32:3,64:3,128:3" if flashnext_experts == "w3" and tp == 1 else "16:3"
        # the 8-bit GDN state pays with many requests; `fast` keeps FP32: no gain at one request, slower prefill
        p["SHARAR_SSM_STATE"] = "int8"
    # the SSD tier in every profile: a restore gives back the state the GPU held, and an evicted or restarted prefix
    # resumes in a fraction of a second instead of a full prefill (SHARAR_SSD=0 turns it off)
    p["SHARAR_SSD"] = "1"
    return p


def nemotron_defaults(profile: str) -> dict[str, str]:
    """Nemotron-3.5-Super's profiles (one Spark, its MTP head; docs/nemotron.md#serving), Flash-Next's knobs where they
    apply."""
    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown serving profile {profile!r} (choose from {', '.join(PROFILE_NAMES)})")
    # exact: depth 3, the best exact depth (one request 38.0 tok/s; depth 4 37.7)
    p = {"SHARAR_ACCEPT_DEFAULT": "off", "SHARAR_ACCEPT_SOFT": "0", "SHARAR_W3_DECODE": "0",
         "SHARAR_DEVICE_ACCEPT": "0", "SHARAR_MAX_BATCH": "4", "SHARAR_SPEC": "3", "SHARAR_SSD": "1"}
    if profile == "exact":
        return p
    # relaxed top-3 within 2.0 logits for greedy requests, sampled ones relaxed by s = 0.5 (Flash-Next's setting) from
    # sampled MTP drafts, device-resident acceptance and both calibrated 3-bit copies (h2h at one request, depth 5: 54.6 -> 81.0
    # tok/s); depth up to 6 for two requests and 5 for three or four (two requests, 512 greedy tokens: 128.6 tok/s at 6
    # against 116.3 at 4; four: 184.4 at 5 against 168.6)
    p.update(SHARAR_ACCEPT_DEFAULT="3:2.0", SHARAR_ACCEPT_SOFT="0.5", SHARAR_MTP_SAMPLE="1", SHARAR_DEVICE_ACCEPT="1",
             SHARAR_SPEC_DEPTHS="2:6,4:5", SHARAR_W3_DECODE="1")
    # adaptive depth for one request, up to 8 (engine/depth.AcceptanceDepth, docs/nemotron.md#adaptive-depth): a drafted
    # token costs about 2.2 ms against a depth-1 round's 39-41 ms, so 0.065; h2h against depth 5 greedy +3%, t=0.7 +6%
    p.update(SHARAR_SPEC="8", SHARAR_SPEC_STEP_COST="0.065")
    # the FP16 Mamba state with stochastic rounding (half the state traffic of decode, verify and replay; its rounding
    # is keyed on token position and layer, so a request's arithmetic does not depend on the batch)
    p["SHARAR_SSM_STATE"] = "fp16_sr"
    # expert sharing (model/nemotron/moe.py: each verify row keeps its top 12 experts, a call reads at most 48 a request
    # and 128 in all), gated together with the FP16 state against exact (GSM8K-300 97.7% each at one to four requests):
    # 102.5 tok/s at one request; at 8 / 16 streams 277 / 353 tok/s against 190 / 239 without
    p.update(SHARAR_MOE_KEEP="12", SHARAR_MOE_BUDGET="48:128")
    if profile == "throughput":  # up to 16 requests, depth 5 to eight (a bucket not named verifies SHARAR_SPEC), 3 past
        p.update(SHARAR_MAX_BATCH="16", SHARAR_SPEC_DEPTHS="2:6,4:5,8:5,16:3")
    return p


DEFAULT_CONTEXT = 262144


def target_name(name: str | None) -> str:
    try:
        return ALIASES[name or "qwen38"]
    except KeyError:
        raise ValueError(f"unknown target {name!r} (choose 27b, flashnext, nemotron or bolt)") from None


def model_config(name: str):
    """The target's pinned-checkpoint module (MODEL_ID, MODEL_REVISION, DRAFT_PINS when it has drafters); torch-free."""
    if name == "bolt":
        from sharar.model.bolt import config as c
    elif name == "flashnext":
        from sharar.model.flashnext import config as c
    elif name == "nemotron":
        from sharar.model.nemotron import config as c
    else:
        from sharar.model import qwen38_config as c
    return c


def config_file(env: dict | None = None) -> Path | None:
    """The user's settings file, if there is one."""
    env = os.environ if env is None else env
    if env.get("SHARAR_CONFIG_FILE"):
        return Path(expand(env["SHARAR_CONFIG_FILE"], env))
    for candidate in (CHECKOUT / "config" / "gb10.env", Path(env.get("HOME") or Path.home()) / ".config" / "sharar" / "sharar.env"):
        if candidate.is_file():
            return candidate
    return None


def expand(value: str, env: dict | None = None) -> str:
    """$VAR and a leading ~ expanded from `env` (default: this process's environment), as a shell would."""
    env = os.environ if env is None else env
    value = re.sub(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", lambda m: env.get(m.group(1), m.group(0)), value)
    if value == "~" or value.startswith("~/"):
        value = (env.get("HOME") or str(Path.home())) + value[1:]
    return value


def read_config_file(path: Path | None, env: dict | None = None) -> dict[str, str]:
    """KEY=VALUE lines (as a shell would source them: later lines win, $VAR and ~ expanded, quotes stripped)."""
    values: dict[str, str] = {}
    if path is None or not path.is_file():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.split(" #")[0].strip().strip('"').strip("'")
        if re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            values[key] = expand(value, env)
    return values


@dataclass
class Settings:
    """One target's resolved settings: what the launcher, the engine and the tools agree on."""
    target: str
    profile: str
    values: dict[str, str]          # every SHARAR_* setting after resolution (strings, as the environment holds them)
    port: int
    host: str
    service: str
    results: Path
    hf_cache: Path
    model_cache: Path               # where this target's checkpoint lives (Flash-Next may have its own cache)
    max_context: int
    traces: Path | None
    draft_dir: Path | None          # a DFlash2 checkpoint to serve; None: the pinned publisher draft
    ssd: Path | None                # the SSD tier's directory when SHARAR_SSD=1
    warnings: list[str] = field(default_factory=list)

    def get(self, key: str, default: str = "") -> str:
        return self.values.get(key) or default

    @property
    def proposer(self) -> str:
        return self.get("SHARAR_PROPOSER", "dflash2")

    @property
    def spec(self) -> int:
        return int(self.get("SHARAR_SPEC", "15"))

    @property
    def parallel_draft(self) -> bool:
        return self.proposer == "dflash2" and self.spec > 0

    @property
    def w3_root(self) -> Path:
        return self.results / "w3-checkpoints"

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def log_file(self, name: str | None = None) -> Path:
        return self.results / "logs" / f"{name or self.service}.log"

    def engine_env(self, w3_checkpoint: Path | None) -> dict[str, str]:
        """The engine's SHARAR_* environment: every resolved setting, with the launcher's own choices applied."""
        env = {k: v for k, v in self.values.items() if k.startswith("SHARAR_") and v != ""}
        for k in ("SHARAR_DRAFT", "SHARAR_TRACES", "SHARAR_W3_CHECKPOINT", "SHARAR_SSD_DIR", "SHARAR_CONFIG_FILE"):
            env.pop(k, None)
        env.update(SHARAR_TARGET=self.target, SHARAR_PROFILE=self.profile, SHARAR_HF_CACHE=str(self.hf_cache),
                   SHARAR_RESULTS=str(self.results), SHARAR_MAX_CONTEXT=str(self.max_context))
        if self.target == "flashnext":
            env["SHARAR_FLASHNEXT_CACHE"] = str(self.model_cache)
        if w3_checkpoint is not None:
            env["SHARAR_W3_CHECKPOINT"] = str(w3_checkpoint)
        elif env.get("SHARAR_W3_DECODE") == "1" and TARGET_SETTINGS[self.target].get("w3_calibrated_only"):
            env["SHARAR_W3_DECODE"] = "0"  # no nearest-rounded copies for this target
        if self.warnings:
            env["SHARAR_W3_WARNING"] = "; ".join(self.warnings)
        if self.ssd is not None:
            env["SHARAR_SSD_DIR"] = str(self.ssd)
        return env

    def engine_args(self, tp: int = 1) -> list[str]:
        """`sharar engine` arguments for these settings."""
        args = ["--target", self.target, "--host", self.host, "--port", str(self.port), "--proposer", self.proposer,
                "--verify-state", self.get("SHARAR_VERIFY_STATE", "reconstruct"), "--spec", str(self.spec),
                "--context-length", str(self.max_context), "--max-batch", self.get("SHARAR_MAX_BATCH", "4"),
                "--draft-precision", self.get("SHARAR_DRAFT_PRECISION", "bf16"),
                "--accept-default", self.get("SHARAR_ACCEPT_DEFAULT", "off")]
        if self.get("SHARAR_ACCEPT_SOFT"):
            args += ["--accept-soft", self.get("SHARAR_ACCEPT_SOFT")]
        if self.get("SHARAR_SPEC_DEPTHS") and self.spec:
            args += ["--spec-depths", self.get("SHARAR_SPEC_DEPTHS")]
        if self.draft_dir is not None:
            args += ["--draft", str(self.draft_dir)]
        if self.traces is not None:
            args += ["--traces", str(self.traces), "--traces-max-mib", self.get("SHARAR_TRACES_MAX_MIB", "32768")]
        if tp > 1:
            args += ["--tp", str(tp)]
        return args


def _finite_nonnegative(text: str) -> bool:
    try:
        return math.isfinite(float(text)) and float(text) >= 0
    except ValueError:
        return False


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def resolve(target: str | None = None, profile: str | None = None, overrides: dict[str, str] | None = None,
            env: dict[str, str] | None = None, *, traces: bool = False, tp: int = 1) -> Settings:
    """One target's settings. `overrides` are command-line choices (SHARAR_* names); `traces` turns recording on (as does
    SHARAR_TRACES naming a directory)."""
    env = dict(os.environ if env is None else env)
    file_values = read_config_file(config_file(env), env)
    user = {**file_values, **{k: v for k, v in env.items() if k.startswith("SHARAR_")}}
    name = target_name(target or (overrides or {}).get("SHARAR_TARGET") or user.get("SHARAR_TARGET"))
    values: dict[str, str] = {}
    values.update({k: v for k, v in TARGET_SETTINGS[name].items() if k.startswith("SHARAR_")})
    values.update(overrides or {})
    values.update({k: v for k, v in user.items() if v != "" and k not in (overrides or {})})
    values["SHARAR_TARGET"] = name
    profile = profile or values.get("SHARAR_PROFILE") or "fast"
    defaults = profile_defaults(profile, name, values.get("SHARAR_FLASHNEXT_EXPERTS", "w3"), tp, values["SHARAR_PROPOSER"])
    for k, v in defaults.items():
        if k == "SHARAR_SSM_STATE" and values.get("SHARAR_VERIFY_STATE", "reconstruct") != "reconstruct":
            continue  # the quantized state commits through the verify ring: other accepted states keep FP32
        values.setdefault(k, v)
    if defaults.get("SHARAR_SPEC_DEPTHS") and values.get("SHARAR_SPEC_DEPTHS") == defaults["SHARAR_SPEC_DEPTHS"] \
            and values.get("SHARAR_SPEC", "").isdigit() and values.get("SHARAR_MAX_BATCH", "4").isdigit():
        # the profile's depths per batch bucket stay within a depth and a batch size the user chose; a batch limit that
        # is not one of the profile's buckets is a bucket of its own, verifying what the bucket it would pad to does
        spec, batch = int(values["SHARAR_SPEC"]), int(values.get("SHARAR_MAX_BATCH", "4"))
        named = sorted((int(b), min(int(d), spec)) for b, d in
                       (item.split(":") for item in values["SHARAR_SPEC_DEPTHS"].split(",")))
        kept = {b: d for b, d in named if b <= batch}
        wider = [d for b, d in named if b > batch]
        if wider and batch not in kept:
            kept[batch] = wider[0]
        values["SHARAR_SPEC_DEPTHS"] = ",".join(f"{b}:{d}" for b, d in sorted(kept.items()))
    if defaults.get("SHARAR_SPEC_POLICY") == values.get("SHARAR_SPEC_POLICY") == "budget" and values.get("SHARAR_SPEC") == "0" \
            and "SHARAR_SPEC_POLICY" not in {**user, **(overrides or {})}:
        values.pop("SHARAR_SPEC_POLICY")  # plain decode: the profile's adaptive verification has nothing to verify
    values["SHARAR_PROFILE"] = profile

    home = Path(env.get("HOME") or Path.home())
    path = lambda v: Path(expand(str(v), env))  # noqa: E731
    hf_cache = path(values.get("SHARAR_HF_CACHE") or env.get("HF_HOME") or home / ".cache" / "huggingface")
    results = path(values.get("SHARAR_RESULTS") or Path(env.get("XDG_DATA_HOME") or home / ".local" / "share") / "sharar")
    model_cache = path(values.get("SHARAR_FLASHNEXT_CACHE") or hf_cache) if name == "flashnext" else hf_cache

    # validation (the engine checks the rest)
    _check(values.get("SHARAR_DRAFT_PRECISION", "bf16") in ("bf16", "fp8"), "SHARAR_DRAFT_PRECISION must be bf16 or fp8")
    for key in ("SHARAR_W3_DECODE", "SHARAR_DEVICE_ACCEPT", "SHARAR_MTP_SAMPLE"):
        _check(values.get(key, "0") in ("0", "1"), f"{key} must be 0 or 1")
    _check(_finite_nonnegative(values.get("SHARAR_ACCEPT_SOFT", "0")), "SHARAR_ACCEPT_SOFT must be a finite number >= 0 (0: exact)")
    batch = values.get("SHARAR_MAX_BATCH", "4")
    _check(batch.isdigit() and 1 <= int(batch) <= 128, "SHARAR_MAX_BATCH must be in [1, 128]")
    depths = values.get("SHARAR_SPEC_DEPTHS", "")
    _check(not depths or re.fullmatch(r"\d+:\d+(,\d+:\d+)*", depths) is not None, "SHARAR_SPEC_DEPTHS must be BUCKET:DEPTH pairs, e.g. 8:1,16:0")
    _check(values.get("SHARAR_SPEC_POLICY", "fixed") in ("fixed", "budget"), "SHARAR_SPEC_POLICY must be fixed or budget")
    _check(values.get("SHARAR_FLASHNEXT_EXPERTS", "w3") in ("w3", "nvfp4"), "SHARAR_FLASHNEXT_EXPERTS must be w3 or nvfp4")
    _check(values.get("SHARAR_BOLT_EXPERTS", "b12x") in ("bf16", "fp8", "b12x"), "SHARAR_BOLT_EXPERTS must be bf16, fp8 or b12x")
    _check(values.get("SHARAR_DRAFT_HEAD", "fp8") in ("fp8", "w3", "w2"), "SHARAR_DRAFT_HEAD must be fp8, w3 or w2")
    _check(values.get("SHARAR_QSA_SPARSE_GRAPHS", "1") in ("0", "1"), "SHARAR_QSA_SPARSE_GRAPHS must be 0 or 1")
    _check(values.get("SHARAR_PLE_RESIDENT", "auto") in ("auto", "0"), "SHARAR_PLE_RESIDENT must be auto or 0")
    ssm_states = SSM_STATES[TARGET_SETTINGS[name].get("recurrence", "gdn")]
    _check(values.get("SHARAR_SSM_STATE", "fp32") in ssm_states, "SHARAR_SSM_STATE must be " + ", ".join(ssm_states[:-1]) + " or " + ssm_states[-1])
    kv_format = values.get("SHARAR_KV_FORMAT", "fp8")
    _check(kv_format in ("fp8", "nvfp4", "sim-fp4", "sim-k8v4", "sim-bf16"),
           "SHARAR_KV_FORMAT must be fp8 or nvfp4 (sim-fp4, sim-k8v4, sim-bf16: the accuracy screen)")
    _check(kv_format == "fp8" or tp == 1, "SHARAR_KV_FORMAT=nvfp4 is unmeasured on two Sparks: serve FP8 KV tensor-parallel")
    _check(kv_format == "fp8" or not TARGET_SETTINGS[name].get("fp8_kv_only"), f"{name} serves an FP8 KV cache only (SHARAR_KV_FORMAT=fp8): NVFP4 KV is not supported for it yet")
    _check(values.get("SHARAR_DRAFT_VOCAB", "0").isdigit(), "SHARAR_DRAFT_VOCAB must be a row count (0 = the full lm_head)")
    _check(values.get("SHARAR_SSD", "0") in ("0", "1"), "SHARAR_SSD must be 0 or 1")
    block = values.get("SHARAR_SSD_BLOCK_TOKENS", "256")
    _check(block.isdigit() and int(block) > 0 and 2048 % int(block) == 0, "SHARAR_SSD_BLOCK_TOKENS must divide 2048")
    _check(re.fullmatch(r"[0-9]+(\.[0-9]+)?", values.get("SHARAR_SSD_GIB", "200")) is not None, "SHARAR_SSD_GIB must be a number of GiB")
    _check(re.fullmatch(r"[0-9]+(\.[0-9]+)?", values.get("SHARAR_HOST_FLOOR_GIB", "6")) is not None,
           "SHARAR_HOST_FLOOR_GIB must be a number of GiB")
    # retired: the guard acts at half the host floor. A value left from main's 16 GiB guard would now stop the server
    # after warmup, since the pools leave only the floor free, so it is refused rather than ignored
    _check("SHARAR_MIN_AVAILABLE_KIB" not in values,
           "SHARAR_MIN_AVAILABLE_KIB is retired: the memory guard acts at half SHARAR_HOST_FLOOR_GIB (6 GiB by default); "
           "set that instead")
    _check(values.get("SHARAR_TRACES_MAX_MIB", "32768").isdigit() and int(values.get("SHARAR_TRACES_MAX_MIB", "32768")) > 0,
           "SHARAR_TRACES_MAX_MIB must be a positive integer")
    if not {**user, **(overrides or {})}.get("SHARAR_DRAFT_PRECISION") and not (
            int(values.get("SHARAR_SPEC", "15")) and values["SHARAR_PROPOSER"] == "dflash2"):
        values["SHARAR_DRAFT_PRECISION"] = "bf16"  # the target's FP8 draft applies when its parallel draft runs
    precision, proposer, spec = values.get("SHARAR_DRAFT_PRECISION"), values["SHARAR_PROPOSER"], int(values.get("SHARAR_SPEC", "15"))
    if name == "bolt":
        _check(spec == 0 or (proposer == "dflash2" and 1 <= spec <= 7),
               "Bolt has no in-checkpoint MTP; use LiLiCorr DFlash2 at depth 1..7, or SHARAR_SPEC=0")
        _check(precision == "bf16", "Bolt LiLiCorr uses its published W4A16/BF16 draft weights")
        _check(kv_format in ("fp8", "sim-bf16"), "Bolt supports FP8 KV or sim-bf16 for parity")
    _check(proposer in ("mtp", "dflash2"), f"SHARAR_PROPOSER must be mtp or dflash2 (got {proposer!r})")
    _check(precision != "fp8" or (spec and proposer == "dflash2"), "FP8 draft precision requires enabled DFlash2 speculation")
    if values.get("SHARAR_SPEC_POLICY") == "budget":
        _check(name == "qwen38" and spec and proposer == "dflash2",
               "SHARAR_SPEC_POLICY=budget (adaptive verification) serves the 27B's DFlash2 with SHARAR_SPEC above 0; "
               "Flash-Next, Nemotron and MTP keep their depth tables (SHARAR_SPEC_POLICY=fixed)")
        _check(not any(d == "0" for d in re.findall(r":(\d+)", depths)),
               "SHARAR_SPEC_DEPTHS cannot name depth 0 (plain decode) under SHARAR_SPEC_POLICY=budget, which verifies "
               "at least one draft a request: set SHARAR_SPEC_POLICY=fixed")
        _check(float(values.get("SHARAR_SPEC_STEP_COST") or 0) == 0,
               "SHARAR_SPEC_STEP_COST (adaptive depth) and SHARAR_SPEC_POLICY=budget (adaptive verification) exclude "
               "each other: the budget decides each request's depth itself")
    max_context = int(values.get("SHARAR_MAX_CONTEXT") or DEFAULT_CONTEXT)
    _check(2 <= max_context <= DEFAULT_CONTEXT, f"SHARAR_MAX_CONTEXT must be in [2, {DEFAULT_CONTEXT}]")
    if tp > 1:
        _check(name in ("qwen38", "flashnext"), "tensor parallel serves the 27B and Flash-Next (Nemotron serves on one Spark)")
        _check(not (name == "qwen38" and spec and proposer == "mtp"),
               "the 27B serves tensor-parallel with DFlash2, not MTP")
        _check(not (name == "flashnext" and spec and proposer == "dflash2"),
               "Flash-Next serves DFlash2 on one Spark: serve MTP tensor-parallel")
        _check(values.get("SHARAR_DRAFT_VOCAB", "0") == "0", "a reduced draft vocabulary (SHARAR_DRAFT_VOCAB) is not served tensor-parallel")

    # the draft: the pinned publisher draft, or a DFlash2 checkpoint directory
    draft_value = values.get("SHARAR_DRAFT", "")
    draft_dir = None
    if draft_value and draft_value != "pinned":
        draft_dir = path(draft_value).resolve()
        _check((draft_dir / "config.json").is_file() and (draft_dir / "model.safetensors").is_file(),
               f"SHARAR_DRAFT={draft_dir} needs config.json and model.safetensors")
        _check(proposer == "dflash2" and spec > 0, "a draft checkpoint requires enabled DFlash2 speculation")

    trace_value = values.get("SHARAR_TRACES", "")
    trace_dir = None
    if trace_value != "off" and (traces or trace_value):
        trace_dir = path(trace_value or results / "traces")
    ssd = (path(values.get("SHARAR_SSD_DIR") or results / "ssd" / name)  # the same path on each Spark; one store per target
           if values.get("SHARAR_SSD") == "1" else None)

    return Settings(target=name, profile=profile, values=values, port=int(values["SHARAR_PORT"]),
                    host=values.get("SHARAR_HOST", "127.0.0.1"), service=values["SHARAR_SERVICE"], results=results,
                    hf_cache=hf_cache, model_cache=model_cache, max_context=max_context, traces=trace_dir,
                    draft_dir=draft_dir, ssd=ssd)
