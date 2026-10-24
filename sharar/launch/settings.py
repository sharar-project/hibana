"""What `sharar serve` serves: one target's settings, resolved from every source in one place.

Order, first wins: command-line choices (`resolve`'s `target`, `profile` and `overrides`), the process environment
(`SHARAR_*`), the user's config file (`SHARAR_CONFIG_FILE`, else `config/gb10.env` in the checkout, else
`~/.config/sharar/sharar.env`; `KEY=VALUE` lines), the target's settings (`TARGET_SETTINGS`), the serving profile's
defaults (`profile_defaults`), then the built-in defaults. `resolve` returns the engine's environment and its `sharar engine` arguments; nothing here
imports torch, so the launcher stays fast and testable on a CPU.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


CHECKOUT = Path(__file__).resolve().parents[2]  # the repository root when sharar is installed editable

# Canonical target names and the names users type.
ALIASES = {"27b": "qwen38", "27B": "qwen38", "qwen38": "qwen38", "qwen3.8-27b": "qwen38",
           "flashnext": "flashnext", "flash-next": "flashnext", "Flash-Next": "flashnext"}
TARGET_CHOICES = ("27b", "flashnext", "qwen38", "flash-next")

# What each target serves when nothing else says so (port, service, drafter).
TARGET_SETTINGS = {
    "qwen38": {"SHARAR_SERVICE": "sharar-qwen38", "SHARAR_PORT": "30001", "SHARAR_PROPOSER": "dflash2", "SHARAR_SPEC": "15",
               "SHARAR_VERIFY_STATE": "reconstruct", "SHARAR_DRAFT_PRECISION": "fp8"},
    "flashnext": {"SHARAR_SERVICE": "sharar-flashnext", "SHARAR_PORT": "30004", "SHARAR_PROPOSER": "mtp",
                  "SHARAR_VERIFY_STATE": "reconstruct", "SHARAR_DRAFT_PRECISION": "fp8", "SHARAR_DRAFT": "pinned",
                  "SHARAR_DRAFT_HEAD": "w2", "SHARAR_FLASHNEXT_TRUNK": "fp8"},
}

# The serving profiles: the default of every speed/fidelity knob a setting does not fix.
#   exact     the target's own distribution: exact acceptance, no lossy weight copies (benchmarks)
#   fast      (default) everything measured to be worth it: relaxed top-3 within 2.0 logits for greedy requests and
#             Flash-Next's sampled ones, top-2 within 1.0 for the 27B's sampled ones (at 3:2.0 its sampled thinking looped
#             in Claude Code sessions), the calibrated 3-bit copies, device-resident acceptance on Flash-Next,
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
    flashnext = target == "flashnext"
    if profile == "exact":
        p = {"SHARAR_ACCEPT_DEFAULT": "off", "SHARAR_ACCEPT_DEFAULT_SAMPLED": "0", "SHARAR_W3_DECODE": "0", "SHARAR_DEVICE_ACCEPT": "0"}
    elif profile in ("fast", "throughput"):
        p = {"SHARAR_ACCEPT_DEFAULT": "3:2.0", "SHARAR_ACCEPT_DEFAULT_SAMPLED": "1", "SHARAR_W3_DECODE": "1",
             "SHARAR_DEVICE_ACCEPT": "1" if flashnext else "0"}
        if not flashnext:
            p["SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD"] = "2:1.0"
    else:
        raise ValueError(f"unknown serving profile {profile!r} (choose from {', '.join(PROFILE_NAMES)})")
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
                # top-2 within 1.0, the 27B's sampled rule: at Flash-Next's top-3 within 2.0, DFlash2 lost 3.3 points of
                # GSM8K-300 one request at a time against MTP with its 3-bit draft head (-6.2 to -0.4); at 2:1.0 it ties
                # (-0.3, -2.9 to +2.2) for 5% fewer tokens a round there
                p["SHARAR_ACCEPT_DEFAULT"] = "2:1.0"
                if profile == "fast":
                    p.update(SHARAR_SPEC="9", SHARAR_SPEC_DEPTHS="2:7,4:7")
        elif profile == "fast":  # adaptive depth: h2h greedy median 92.2 -> 101.3 tok/s, 4 streams within noise
            # two Sparks: 0.14 drafted chat and prose too deep (-12%, -7% against depth 4); 0.2 keeps them within noise
            p.update(SHARAR_SPEC="8", SHARAR_SPEC_STEP_COST="0.14" if tp == 1 else "0.2", SHARAR_SPEC_DEPTHS="2:4,4:4")
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


DEFAULT_CONTEXT = 262144


def target_name(name: str | None) -> str:
    try:
        return ALIASES[name or "qwen38"]
    except KeyError:
        raise ValueError(f"unknown target {name!r} (choose 27b or flashnext)") from None


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
        elif env.get("SHARAR_W3_DECODE") == "1" and self.target == "flashnext":
            env["SHARAR_W3_DECODE"] = "0"  # Flash-Next's 3-bit copies are calibrated only
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
        accept, sampled = self.get("SHARAR_ACCEPT_DEFAULT", "off"), self.get("SHARAR_ACCEPT_DEFAULT_SAMPLED", "0")
        if sampled == "1" and accept != "off":
            args.append("--accept-default-sampled")
            if self.get("SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD"):
                args += ["--accept-default-sampled-threshold", self.get("SHARAR_ACCEPT_DEFAULT_SAMPLED_THRESHOLD")]
        if self.get("SHARAR_SPEC_DEPTHS") and self.spec:
            args += ["--spec-depths", self.get("SHARAR_SPEC_DEPTHS")]
        if self.draft_dir is not None:
            args += ["--draft", str(self.draft_dir)]
        if self.traces is not None:
            args += ["--traces", str(self.traces), "--traces-max-mib", self.get("SHARAR_TRACES_MAX_MIB", "32768")]
        if tp > 1:
            args += ["--tp", str(tp)]
        return args


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
    values.update(TARGET_SETTINGS[name])
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
    values["SHARAR_PROFILE"] = profile

    home = Path(env.get("HOME") or Path.home())
    path = lambda v: Path(expand(str(v), env))  # noqa: E731
    hf_cache = path(values.get("SHARAR_HF_CACHE") or env.get("HF_HOME") or home / ".cache" / "huggingface")
    results = path(values.get("SHARAR_RESULTS") or Path(env.get("XDG_DATA_HOME") or home / ".local" / "share") / "sharar")
    model_cache = path(values.get("SHARAR_FLASHNEXT_CACHE") or hf_cache) if name == "flashnext" else hf_cache

    # validation (the engine checks the rest)
    _check(values.get("SHARAR_DRAFT_PRECISION", "bf16") in ("bf16", "fp8"), "SHARAR_DRAFT_PRECISION must be bf16 or fp8")
    for key in ("SHARAR_W3_DECODE", "SHARAR_DEVICE_ACCEPT", "SHARAR_ACCEPT_DEFAULT_SAMPLED"):
        _check(values.get(key, "0") in ("0", "1"), f"{key} must be 0 or 1")
    batch = values.get("SHARAR_MAX_BATCH", "4")
    _check(batch.isdigit() and 1 <= int(batch) <= 128, "SHARAR_MAX_BATCH must be in [1, 128]")
    depths = values.get("SHARAR_SPEC_DEPTHS", "")
    _check(not depths or re.fullmatch(r"\d+:\d+(,\d+:\d+)*", depths) is not None, "SHARAR_SPEC_DEPTHS must be BUCKET:DEPTH pairs, e.g. 8:1,16:0")
    _check(values.get("SHARAR_FLASHNEXT_EXPERTS", "w3") in ("w3", "nvfp4"), "SHARAR_FLASHNEXT_EXPERTS must be w3 or nvfp4")
    _check(values.get("SHARAR_DRAFT_HEAD", "fp8") in ("fp8", "w3", "w2"), "SHARAR_DRAFT_HEAD must be fp8, w3 or w2")
    _check(values.get("SHARAR_QSA_SPARSE_GRAPHS", "1") in ("0", "1"), "SHARAR_QSA_SPARSE_GRAPHS must be 0 or 1")
    _check(values.get("SHARAR_PLE_RESIDENT", "auto") in ("auto", "0"), "SHARAR_PLE_RESIDENT must be auto or 0")
    _check(values.get("SHARAR_SSM_STATE", "fp32") in ("fp32", "int8", "int4"), "SHARAR_SSM_STATE must be fp32, int8 or int4")
    kv_format = values.get("SHARAR_KV_FORMAT", "fp8")
    _check(kv_format in ("fp8", "nvfp4", "sim-fp4", "sim-k8v4", "sim-bf16"),
           "SHARAR_KV_FORMAT must be fp8 or nvfp4 (sim-fp4, sim-k8v4, sim-bf16: the accuracy screen)")
    _check(kv_format == "fp8" or tp == 1, "SHARAR_KV_FORMAT=nvfp4 is unmeasured on two Sparks: serve FP8 KV tensor-parallel")
    _check(values.get("SHARAR_DRAFT_VOCAB", "0").isdigit(), "SHARAR_DRAFT_VOCAB must be a row count (0 = the full lm_head)")
    _check(values.get("SHARAR_SSD", "0") in ("0", "1"), "SHARAR_SSD must be 0 or 1")
    block = values.get("SHARAR_SSD_BLOCK_TOKENS", "256")
    _check(block.isdigit() and int(block) > 0 and 2048 % int(block) == 0, "SHARAR_SSD_BLOCK_TOKENS must divide 2048")
    _check(re.fullmatch(r"[0-9]+(\.[0-9]+)?", values.get("SHARAR_SSD_GIB", "200")) is not None, "SHARAR_SSD_GIB must be a number of GiB")
    _check(values.get("SHARAR_TRACES_MAX_MIB", "32768").isdigit() and int(values.get("SHARAR_TRACES_MAX_MIB", "32768")) > 0,
           "SHARAR_TRACES_MAX_MIB must be a positive integer")
    if not {**user, **(overrides or {})}.get("SHARAR_DRAFT_PRECISION") and not (
            int(values.get("SHARAR_SPEC", "15")) and values["SHARAR_PROPOSER"] == "dflash2"):
        values["SHARAR_DRAFT_PRECISION"] = "bf16"  # the target's FP8 draft applies when its parallel draft runs
    precision, proposer, spec = values.get("SHARAR_DRAFT_PRECISION"), values["SHARAR_PROPOSER"], int(values.get("SHARAR_SPEC", "15"))
    _check(proposer in ("mtp", "dflash2"), f"SHARAR_PROPOSER must be mtp or dflash2 (got {proposer!r})")
    _check(precision != "fp8" or (spec and proposer == "dflash2"), "FP8 draft precision requires enabled DFlash2 speculation")
    max_context = int(values.get("SHARAR_MAX_CONTEXT") or DEFAULT_CONTEXT)
    _check(2 <= max_context <= DEFAULT_CONTEXT, f"SHARAR_MAX_CONTEXT must be in [2, {DEFAULT_CONTEXT}]")
    if tp > 1:
        _check(name in ("qwen38", "flashnext"), "tensor parallel serves the 27B and Flash-Next")
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
