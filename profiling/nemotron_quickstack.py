#!/usr/bin/env python3
"""Quick-stack proxy for Nemotron-3.5-Super on Sharar: a speed and accuracy screen per serving configuration in about
five minutes, in front of the thorough tools (tools/h2h-pr.py for served speed, tools/evaluate.py / the lossy gate
for GSM8K-300 and the quick suite).

    python profiling/nemotron_quickstack.py --name base --config exact --config fast --config fast+w3
    python profiling/nemotron_quickstack.py --name stack1 --config fast+w3 \\
        --config fp16=fast+w3:SSM_STATE=fp16 --config fp16-d6=fp16:SPEC=6 \\
        --config fp16-share=fp16:MOE_KEEP=12,MOE_BUDGET=48:100000
    python profiling/nemotron_quickstack.py --name stack1 --table      # the table of what has finished

Do not wrap it in flock: it takes the shared GPU lock ($SHARAR_RESULTS/.gpu.lock) itself, in windows of at most
--hold-minutes (25), and gives it up for --yield-seconds (90) between windows; a window ends between configurations
(the model process exits, so the next window reloads). A rerun with the same --name skips the configurations that
finished, so an interrupted run resumes.

**Configurations** (`--config NAME=BASE[:KEY=VALUE,...]`, repeatable, or `--configs FILE.json`, a list of
{"name", "base", "env"}): BASE is a serving profile (exact / fast / throughput), a built-in (`exact`, `fast`,
`fast+w3` = fast with both calibrated 3-bit copies, the trunk's and the routed experts'), or a configuration named
before it; KEY=VALUE are SHARAR_* settings (the prefix may be left out) added on top. Each resolves for the target
exactly as `sharar serve nemotron --<profile>` resolves it with those settings in its environment
(sharar/launch/settings.py: the user's config file, the target's settings, the profile's defaults), as h2h-pr's
`--arm` does. The first configuration is the base the table's deltas are taken against.

**One load per weight set.** Settings read when the model loads (SHARAR_W3_DECODE / SHARAR_W3_CHECKPOINT,
SHARAR_NEMOTRON_EXPERTS, ...; LOAD_FIELDS) form a weight set, loaded once in its own worker process; everything else is
applied per configuration on the loaded model: the executor is rebuilt (its state pool, so SHARAR_SSM_STATE; the MTP
proposer and its draft head; depth and depth per batch; device acceptance; graphs captured under the configuration),
the trunk's MoE blocks take the sharing settings (SHARAR_MOE_KEEP / BUDGET / EXTRA / FLOORS / TAU / FREE_RIDERS, as
`NemotronMoe.__init__` reads them) before capture, and greedy requests carry the configuration's acceptance rule.

**Per configuration** (about 4-6 minutes: executor ~1, speed ~1.5, KL ~1, GSM8K ~1.5):

- speed: batch 1 = six h2h_v1 decode prompts (code-lru, code-go, math-prob, prose-history, json-extract,
  chat-advice: their median tracks the 18-prompt h2h median in every measured arm), one at a time, chat template,
  thinking off, natural stops, up to --tokens (384) tokens, cold prefix cache; the median of their decode tok/s (prefill
  and the first token excluded), as h2h reports. Batch 4 = h2h's eight concurrency prompts as two batches of four,
  --tokens each (ignore_eos: four live rows throughout, like closed-loop streams), aggregate decode tok/s. Tokens per
  verify round (per request) and the share of relaxed acceptances at batch 1. `h2h~` columns divide by CALIBRATION
  (the proxy / h2h ratio measured on the base configurations, below) to read the proxy in h2h terms.
- KL: teacher-forced against a frozen reference (the exact profile's greedy continuations of 24 held-out prompts,
  --kl-tokens each: the 22 lossy-gate prompts outside the speed set, code / math / prose / multilingual / chat /
  structured, plus two GSM8K test items past the first 300), per position the truncated KL(reference || config) over
  the reference's top 32, top-1 agreement and the NLL of the reference token. The continuation is scored the way a
  served verify round computes it: calls of depth + 1 rows (the configuration's batch-1 depth), the trunk's MoE routed
  as one verify call of that request (so sharing's floor / cap act on the served call shape), each Mamba layer
  advanced token by token through the decode kernel (an FP16 state rounds after every token as decode and verify
  store it; a plain chunked prefill would round once a chunk), the 3-bit copies on (calls of at most 64 rows). The
  prompt prefills in 16-row calls (PROMPT_ROWS: wider NVFP4 MoE calls run b12x, which is not reproducible run to run),
  so a configuration scores bitwise the same every time and exact against itself is 0. The reference is scored at
  each call shape in --ref-rows (4-7 rows: depths 3-6) and a configuration is compared with exact at its own shape:
  merely changing the call shape moves the logits as much as a lossy option (exact at 6-row calls against 4-row:
  KL ~0.017, the network amplifying last-bit differences), which would otherwise be charged to the option. What it cannot see: relaxed acceptance and depth change which tokens are emitted, not
  the logits; GSM8K below sees them. KL vs anchor: the same pass against the anchor configuration's own top-32
  (--kl-anchor, saved when it runs first in the run: OUT/<name>/anchor-r<rows>.pt), i.e. what a stacked knob adds on
  top of the 3-bit stack. Losses do not add: the 3-bit copies' KL (~0.1) dominates the KL against exact, and a knob
  whose KL against exact is 0.02 alone can leave the stack's KL against exact unchanged; KL vs anchor isolates it.
- GSM8K-50: the first --gsm (50) items of GSM8K test (the gate's GSM8K-300 starts with them), sharar-eval's prompt
  and grader, greedy with the configuration's acceptance rule, four requests at a time, up to --gsm-max-tokens (512),
  thinking off: accuracy, paired wins / losses against the reference (the exact profile's own answers), and the share
  of answers identical to the reference token for token.

The reference (continuations, top-32 log-probabilities, the floor, exact's GSM8K-50 answers) is built once with the
exact profile and cached under OUT/reference/<key>/ (--rebuild-reference). Rows: OUT/<name>/results.jsonl, one per
configuration (settings, every metric, per-prompt values, stage seconds); the table: OUT/<name>/table.md. Numbers are
Sharar's on the machine named in each row.

Flags: `KL!` when a configuration's KL vs anchor (--kl-anchor, default fast+w3: the 3-bit stack) exceeds
--kl-margin (25%) of the anchor's own KL against exact (without an anchor pass: its KL against exact more than 25%
above the anchor's); `GSM!` when it loses at least 3 more GSM8K-50 items than it wins against the
reference. A configuration worth a full h2h + GSM8K-300 is one that is faster without a flag.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import gc
import hashlib
import json
import math
import os
import re
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT_ROOT = Path("/home/brian/Desktop/research/nemotron-logs/quickstack")
H2H_WORKLOAD = Path("/home/brian/Desktop/research/sharar-eval-58ea950/evals/workloads/h2h_v1.json")
LOSSY_WORKLOAD = REPO / "tools" / "gate" / "workloads" / "lossy_v1.json"
W3_BOTH = "/home/brian/data/sharar-results/w3-checkpoints/nemotron-trunk3-experts3-gptq-20261024"

# speed prompts (h2h_v1 decode ids): their median tracks the h2h 18-prompt median (exact 46 vs 45.9, fast 56 vs 54.7,
# fast + both 3-bit copies 80.5 vs 81.0)
SPEED_B1 = ("code-lru", "code-go", "math-prob", "prose-history", "json-extract", "chat-advice")
GSM_KL = (1000, 1001)  # GSM8K test items in the KL set (past the first 300 the gate and GSM8K-50 read)
TOPK = 32
CONTEXT = 8192
PROMPT_ROWS = 16  # prompt prefill calls (the small-call kernels: wider NVFP4 MoE calls run b12x, not reproducible run to run)
REF_VERSION = 2  # bump when the reference's meaning changes (it keys the cache)

BUILTIN = {"exact": ("exact", {}), "fast": ("fast", {}),
           "fast+w3": ("fast", {"SHARAR_W3_DECODE": "1", "SHARAR_W3_CHECKPOINT": W3_BOTH})}
# Served h2h_v1 (tools/h2h-pr.py, Sharar, spark-9931, greedy): 1 request median decode tok/s, 4 closed-loop streams.
H2H = {"exact": (45.9, 79.8), "fast": (54.7, 92.6), "fast+w3": (81.0, 132.7)}
# proxy / h2h, measured on the three base configurations (README: Quick-stack proxy); None: not calibrated
CALIBRATION = {"b1": 1.039, "b4": 1.000}

# EngineConfig fields read when the model loads: configurations that differ in one of them are separate weight sets
LOAD_FIELDS = ("target", "hf_cache", "w3_decode", "w3_checkpoint", "nemotron_experts", "stable_numerics", "cuda_arch")
REPORTED = ("SHARAR_PROFILE", "SHARAR_SPEC", "SHARAR_SPEC_DEPTHS", "SHARAR_SPEC_STEP_COST", "SHARAR_ACCEPT_DEFAULT",
            "SHARAR_DEVICE_ACCEPT", "SHARAR_DRAFT_HEAD", "SHARAR_SSM_STATE", "SHARAR_W3_DECODE", "SHARAR_W3_CHECKPOINT",
            "SHARAR_NEMOTRON_EXPERTS", "SHARAR_MOE_KEEP", "SHARAR_MOE_BUDGET", "SHARAR_MOE_EXTRA", "SHARAR_MOE_FLOORS",
            "SHARAR_MOE_TAU", "SHARAR_MOE_FREE_RIDERS", "SHARAR_VERIFY_STATE")
KEEP_ENV = ("SHARAR_HF_CACHE", "SHARAR_RESULTS", "SHARAR_CONFIG_FILE")  # the user's paths; other SHARAR_* knobs come from configs


def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- configurations

def parse_config(text: str, known: dict) -> tuple[str, str, dict]:
    """'NAME=BASE[:K=V,...]' or a built-in / earlier NAME -> (name, profile, SHARAR_* overrides)."""
    if "=" not in text.split(":", 1)[0]:
        if text not in known:
            raise SystemExit(f"--config {text}: not a built-in ({', '.join(BUILTIN)}) or an earlier configuration")
        return text, *known[text]
    name, rest = text.split("=", 1)
    base, _, sets = rest.partition(":")
    env = {}
    for item in re.split(r",(?=[A-Za-z_0-9]+=)", sets) if sets else []:
        k, _, v = item.partition("=")
        env[k if k.startswith("SHARAR_") else f"SHARAR_{k}"] = v
    return name, *combine(base, env, known)


def combine(base: str, env: dict, known: dict) -> tuple[str, dict]:
    from sharar.launch.settings import PROFILE_NAMES
    if base in known:
        profile, parent = known[base]
        return profile, {**parent, **env}
    if base in PROFILE_NAMES:
        return base, dict(env)
    raise SystemExit(f"configuration base {base!r}: a profile ({', '.join(PROFILE_NAMES)}), a built-in or an earlier name")


def resolve(profile: str, overrides: dict) -> dict:
    """The engine environment `sharar serve nemotron --<profile>` would run with these settings, plus the spec choices."""
    from sharar.engine.config import EngineConfig
    from sharar.launch import preflight
    from sharar.launch import settings as launch
    from sharar.api.validation import parse_accept_default
    user = {k: v for k, v in os.environ.items() if not k.startswith("SHARAR_") or k in KEEP_ENV}
    s = launch.resolve("nemotron", profile, overrides=overrides, env=user)
    eng = s.engine_env(preflight.ensure_w3(s, fetch=False))
    eng["SHARAR_SSD"] = "0"  # cold, in-process measurement: no SSD tier
    cfg = EngineConfig.from_env(eng)
    if s.proposer != "mtp":
        raise SystemExit("the quick-stack proxy measures the MTP proposer")
    spec = int(cfg.spec) if cfg.spec not in (None, "", "auto") else 3
    accept = parse_accept_default(cfg.accept_default) if spec else (1, 0.0)
    return {"profile": profile, "overrides": overrides, "engine_env": eng, "spec": spec, "accept": list(accept),
            "verify_state": s.get("SHARAR_VERIFY_STATE", "reconstruct"),
            "settings": {k: eng[k] for k in REPORTED if k in eng}}


def engine_config(r: dict):
    from sharar.engine.config import EngineConfig
    cfg = EngineConfig.from_env(r["engine_env"])
    return cfg.replace(spec=str(r["spec"]), max_context=CONTEXT, max_batch=4,
                       accept_default=f"{r['accept'][0]}:{r['accept'][1]}", spec_depths=cfg.spec_depths if r["spec"] else None)


def weight_key(r: dict) -> str:
    cfg = engine_config(r)
    return json.dumps({f: getattr(cfg, f) for f in LOAD_FIELDS}, sort_keys=True)


def settings_hash(r: dict) -> str:
    return hashlib.sha256(json.dumps(r["engine_env"], sort_keys=True).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- inputs

def chat_ids(tok, tmpl, content: str) -> list[int]:
    return tok.encode(tmpl.render([{"role": "user", "content": content}], enable_thinking=False))


GSM_SUFFIX = "\n\nSolve the problem step by step. On the last line write only: Answer: <number>"  # sharar-eval's
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gsm_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def gsm_pred(text: str) -> str | None:
    tail = text.rsplit("Answer:", 1)
    nums = NUM.findall(tail[1] if len(tail) == 2 else text) or NUM.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def same_number(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-6)
    except ValueError:
        return a == b


def gsm_path() -> Path:
    from sharar.launch.settings import resolve as launch_resolve
    return launch_resolve("nemotron").results / "datasets" / "accuracy" / "gsm8k_test.jsonl"


def workload(args) -> dict:
    """Prompts of every stage, as text (tokenized in the worker)."""
    h2h = json.loads(args.h2h_workload.read_text())
    by_id = {p["id"]: p for p in h2h["decode"]}
    gsm = [json.loads(line) for line in gsm_path().read_text().splitlines() if line.strip()]
    kl = [{"id": p["id"], "domain": p["domain"], "prompt": p["prompt"]}
          for p in json.loads(LOSSY_WORKLOAD.read_text())["prompts"] if p["id"] not in SPEED_B1]
    kl += [{"id": f"gsm-{i}", "domain": "math", "prompt": gsm[i]["question"] + GSM_SUFFIX} for i in GSM_KL]
    return {"b1": [by_id[i]["prompt"] for i in SPEED_B1],
            "b4": [by_id[i]["prompt"] for i in h2h["concurrency"]["prompts"]],
            "kl": kl, "gsm": [{"question": g["question"], "gold": gsm_gold(g["answer"])} for g in gsm[:args.gsm]]}


def reference_dir(args, w: dict) -> Path:
    from sharar import target as targets
    t = targets.get("nemotron")
    key = hashlib.sha256(json.dumps({"v": REF_VERSION, "model": f"{t.model_id}@{t.model_revision}", "kl": w["kl"],
                                     "gsm": w["gsm"], "kl_tokens": args.kl_tokens,
                                     "gsm_max_tokens": args.gsm_max_tokens}, sort_keys=True).encode()).hexdigest()[:12]
    return args.out_root / "reference" / key


# ---------------------------------------------------------------- worker: engine side

EMU = {"on": False, "slots": None, "positions": None}


def install_emulation() -> None:
    """While EMU is on, a prefill forward computes as one served verify call: the trunk's MoE blocks route it as a
    verify call of one request (its rows its steps), and each Mamba layer advances token by token through the decode
    kernel (FP16 state rounded after every token, as decode and verify store it)."""
    import torch
    from sharar.kernels import mamba2
    from sharar.model.nemotron import moe as M
    prefill = mamba2.prefill

    def emulated_prefill(zxbcdt, params, conv_state, ssm_state, plan, *, layer=0, stochastic=True):
        if not EMU["on"]:
            return prefill(zxbcdt, params, conv_state, ssm_state, plan, layer=layer, stochastic=stochastic)
        return torch.cat([mamba2.decode(zxbcdt[i:i + 1], params, conv_state, ssm_state, EMU["slots"],
                                        EMU["positions"][i:i + 1], layer=layer, stochastic=stochastic)
                          for i in range(zxbcdt.shape[0])])

    mamba2.prefill = emulated_prefill
    call = M.NemotronMoe.__call__

    def emulated_call(self, x, ctx=None):
        if EMU["on"] and ctx is not None and ctx.mode == "prefill" and self.block is not None:
            ctx = SimpleNamespace(mode="verify", seq_slots=None, verify_steps=x.shape[0], slots=None)
        return call(self, x, ctx)

    M.NemotronMoe.__call__ = emulated_call


def apply_moe(model, cfg) -> dict:
    """The trunk's MoE blocks take cfg's sharing settings, as NemotronMoe.__init__ reads them (the MTP head's are
    built with the proposer, from the installed config)."""
    from sharar.model.nemotron.moe import NemotronMoe, floors
    table = floors(cfg.moe_floors) if cfg.moe_floors else None
    for layer in model.layers:
        m = layer.mixer
        if not isinstance(m, NemotronMoe):
            continue
        floor = table.get(m.block, m.topk) if table is not None else cfg.moe_keep
        m.tau = cfg.moe_tau if floor or cfg.moe_floors is None else 0.0
        if m.tau and not floor:
            floor = 1
        m.keep = floor if floor < m.topk and (m.block is not None or not cfg.moe_floors) else 0
        m.budget = cfg.moe_budget if m.keep else 0
        m.per_request = cfg.moe_budget_per_request
        m.free_riders = cfg.moe_free_riders
        m.extra = cfg.moe_extra if m.keep else 0
    moes = [b.mixer for b in model.layers if isinstance(b.mixer, NemotronMoe)]
    return {"keep": sorted({m.keep for m in moes}), "budget": sorted({m.budget for m in moes})}


def build_executor(model, r: dict):
    from sharar.cache.spec_budget import SpecConfig
    from sharar.engine import config
    from sharar.engine.executor import Executor
    cfg = engine_config(r)
    config.install(cfg)
    moe = apply_moe(model, cfg)
    spec = SpecConfig("mtp", r["spec"], r["verify_state"], "bf16", None) if r["spec"] else None
    # fixed pools: a planned executor takes the spare memory, which the next one's plan would find still held
    ex = Executor(model, log=None, max_batch=4, max_context=CONTEXT, num_pages=65536, num_slots=12, spec_config=spec)
    ex.capture_graphs(log=None)
    ex.warmup(log=None)
    policy = ex.spec.depth_policy if ex.spec is not None else None
    depth1 = getattr(policy, "depth", None) or r["spec"]
    return ex, {"moe": moe, "depth_b1": depth1, "depths": getattr(policy, "depths", None),
                "ssm": ex.ssm_format.describe()}


def params(r: dict, n: int, ignore_eos: bool = False):
    from sharar import target as targets
    from sharar.engine.sampler import SamplingParams
    return SamplingParams(max_new_tokens=n, ignore_eos=ignore_eos, accept_top_k=r["accept"][0], accept_margin=r["accept"][1],
                          stop_token_ids=tuple(targets.get("nemotron").stop_token_ids), trace=False)


def measure_speed(ex, r: dict, ids: dict, tokens: int) -> dict:
    ex.flush_cache()
    ex.generate(ids["b1"][0], params(r, 32))  # first-request effects
    b1 = []
    for i, p in enumerate(ids["b1"]):
        ex.flush_cache()
        q = ex.generate(p, params(r, tokens))
        n = len(q.output_ids) - 1
        b1.append({"id": SPEED_B1[i], "tokens": n, "tok_s": n / q.decode_seconds, "rounds": q.spec_rounds,
                   "relaxed": q.spec_relaxed})
    b4 = []
    for g in range(len(ids["b4"]) // 4):
        ex.flush_cache()
        qs = ex.generate_batch([(p, params(r, tokens, ignore_eos=True)) for p in ids["b4"][4 * g:4 * g + 4]])
        n = sum(len(q.output_ids) - 1 for q in qs)
        b4.append({"tok_s": n / qs[0].decode_seconds, "tokens_per_round": n / max(1, sum(q.spec_rounds for q in qs))})
    toks = sum(x["tokens"] for x in b1)
    return {"b1_tok_s": statistics.median(x["tok_s"] for x in b1), "b1_mean_tok_s": statistics.mean(x["tok_s"] for x in b1),
            "b1_tokens_per_round": toks / max(1, sum(x["rounds"] for x in b1)),
            "b1_relaxed": sum(x["relaxed"] for x in b1) / max(1, toks),
            "b4_tok_s": statistics.mean(x["tok_s"] for x in b4),
            "b4_tokens_per_round": statistics.mean(x["tokens_per_round"] for x in b4),
            "b1_prompts": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items()} for x in b1],
            "b4_batches": [{k: round(v, 3) for k, v in x.items()} for x in b4]}


def logprob_chunks(ex, model, ids: list[int], start: int, rows: int):
    """(first row, log-probabilities [n, V] FP32) of each continuation chunk of a cold prompt: the prompt minus its
    last token prefills in calls of PROMPT_ROWS, then calls of `rows` from row start - 1 (row t predicts ids[t + 1])
    under EMU. Calls of at most 16 rows keep every kernel reproducible: a wider NVFP4 MoE call runs b12x, whose
    results vary run to run (one prompt prefilled in one call scored three times: log-probabilities up to 1.5 apart;
    in 16-row calls: bitwise equal), which would put a KL of ~0.015 between two runs of the same configuration."""
    import torch
    from sharar.engine.sampler import SamplingParams
    ex.flush_cache()
    req = ex.admit(list(ids), SamplingParams(max_new_tokens=1, trace=False))
    req.cacheable = False
    try:
        if req.prefill_pos != 0:
            raise RuntimeError(f"expected a cold prefill, {req.prefill_pos} tokens were cached")
        slots = torch.tensor([req.slot], dtype=torch.int32, device="cuda")
        while req.prefill_pos < req.prefill_len:
            row = req.prefill_pos
            emulate = row >= start - 1
            n = rows if emulate else min(start - 1 - row, PROMPT_ROWS)
            batch = ex._prepare_single_prefill(req, n)
            EMU.update(on=emulate, slots=slots, positions=batch.context.positions)
            try:
                out = model.forward(batch.ids, batch.context)
            finally:
                EMU["on"] = False
            if emulate:
                logp = torch.log_softmax(model.logits(out["norm"]).float(), dim=-1)
                yield row, logp
                del logp
            ex._finish_single_prefill(batch, out)
    finally:
        req.cacheable = False
        ex.release(req)
        ex.flush_cache()


def base_rows(ex, model, seq: dict, rows: int) -> dict:
    import torch
    ids, lo = seq["ids"], seq["start"] - 1
    lps, ixs = [], []
    for row, logp in logprob_chunks(ex, model, ids, seq["start"], rows):
        a, b = max(lo, row), min(row + logp.shape[0], len(ids) - 1)
        if a < b:
            top = logp[a - row:b - row].topk(TOPK, dim=-1)
            lps.append(top.values)
            ixs.append(top.indices)
    return {"id": seq["id"], "lp": torch.cat(lps).half().cpu(), "ix": torch.cat(ixs).int().cpu()}


def score(ex, model, seqs: list[dict], base: list[dict], rows: int, anchor: list[dict] | None = None,
          keep_top: bool = False) -> dict:
    """Per position: top-1 agreement with the reference, truncated KL(reference || config), NLL of the reference token;
    with `anchor` (another configuration's top-32, e.g. the 3-bit stack's), KL(anchor || config) and top-1 against it
    too (what a knob adds on top of the anchor); `keep_top`: also this configuration's own top-32 (`tops`)."""
    import torch
    per, tot = [], {"n": 0, "kl": 0.0, "top1": 0.0, "nll": 0.0, "ref_nll": 0.0, "kl_a": 0.0, "top1_a": 0.0}
    domains: dict[str, list] = {}
    tops = []
    for si, (seq, ref) in enumerate(zip(seqs, base)):
        ids, lo = seq["ids"], seq["start"] - 1
        b_lp, b_ix = ref["lp"].cuda().float(), ref["ix"].cuda().long()
        if anchor is not None:
            a_lp, a_ix = anchor[si]["lp"].cuda().float(), anchor[si]["ix"].cuda().long()
        nxt = torch.tensor(ids[1:], device="cuda")
        kl = top1 = nll = kl_a = top1_a = 0.0
        lps, ixs = [], []
        for row, logp in logprob_chunks(ex, model, ids, seq["start"], rows):
            a, b = max(lo, row), min(row + logp.shape[0], len(ids) - 1)
            if a >= b:
                continue
            part, i, j = logp[a - row:b - row], a - lo, b - lo
            q = part.gather(1, b_ix[i:j])
            kl += float((b_lp[i:j].exp() * (b_lp[i:j] - q)).sum())
            top1 += float((part.gather(1, b_ix[i:j, :1])[:, 0] >= part.max(-1).values).sum())  # ties (BF16 logits) agree
            nll += float(-part.gather(1, nxt[a:b, None]).sum())
            if anchor is not None:
                kl_a += float((a_lp[i:j].exp() * (a_lp[i:j] - part.gather(1, a_ix[i:j]))).sum())
                top1_a += float((part.gather(1, a_ix[i:j, :1])[:, 0] >= part.max(-1).values).sum())
            if keep_top:
                top = part.topk(TOPK, dim=-1)
                lps.append(top.values.half().cpu())
                ixs.append(top.indices.int().cpu())
        if keep_top:
            tops.append({"id": seq["id"], "lp": torch.cat(lps), "ix": torch.cat(ixs)})
        tot["kl_a"] += kl_a
        tot["top1_a"] += top1_a
        n = len(ids) - 1 - lo
        ref_nll = float(-(b_lp * (b_ix == nxt[lo:, None])).sum())  # the reference's own NLL (its token is in its top 32)
        per.append({"id": seq["id"], "domain": seq["domain"], "n": n, "kl": kl / n, "top1": top1 / n, "nll": nll / n})
        domains.setdefault(seq["domain"], []).append((kl, n))
        for k, v in (("n", n), ("kl", kl), ("top1", top1), ("nll", nll), ("ref_nll", ref_nll)):
            tot[k] += v
    n = tot["n"]
    return {"kl": tot["kl"] / n, "top1": tot["top1"] / n, "nll": tot["nll"] / n, "ref_nll": tot["ref_nll"] / n,
            "positions": n, "rows": rows,
            "kl_by_domain": {d: sum(k for k, _ in v) / sum(m for _, m in v) for d, v in domains.items()},
            "sequences": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in p.items()} for p in per],
            **({"kl_vs_anchor": tot["kl_a"] / n, "top1_vs_anchor": tot["top1_a"] / n} if anchor is not None else {}),
            "tops": tops}


def run_gsm(ex, r: dict, tok, ids: list[list[int]], items: list[dict], max_tokens: int) -> dict:
    out = []
    for g in range(0, len(ids), 4):
        ex.flush_cache()
        qs = ex.generate_batch([(p, params(r, max_tokens)) for p in ids[g:g + 4]])
        for it, q in zip(items[g:g + 4], qs):
            text = tok.decode(q.output_ids)
            pred = gsm_pred(text)
            out.append({"correct": same_number(pred, it["gold"]), "pred": pred, "gold": it["gold"],
                        "tokens": len(q.output_ids), "output_ids": list(q.output_ids)})
    return {"correct": sum(x["correct"] for x in out), "n": len(out), "items": out}


def compare_gsm(g: dict, ref: dict | None) -> dict:
    res = {"gsm_correct": g["correct"], "gsm_n": g["n"], "gsm_acc": g["correct"] / g["n"]}
    if ref:
        a, b = ref["items"], g["items"]
        res.update(gsm_wins=sum(1 for x, y in zip(a, b) if y["correct"] and not x["correct"]),
                   gsm_losses=sum(1 for x, y in zip(a, b) if x["correct"] and not y["correct"]),
                   gsm_identical=sum(1 for x, y in zip(a, b) if x["output_ids"] == y["output_ids"]) / len(a),
                   gsm_ref_correct=ref["correct"])
    return res


def shape_floor(a: list[dict], b: list[dict]) -> dict:
    """KL(a || b) and top-1 agreement between two references' top-32 (b's log-probability of a token outside its top
    32 taken as its 32nd): the kernel-shape noise between two call shapes of the exact profile."""
    import torch
    kl = top1 = n = 0.0
    for x, y in zip(a, b):
        alp, aix, blp, bix = x["lp"].float(), x["ix"].long(), y["lp"].float(), y["ix"].long()
        hit = aix[:, :, None] == bix[:, None, :]
        q = torch.where(hit.any(-1), (blp[:, None, :] * hit).sum(-1), blp[:, -1:])
        kl += float((alp.exp() * (alp - q)).sum())
        top1 += float((aix[:, 0] == bix[:, 0]).sum())
        n += aix.shape[0]
    return {"kl": kl / n, "top1": top1 / n}


def reference_complete(ref: Path, rows: list[int]) -> bool:
    return (ref / "meta.json").exists() and all((ref / f"base-r{r}.pt").exists() for r in rows)


def build_reference(model, tok, tmpl, ref: Path, w: dict, args, r: dict) -> None:
    """What the reference lacks, with the exact profile: its greedy continuations (sequences.json), their top-32
    log-probabilities at each call shape a configuration scores at (base-r<rows>.pt, --ref-rows: every configuration
    is compared with exact computed at its own call shape, so the kernel-shape noise between, e.g., 4-row and 6-row
    calls is not counted as its loss), the shape noise itself (floor.json) and exact's GSM8K-50 answers (gsm.json)."""
    import torch
    ref.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ex, info = build_executor(model, r)
    if (ref / "sequences.json").exists():
        seqs = json.loads((ref / "sequences.json").read_text())
    else:
        prompts = [chat_ids(tok, tmpl, p["prompt"]) for p in w["kl"]]
        seqs = []
        for g in range(0, len(prompts), 4):
            ex.flush_cache()
            qs = ex.generate_batch([(p, params(r, args.kl_tokens)) for p in prompts[g:g + 4]])
            for p, meta, q in zip(prompts[g:g + 4], w["kl"][g:g + 4], qs):
                seqs.append({"id": meta["id"], "domain": meta["domain"], "start": len(p), "ids": p + list(q.output_ids)})
        (ref / "sequences.json").write_text(json.dumps(seqs) + "\n")
        log(f"reference: {len(seqs)} continuations ({time.time() - t0:.0f}s)")
    rows_list = sorted({info["depth_b1"] + 1, *args.ref_rows})
    for rows in rows_list:
        if not (ref / f"base-r{rows}.pt").exists():
            torch.save([base_rows(ex, model, s, rows) for s in seqs], ref / f"base-r{rows}.pt")
            log(f"reference: top-32 at {rows}-row calls ({time.time() - t0:.0f}s)")
    if not (ref / "gsm.json").exists():
        gsm = run_gsm(ex, r, tok, [chat_ids(tok, tmpl, it["question"] + GSM_SUFFIX) for it in w["gsm"]], w["gsm"],
                      args.gsm_max_tokens)
        (ref / "gsm.json").write_text(json.dumps(gsm) + "\n")
        log(f"reference: GSM8K-{gsm['n']} {gsm['correct']} ({time.time() - t0:.0f}s)")
    own = info["depth_b1"] + 1
    bases = {k: torch.load(ref / f"base-r{k}.pt") for k in rows_list}
    floor = {"own_rows": own, "by_rows": {str(k): shape_floor(bases[own], bases[k]) for k in rows_list if k != own}}
    (ref / "floor.json").write_text(json.dumps(floor) + "\n")
    log("reference: shape noise against " + ", ".join(f"{k} rows KL {v['kl']:.4f} top-1 {v['top1']:.2%}"
                                                       for k, v in floor["by_rows"].items()))
    meta = {"settings": r["settings"], "settings_hash": settings_hash(r), "rows": own, "ref_rows": rows_list,
            "commit": git_head(), "host": socket.gethostname(), "seconds": round(time.time() - t0, 1),
            "created": dt.datetime.now().isoformat()}
    (ref / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    del ex, bases
    gc.collect()
    torch.cuda.empty_cache()


def git_head() -> str:
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no", "sharar"],
                           capture_output=True, text=True).stdout.strip()
    return r.stdout.strip() + ("+dirty" if dirty else "")


def worker(job_path: Path) -> int:
    """One weight set: load once, then the reference (when asked) and each configuration while the window lasts."""
    import torch
    from sharar import target as targets
    from sharar.api.chat_template import ChatTemplate
    from sharar.engine import config
    from sharar.tokenizer import SnapshotTokenizer
    import ctypes
    import signal
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG: never outlive the driver that holds the lock
    if os.getppid() == 1:
        return 1
    job = json.loads(job_path.read_text())
    args = SimpleNamespace(**job["args"])
    w, deadline, est = job["workload"], job["deadline"], job["estimate_s"]
    out, progress = Path(job["results"]), Path(job["progress"])
    ref = Path(job["reference"])
    first = job["reference_config"] if job["build_reference"] else job["configs"][0]["resolved"]
    config.install(engine_config(first))
    t = targets.get("nemotron")
    t0 = time.time()
    model = targets.load_model(t, t.snapshot_dir(), log=None)
    load_s = time.time() - t0
    log(f"weight set loaded in {load_s:.0f}s ({torch.cuda.memory_allocated() / 2**30:.1f} GiB)")
    install_emulation()
    tok, tmpl = SnapshotTokenizer(t.snapshot_dir()), ChatTemplate(t.snapshot_dir())
    if job["build_reference"]:
        if time.time() + job["reference_estimate_s"] > deadline:
            log("window too short for the reference: yielding")
            return 3
        build_reference(model, tok, tmpl, ref, w, args, job["reference_config"])
    seqs = json.loads((ref / "sequences.json").read_text())
    bases = {int(p.stem.split("-r")[1]): p for p in ref.glob("base-r*.pt")}
    ref_gsm = json.loads((ref / "gsm.json").read_text())
    ref_meta = json.loads((ref / "meta.json").read_text())
    ids = {"b1": [chat_ids(tok, tmpl, p) for p in w["b1"]], "b4": [chat_ids(tok, tmpl, p) for p in w["b4"]]}
    gsm_ids = [chat_ids(tok, tmpl, it["question"] + GSM_SUFFIX) for it in w["gsm"]]
    parts = set(args.parts.split(","))
    for c in job["configs"]:
        if time.time() + est > deadline:
            log(f"window ends before {c['name']} (estimate {est:.0f}s): yielding")
            return 3
        r = c["resolved"]
        with progress.open("a") as f:
            f.write(json.dumps({"started": c["name"], "hash": settings_hash(r)}) + "\n")
        log(f"{c['name']}: {json.dumps(r['settings'])}")
        stages, row = {}, {}
        t1 = time.time()
        ex, info = build_executor(model, r)
        stages["executor"] = time.time() - t1
        if "speed" in parts:
            t1 = time.time()
            row.update(measure_speed(ex, r, ids, args.tokens))
            stages["speed"] = time.time() - t1
            log(f"{c['name']}: b1 {row['b1_tok_s']:.1f} tok/s ({row['b1_tokens_per_round']:.2f} tok/round), "
                f"b4 {row['b4_tok_s']:.1f} ({stages['speed']:.0f}s)")
        if "kl" in parts:
            t1 = time.time()
            rows = info["depth_b1"] + 1
            ref_rows = min(bases, key=lambda k: (abs(k - rows), k))  # exact at this call shape (--ref-rows)
            # the anchor's own top-32 (saved when it ran in this run, at its call shape): what a knob adds on top of it
            is_anchor = c["name"] == getattr(args, "kl_anchor", None)
            anchors = {int(p.stem.split("-r")[1]): p for p in out.parent.glob("anchor-r*.pt")} if not is_anchor else {}
            a_rows = min(anchors, key=lambda k: (abs(k - rows), k)) if anchors else None
            s = score(ex, model, seqs, torch.load(bases[ref_rows]), rows,
                      anchor=torch.load(anchors[a_rows]) if a_rows else None, keep_top=is_anchor)
            if is_anchor:
                torch.save(s["tops"], out.parent / f"anchor-r{rows}.pt")
            row.update(kl=s["kl"], top1=s["top1"], nll=s["nll"], ref_nll=s["ref_nll"], kl_rows=rows, kl_ref_rows=ref_rows,
                       kl_by_domain=s["kl_by_domain"], kl_sequences=s["sequences"])
            if a_rows:
                row.update(kl_vs_anchor=s["kl_vs_anchor"], top1_vs_anchor=s["top1_vs_anchor"], kl_anchor_rows=a_rows)
            stages["kl"] = time.time() - t1
            log(f"{c['name']}: KL {s['kl']:.5f} top-1 {s['top1']:.4f} NLL {s['nll']:.4f} ({stages['kl']:.0f}s)")
        if "gsm" in parts:
            t1 = time.time()
            same = settings_hash(r) == ref_meta["settings_hash"]
            g = ref_gsm if same else run_gsm(ex, r, tok, gsm_ids, w["gsm"], args.gsm_max_tokens)
            row.update(compare_gsm(g, ref_gsm), gsm_reused_reference=same,
                       gsm_tokens=sum(x["tokens"] for x in g["items"]))
            stages["gsm"] = time.time() - t1
            log(f"{c['name']}: GSM8K-{g['n']} {g['correct']} (reference {ref_gsm['correct']}, "
                f"+{row['gsm_wins']} -{row['gsm_losses']}) ({stages['gsm']:.0f}s)")
        del ex
        gc.collect()
        torch.cuda.empty_cache()
        total = sum(stages.values())
        est = max(est * 0.5 + total * 0.5, total)
        result = {"engine": "Sharar", "machine": socket.gethostname(), "model": "nemotron-3.5-super", "name": c["name"],
                  "hash": settings_hash(r), "profile": r["profile"], "overrides": r["overrides"], "settings": r["settings"],
                  "spec": r["spec"], "accept": r["accept"], **info, "parts": sorted(parts), **row,
                  "stages_s": {k: round(v, 1) for k, v in stages.items()}, "config_s": round(total, 1),
                  "load_s": round(load_s, 1), "commit": git_head(), "reference": ref.name,
                  "finished": dt.datetime.now().isoformat()}
        with out.open("a") as f:
            f.write(json.dumps(result) + "\n")
        log(f"{c['name']}: done in {total:.0f}s")
    return 0


# ---------------------------------------------------------------- driver: lock, windows, table

class GpuLock:
    def __init__(self, path: Path | None):
        self.path, self.f, self.since = path, None, 0.0

    def acquire(self) -> None:
        self.since = time.time()
        if self.path is None:
            return
        self.f = open(self.path, "a")
        log(f"waiting for the GPU lock {self.path}")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        while available_gib() < 80:
            log(f"{available_gib():.0f} GiB available (another model process?); waiting")
            time.sleep(30)
        self.since = time.time()
        log("GPU lock held")

    def release(self) -> None:
        if self.f is not None:
            fcntl.flock(self.f, fcntl.LOCK_UN)
            self.f.close()
            self.f = None
            log("GPU lock released")


def available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def fmt(v, spec: str, none: str = "-") -> str:
    return none if v is None else format(v, spec)


def table(rows: list[dict], order: list[str], anchor: str | None, margin: float, floor: dict | None) -> str:
    by = {r["name"]: r for r in rows}
    names = [n for n in order if n in by] + [n for n in by if n not in order]
    if not names:
        return "(no finished configurations)"
    base = by[names[0]]
    anchor_kl = by[anchor]["kl"] if anchor in by and "kl" in by[anchor] else None
    cal = CALIBRATION
    head = ["config", "b1 tok/s", "b1 h2h~", "b4 tok/s", "b4 h2h~", "tok/round b1", "relaxed", "KL", "top-1", "NLL",
            "KL vs anchor",
            "GSM8K", "W/L", "same", "d b1", "d b4", "flags"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for n in names:
        r = by[n]
        flags = []
        if anchor_kl is not None and (r["kl_vs_anchor"] > anchor_kl * margin if r.get("kl_vs_anchor") is not None
                                      else r.get("kl") is not None and r["kl"] > anchor_kl * (1 + margin)):
            flags.append("KL!")
        if r.get("gsm_losses", 0) - r.get("gsm_wins", 0) >= 3:
            flags.append("GSM!")
        if r.get("error"):
            flags.append("ERROR")
        d = lambda k: (f"{100 * (r[k] / base[k] - 1):+.1f}%" if r.get(k) and base.get(k) else "-")  # noqa: E731
        lines.append("| " + " | ".join([
            n, fmt(r.get("b1_tok_s"), ".1f"), fmt(r["b1_tok_s"] / cal["b1"] if cal["b1"] and r.get("b1_tok_s") else None, ".1f"),
            fmt(r.get("b4_tok_s"), ".1f"), fmt(r["b4_tok_s"] / cal["b4"] if cal["b4"] and r.get("b4_tok_s") else None, ".1f"),
            fmt(r.get("b1_tokens_per_round"), ".2f"), fmt(r.get("b1_relaxed"), ".1%"),
            fmt(r.get("kl"), ".4f"), fmt(r.get("top1"), ".2%"), fmt(r.get("nll"), ".4f"), fmt(r.get("kl_vs_anchor"), ".4f"),
            f"{r['gsm_correct']}/{r['gsm_n']}" if "gsm_correct" in r else "-",
            f"+{r['gsm_wins']}/-{r['gsm_losses']}" if "gsm_wins" in r else "-", fmt(r.get("gsm_identical"), ".0%"),
            d("b1_tok_s"), d("b4_tok_s"), " ".join(flags)]) + " |")
    notes = []
    if floor:
        notes.append(f"Kernel-shape noise (exact at {floor['own_rows']}-row calls against exact at others, not counted in KL, "
                     "which compares each configuration with exact at its own call shape): "
                     + ", ".join(f"{k} rows KL {v['kl']:.4f}" for k, v in floor["by_rows"].items()) + ".")
    if anchor_kl is not None:
        notes.append(f"KL! = KL vs anchor above {margin:.0%} of {anchor}'s own KL against exact ({anchor_kl:.4f} x {margin:g} = "
                     f"{anchor_kl * margin:.4f}: a stacked knob may add at most that share of what the anchor costs), or "
                     f"without it a KL more than {margin:.0%} above {anchor}'s; GSM! = at least 3 more GSM8K losses than "
                     "wins against the exact reference (W/L; same = answers identical to it).")
    for n in names:
        r = by[n]
        if n in H2H and r.get("b1_tok_s"):
            notes.append(f"{n}: proxy / h2h = {r['b1_tok_s'] / H2H[n][0]:.3f} (b1), "
                         f"{r['b4_tok_s'] / H2H[n][1]:.3f} (b4) against h2h {H2H[n][0]} / {H2H[n][1]}.")
    machines = sorted({r.get("machine", "?") for r in rows})
    secs = [r["config_s"] for r in rows if r.get("config_s")]
    notes.append(f"Sharar, {', '.join(machines)}; deltas against {names[0]}; "
                 + (f"{statistics.median(secs) / 60:.1f} min a configuration (median)" if secs else ""))
    return "\n".join(lines) + "\n\n" + "\n".join(f"- {x}" for x in notes) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--name", help="the run's directory under --out-root (a rerun resumes it)")
    ap.add_argument("--config", action="append", default=[], help="NAME=BASE[:KEY=VALUE,...] or a built-in name (repeatable)")
    ap.add_argument("--configs", type=Path, help='a JSON list of {"name", "base", "env": {KEY: VALUE}}')
    ap.add_argument("--parts", default="speed,kl,gsm")
    ap.add_argument("--tokens", type=int, default=384, help="speed: output tokens per request")
    ap.add_argument("--kl-tokens", type=int, default=256)
    ap.add_argument("--gsm", type=int, default=50)
    ap.add_argument("--gsm-max-tokens", type=int, default=512)
    ap.add_argument("--kl-anchor", default="fast+w3")
    ap.add_argument("--kl-margin", type=float, default=0.25)
    ap.add_argument("--hold-minutes", type=float, default=25.0)
    ap.add_argument("--yield-seconds", type=float, default=90.0)
    ap.add_argument("--lock", default="auto", help="the GPU lock file; auto: $SHARAR_RESULTS/.gpu.lock; none: no lock")
    ap.add_argument("--estimate-minutes", type=float, default=6.0, help="a configuration's time, until one is measured")
    ap.add_argument("--load-minutes", type=float, default=2.0)
    ap.add_argument("--ref-rows", type=lambda t: [int(x) for x in t.split(",")], default=[4, 5, 6, 7],
                    help="call shapes (depth + 1) the reference is scored at: each configuration is compared with exact "
                         "at its own batch-1 call shape, else the nearest")
    ap.add_argument("--rebuild-reference", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--table", action="store_true", help="print the table of what has finished and exit")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--h2h-workload", type=Path, default=H2H_WORKLOAD)
    args = ap.parse_args()
    if args.worker:
        return worker(args.worker)
    if not args.name:
        ap.error("--name is required")
    run = args.out_root / args.name
    run.mkdir(parents=True, exist_ok=True)
    results, progress = run / "results.jsonl", run / "progress.jsonl"

    known = {k: v for k, v in BUILTIN.items()}
    specs = []
    for text in args.config:
        name, profile, env = parse_config(text, known)
        known[name] = (profile, env)
        specs.append((name, profile, env))
    if args.configs:
        for item in json.loads(args.configs.read_text()):
            env = {(k if k.startswith("SHARAR_") else f"SHARAR_{k}"): str(v) for k, v in item.get("env", {}).items()}
            profile, env = combine(item.get("base", "fast"), env, known)
            known[item["name"]] = (profile, env)
            specs.append((item["name"], profile, env))
    plan_path = run / "plan.json"
    if not specs and plan_path.exists():
        specs = [tuple(x) for x in json.loads(plan_path.read_text())["configs"]]
    if not specs:
        ap.error("no configurations (--config / --configs)")
    order = [s[0] for s in specs]
    if len(set(order)) != len(order):
        ap.error("configuration names must be unique")
    configs = [{"name": n, "resolved": resolve(p, e)} for n, p, e in specs]
    plan_path.write_text(json.dumps({"configs": specs, "args": {k: str(v) for k, v in vars(args).items()}}, indent=1) + "\n")

    w = workload(args)
    ref = reference_dir(args, w)
    floor = json.loads((ref / "floor.json").read_text()) if (ref / "floor.json").exists() else None
    if args.table:
        print(table(read_rows(results), order, args.kl_anchor, args.kl_margin, floor))
        return 0
    if args.rebuild_reference and ref.exists():
        for p in ref.iterdir():
            p.unlink()
    ref_resolved = resolve("exact", {})
    lock_path = None if args.lock == "none" else (
        Path(args.lock) if args.lock != "auto" else __import__("sharar.launch.settings", fromlist=["resolve"]).resolve("nemotron").results / ".gpu.lock")
    lock = GpuLock(lock_path)
    estimate = args.estimate_minutes * 60
    hold = args.hold_minutes * 60
    worker_args = {k: getattr(args, k) for k in ("parts", "tokens", "kl_tokens", "gsm", "gsm_max_tokens", "ref_rows",
                                                 "kl_anchor")}

    if hold < (args.load_minutes + args.estimate_minutes) * 60:
        ap.error("--hold-minutes must cover a load and a configuration")
    retry = {(r["name"], r["hash"]) for r in read_rows(results) if r.get("error")} if args.retry_failed else set()

    def pending() -> list[dict]:
        rows = read_rows(results)
        done = {(r["name"], r["hash"]) for r in rows if not r.get("error")}
        failed = {(r["name"], r["hash"]) for r in rows if r.get("error")} - retry
        return [c for c in configs if (c["name"], settings_hash(c["resolved"])) not in done | failed]

    while True:
        todo = pending()
        need_ref = not reference_complete(ref, args.ref_rows)
        if not todo and not need_ref:
            break
        groups: dict[str, list] = {}
        for c in todo:
            groups.setdefault(weight_key(c["resolved"]), []).append(c)
        ref_key = weight_key(ref_resolved)
        keys = list(groups)
        if need_ref:
            keys = [ref_key] + [k for k in keys if k != ref_key]
        lock.acquire()
        try:
            for key in keys:
                left = lock.since + hold - time.time()
                if left < args.load_minutes * 60 + estimate:
                    break
                group = groups.get(key, [])
                build = need_ref and key == ref_key
                job = {"args": worker_args, "workload": w, "deadline": lock.since + hold, "estimate_s": estimate,
                       "results": str(results), "progress": str(progress), "reference": str(ref),
                       "build_reference": build, "reference_config": ref_resolved, "reference_estimate_s": 12 * 60,
                       "configs": group}
                job_path = run / "job.json"
                job_path.write_text(json.dumps(job))
                logfile = run / "worker.log"
                log(f"worker: {len(group)} configuration(s){' + the reference' if build else ''} on weight set {key}")
                with logfile.open("a") as f:
                    f.write(f"\n==== {dt.datetime.now().isoformat()} {[c['name'] for c in group]}\n")
                    f.flush()
                    # the worker inherits the lock's descriptor: the GPU stays locked while it lives
                    proc = subprocess.Popen([sys.executable, "-u", __file__, "--worker", str(job_path)], cwd=REPO,
                                            env={**os.environ, "PYTHONPATH": str(REPO)}, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True,
                                            pass_fds=(lock.f.fileno(),) if lock.f is not None else ())
                    for line in proc.stdout:
                        f.write(line)
                        f.flush()
                        if line.startswith("["):
                            print(line, end="", flush=True)
                    rc = proc.wait()
                need_ref = not reference_complete(ref, args.ref_rows)
                done_rows = [r for r in read_rows(results) if r.get("config_s")]
                if done_rows:
                    estimate = max(statistics.median(r["config_s"] for r in done_rows), 120.0)
                if rc == 3:
                    break
                if rc != 0:
                    started = [p for p in read_rows(progress)]
                    finished = {(r["name"], r["hash"]) for r in read_rows(results)}
                    last = next((p for p in reversed(started) if (p["started"], p["hash"]) not in finished), None)
                    log(f"worker exited {rc} (see {logfile})")
                    if last is not None:
                        with results.open("a") as f:
                            f.write(json.dumps({"name": last["started"], "hash": last["hash"], "error": f"worker exit {rc}",
                                                "engine": "Sharar", "machine": socket.gethostname()}) + "\n")
                    elif build:
                        raise SystemExit(f"the reference failed: see {logfile}")
        finally:
            lock.release()
        if pending() or not reference_complete(ref, args.ref_rows):
            log(f"yielding the GPU for {args.yield_seconds:.0f}s")
            time.sleep(args.yield_seconds)
    rows = read_rows(results)
    floor = json.loads((ref / "floor.json").read_text()) if (ref / "floor.json").exists() else None
    text = table(rows, order, args.kl_anchor, args.kl_margin, floor)
    (run / "table.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
