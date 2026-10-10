# Documentation

Use these pages for the current implementation. Dated studies and old
milestones were removed from the tree on 2026-09-26; they are at the tag
`archive/pre-cleanup-2026-09-26` (linked below), and their deployment and
next-step statements apply to the source recorded in each report.

| Guide | Contents |
|---|---|
| [Operations](operations.md) | Build, launch, defaults, memory guard and development |
| [API](api.md) | Supported endpoints, request controls and streaming behavior |
| [Architecture](architecture.md) | Current engine shape, capability matrix for both targets, proposed seams and migration order (#24) |
| [Interfaces](interfaces.md) | Execution, proposer, cache and graph-buffer contracts |
| [Ownership invariants](ownership.md) | Termination, cancellation, flush, retraction and failure |
| [Performance](performance.md) | Current BS1/BS4 and resumed-session evidence |
| [Optimization inventory](optimizations.md) | Every optimization in the engine: location, default, measured effect, evidence |
| [Evaluation](evaluation.md) | Evaluating a build, judging a change, the lossy gate, nightly and weekly GB10 checks |
| [Profiling](../profiling/README.md) | In-process profilers, their frozen inputs, draft adaptation tools |
| [Contributing](../CONTRIBUTING.md) | Checks and implementation boundaries |
| [Roadmap](../roadmap.md) | Current priorities and remaining engineering work |
| [Launch plan](launch-plan.md) | Install, harness, drafter-pack and launch work that stands between the engine and its users |
| [Draft adaptation](draft-adaptation.md) | Training the DFlash2 drafter on recorded traffic: `sharar-ctl train`, traces, promotion, design and limits |
| [W3 calibration](w3-calibration.md) | Fit, validate and load activation-aware three-bit target decode copies |
| [27B or Flash-Next](targets.md) | The two targets compared: a long Claude Code session, speed, long context, quality, footprint (#11) |
| [Qwen3.8-Flash-Next](flashnext.md) | The second target: package layout, checkpoint conventions, parity and decode numbers, what is not ported |
| [Cache-policy proposal](request-cache-policy.md) | Unimplemented request-level cache controls |
| [Archive and decisions](https://github.com/abdelfattah-lab/sharar/blob/archive/pre-cleanup-2026-09-26/docs/archive/README.md) | Historical studies, manifests and rejected experiments (pinned tag, not in the tree) |

Write supported behavior in its current guide. Put measured results in dated
reports with source/protocol identities and compact evidence manifests. Keep
one current work order in the roadmap. Do not copy whole experiment narratives
into README or present historical next steps as new priorities.

Run `python3 tools/check_evidence.py` after moving documentation. It discovers
all JSON evidence manifests recursively and checks local Markdown file links;
it does not validate external URLs or heading fragments.
