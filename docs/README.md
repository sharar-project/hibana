# Documentation

Use these pages for the current implementation. Dated measurements and old
milestones live in the archive; their deployment and next-step statements apply
to the source recorded in each report.

| Guide | Contents |
|---|---|
| [Operations](operations.md) | Build, launch, defaults, memory guard and development |
| [API](api.md) | Supported endpoints, request controls and streaming behavior |
| [Interfaces](interfaces.md) | Execution, proposer, cache and graph-buffer contracts |
| [Ownership invariants](phase7a.md) | Termination, cancellation, flush, retraction and failure |
| [Performance](performance.md) | Current BS1/BS4 and resumed-session evidence |
| [Optimization inventory](optimizations.md) | Every optimization in the engine: location, default, measured effect, evidence |
| [Benchmark guide](../bench/README.md) | Commands, reproducibility and shared helper layout |
| [Contributing](../CONTRIBUTING.md) | Checks and implementation boundaries |
| [Roadmap](../roadmap.md) | Current priorities and remaining engineering work |
| [Draft adaptation](draft-adaptation.md) | Training the DFlash2 drafter on recorded traffic: `sharar-ctl train`, traces, promotion, design and limits |
| [W3 calibration](w3-calibration.md) | Fit, validate and load activation-aware three-bit target decode copies |
| [Qwen3.8-Flash-Next](flashnext.md) | The second target: package layout, checkpoint conventions, parity and decode numbers, what is not ported |
| [Cache-policy proposal](request-cache-policy.md) | Unimplemented request-level cache controls |
| [Archive and decisions](archive/README.md) | Historical studies, manifests and rejected experiments |

Write supported behavior in its current guide. Put measured results in dated
reports with source/protocol identities and compact evidence manifests. Keep
one current work order in the roadmap. Do not copy whole experiment narratives
into README or present historical next steps as new priorities.

Run `python3 tools/check_evidence.py` after moving documentation. It discovers
all JSON evidence manifests recursively and checks local Markdown file links;
it does not validate external URLs or heading fragments.
