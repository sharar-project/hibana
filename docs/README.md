# Documentation

One page per topic, describing the code as it is. Measurements and plans live in GitHub issues and in each
change's pull request; history is in git (older studies are at the tag `archive/pre-cleanup-2026-09-26`).

| Page | Contents |
|---|---|
| [Operations](operations.md) | Prerequisites, launcher settings, serving profiles, traces, cold tier and session store |
| [API](api.md) | Endpoints, request controls, tool calls, constrained output, Anthropic Messages |
| [Architecture](architecture.md) | Package layout, the seams both targets share, capability matrix, what lives in which repository |
| [Interfaces](interfaces.md) | Request ownership, proposer lifecycle, feature lifetimes, device-resident acceptance, failure handling |
| [Flash-Next](flashnext.md) | The second target: model, package, checkpoint conventions, sparse attention, its MTP draft |
| [Tensor parallel](tensor-parallel.md) | Two Sparks: setup, sharding, the RDMA all-reduce, limits |
| [3-bit calibration](w3-calibration.md) | Fitting and serving the calibrated 3-bit decode copies |
| [Evaluation](evaluation.md) | Measuring a build or a change, the lossy-change gate, nightly and weekly GB10 checks |
| [Profiling](../profiling/README.md) | In-process profilers |
| [Contributing](../CONTRIBUTING.md) | Checks and the rules for each kind of change |

When a change alters behavior, update its page in the same pull request. Run `python3 tools/check_links.py`
after moving or deleting a page.
