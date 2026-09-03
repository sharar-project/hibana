# GB10 text-serving baseline

`benchmark_text_server.py` captures a reproducible, text-only serving baseline
for a single DGX Spark/GB10. It records the host, container, model and repository
configuration before running the matrix.

The default controlled matrix is:

- Input lengths: 128, 2,048 and 8,192 exact token IDs.
- Output length: 256 tokens with EOS ignored.
- Concurrency: 1 and 4.
- One warmup and five measured batches per cell.
- Prefix-cache flush before every batch.
- `MemAvailable` sampled four times per second and GPU power sampled once per
  second.
- Thinking disabled and no multimodal inputs.

Run it from the DGX Spark host while the reference SGLang container is listening only
on `127.0.0.1:30000`:

```bash
python3 bench/benchmark_text_server.py
```

Each run creates an immutable timestamped directory under
`${SHARAR_RESULTS:-${XDG_STATE_HOME:-$HOME/.local/state}/sharar/gb10}/controlled`.
`batches.jsonl` contains the raw request and resource samples,
`summary.csv` contains the measured rows,
`manifest.json` captures provenance, and `quality.json` contains deterministic
text canaries. Use a distinct `--label` for every controlled variant; the
launcher and benchmark defaults are pinned to the 27B NVFP4 GB10 engine.

Use shorter settings only for validating the harness itself:

```bash
python3 bench/benchmark_text_server.py \
  --input-lengths 32 \
  --output-length 8 \
  --concurrency 1 \
  --warmups 0 \
  --repetitions 1 \
  --skip-quality
```

Compare every candidate against the frozen production baseline. The comparator
rejects request or quality failures, serial output drift, more than 3%
throughput regression, more than 15% TTFT regression, or less than 16 GiB
available memory. Concurrent hash drift is reported but is diagnostic because
batch scheduling is not bitwise deterministic:

```bash
python3 bench/compare_runs.py BASELINE_RUN_DIR CANDIDATE_RUN_DIR
```
