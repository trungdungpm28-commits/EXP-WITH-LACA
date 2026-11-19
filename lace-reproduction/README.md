# LACE reduced reproduction

This project replays Azure Functions Trace 2019 to reproduce the structure and equations of Figures 8 and 9 in the LACE paper. The measured campaign uses LACE, FaasCache GDSF, 10-minute TTL and LRU. See [method mapping](docs/method_mapping.md) for source pins, deviations, timing boundaries, and the latency proxy.

The [larger diagnostic campaign](docs/diagnostics_method.md) covers full-day workload cohorts, seven-way cache ablation, all eight IAT predictors, four feature ablations, and nine-policy CPU and memory measurements. Its outputs live in `results/azure2019-diagnostics/`.

## Setup and rerun

Use Debian/Linux and Python 3.13. Run the commands inside this directory. The Microsoft archive is found automatically in the adjacent lab data directory when present. Pass `--archive /absolute/path/to/archive.tar.xz` to override discovery. Without a matching local file, the parser attempts a bounded resumable download and verifies Microsoft's SHA-256.

```bash
# Start the six-hour clock before dependency setup.
deadline=$(python3 -c 'import time; print(time.monotonic() + 21600)')
setup_seconds=$(python3 -c 'import sys,time; print(max(1, int(float(sys.argv[1])-1800-time.monotonic())))' "$deadline")
timeout --kill-after=5s "${setup_seconds}s" bash -eu -c '
  python3 -m venv .venv
  .venv/bin/python -m pip install --timeout 20 --retries 2 --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
'

# Smoke tests on a small subset and pilot. Choose a fresh output directory.
.venv/bin/python campaign.py --deadline "$deadline" --smoke --output results/smoke-rerun

# Full run, including its own pilot. Pass --reparse for a fresh parse.
.venv/bin/python campaign.py --deadline "$deadline" --output results/full-rerun

# Longer-window campaign, with its own pilot and a frozen historical activity limit.
.venv/bin/python campaign.py --deadline "$deadline" --expanded --output results/expanded-rerun

# Re-export and audit the saved longer-window run.
MPLCONFIGDIR=/tmp/lace-matplotlib .venv/bin/python export.py results/azure2019-expanded
.venv/bin/python audit.py results/azure2019-expanded
```

Use the smoke or the full command as needed. A full run also runs the toy and smoke checks. With `--deadline` omitted, the campaign starts its own six-hour clock after setup. The deadline is an absolute Python monotonic timestamp on the local boot. Subprocesses are bounded by their stage time and the heavy-work cutoff 30 minutes before the deadline. Progress and resource samples are recorded in `run_evidence.csv`. Run into a fresh directory to preserve previous evidence.

## Outputs

- `results/azure2019-expanded/` is the primary longer-window run. `results/azure2019-reduced/` holds the initial comparison. Both contain plotted CSVs and PNGs, matched replay inputs, frozen function and budget manifests, validation and evidence. Each `pilot/` holds separate pilot inputs and measurements. Measured metrics are derived from `runs.csv`.
- `warm_reference_ms` sums trace average execution times. `policy_e2e_ms` adds the historical maximum-minus-average estimated penalty at each cold invocation. Figure 8 is `100 * (policy_e2e_ms - warm_reference_ms) / warm_reference_ms`. Figure 9 is `100 * cold_invocations / request_count`.
- The replay stores timestamps in seconds, latency in milliseconds, and footprint in decimal MB. Diagnostics report CPU time and process RSS on the actual host. Figures identify the reduced scale and use decimal GB for simulated capacity.
