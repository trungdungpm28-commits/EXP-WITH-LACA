# Larger reproduction with full diagnostics

This campaign uses the verified public Azure Functions Trace 2019 release and the same day 1–3 fit / day 4 holdout as the previous reproduction. Day 4 events are reconstructed deterministically from the measured minute counts. No synthetic repetitions or multiplied samples are used. The workload manifest uses the original seeded historical frequency cohorts with the previous activity cap removed. Every tested policy in a matched comparison shares the frozen trace, prehistory convention, capacity and initial empty cache.

| Trace | Selected functions | Observed functions | Full day 4 events |
|---|---:|---:|---:|
| Representative | 400 | 331 | 9,295,955 |
| Rare | 1,000 | 339 | 5,967 |
| Random | 200 | 161 | 462,526 |

Rare stays at its full-day count because the original 1,000 least-active cohort is retained. The extremely active functions in Representative account for much of the growth. The primary replay uses logical budgets of 1, 4, 10, 22 and 40 GB, four policies, and a common estimated startup cost. Resource diagnostics add five policies for the full nine shown in the reference images. The full Representative active footprint is 20.774 GB, so 22 GB can hold every active function. Figure 11 includes the requested 22 GB configuration. A 10 GB supplementary comparison exercises eviction under memory pressure on the same event stream.

## Feature and model comparisons

The full LACE model has six width-96 experts, top-three routing, and 24 offline CPU epochs. The model keeps the architecture and four feature families described in [method mapping](method_mapping.md). The five ablations retrain from the same seed, chronological data split, optimizer schedule and training budget. Four ablations physically remove their corresponding feature columns. The MoE variant removes the auxiliary classification task. Models use the epoch selected by validation MAE.
 
Table 3 compares full LACE, MoE without the auxiliary classifier, MLP, LinearSVR, XGBoost, ordinary least squares, LightGBM and gradient boosting (GBM). Table 4 compares full LACE against the four feature ablations. Every row uses the same deterministic sample of 300,000 Representative requests with realized in-window day-4 IAT labels. IAT error metrics are in seconds, with MSE in seconds squared. MAPE ignores zero IAT targets. Model hyperparameters are explicit in `diagnostics_models.py` and fitted configurations are logged. The baselines use the same chronological training pool and deterministic seed. Their estimators use the local documented configurations because the paper does not give exact baseline hyperparameters.

## Additional cache policies

| Policy | Decision rule |
|---|---|
| Belady | Offline farthest-next-use using the held-out stream. With variable container sizes this is a heuristic oracle for the stated rule, and future-look-up indexing is prepared before timing. |
| HIST | Event-time adaptation of the histogram policy with 240 one-minute IAT bins, CoV threshold 2, 5th and 99th percentiles, 10% margins and a two-hour fallback. The histogram learns only from realized arrivals. Prewarm events enter the cache at their scheduled timestamp if there is free capacity and are charged separately. The upstream simulator's event-ordering issues led to a local implementation. |
| SIZE | Evict largest container footprint first. |
| LND | Greedy-Dual implementation of Landlord with clock-normalized rent and refill to the estimated retrieval cost on hit. Evictions use minimum remaining credit per MB. |
| FREQ | Evict shortest cache-resident hit streak first. |

All comparisons use the same atomically reusable one-container-per-function abstraction. Eviction ties use earliest priority update in the common replay. A 10-minute TTL expires after strictly more than 600 seconds. See [method mapping](method_mapping.md) for original sources, memory semantics and the estimated cold-penalty definition.

Figure 13 takes the maximum post-launch worker RSS sampled by psutil, including loaded arrays and policy runtime. This measurement avoids the pre-exec high-water mark inherited when the parent had recently allocated the replay prefix. The original kernel statistic is retained in the resource CSV. Figure 14 measures process CPU time per user request over cache decisions. For LACE it includes single-request model inference and normalization. Historical feature reconstruction and offline oracle indexing are timed outside the per-request metric. HIST event advancement is included in the per-request policy timing. Prewarm counts, rejected prewarms and estimated prewarm costs are present in the backing CSV. Figures 8–9 and 11 use cached LACE inference on causal features to keep replay efficiency practical.

## Reproduce the new run

From the project directory with Python 3.13 and a validated `cache/` from the initial campaign:

```bash
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-diagnostics.txt
out=results/azure2019-diagnostics-rerun
mkdir "$out"
.venv/bin/python -c 'import time; print(time.monotonic())' > "$out/start.monotonic"
OPENBLAS_NUM_THREADS=1 .venv/bin/python steps.py prepare cache "$out" 10000000 200000 1
OPENBLAS_NUM_THREADS=1 .venv/bin/python diagnostic_campaign.py "$out"
MPLCONFIGDIR=/tmp/lace-plt .venv/bin/python diagnostics_export.py "$out"
```

Execution uses a three-hour absolute wall-clock limit with a final validation reserve. The measured live pilot determines whether all nine resource policies can use all Representative events or need one common shorter prefix. All model, policy and export stage timings are saved in `run_evidence.csv`.

If a shell session ends while the campaign is running, invoke `diagnostic_campaign.py` again on the same output directory on the same boot. Completed stages are retained. The live resource replay saves its event position, timer totals and cache state every 100,000 requests. On restart, per-request CPU time uses the summed timed intervals and Figure 13 applies the same post-launch sampler to the completed worker. Table 3, Table 4 and the completed cache comparisons remain available during resource measurement.

Run `diagnostics_pressure.py "$out"` to generate the 10 GB ablation supplement and `diagnostics_report.py "$out"` to generate the output index and artifact checks.
