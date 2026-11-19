# Azure Functions 2019 · Larger LACE reproduction

The replays use observed Azure minute counts, deterministically expanded into events. Training and cohort selection use days 1–3. Replay uses the entire fourth day for all three cohorts. See [the diagnostic method](../../docs/diagnostics_method.md).

| Workload | Functions selected | Active functions | Events |
|---|---:|---:|---:|
| Representative | 400 | 331 | 9,295,955 |
| Rare | 1,000 | 339 | 5,967 |
| Random | 200 | 161 | 462,526 |

Full LACE trained on 180,000 examples and validated on 20,000. All model rows use the same 300,000 held-out IAT targets. The primary grid uses four policies at 1, 4, 10, 22 and 40 GB. Figures 13–14 use 7,817,079 identical events across the nine policies at 22 GB.

The test IAT median is 24.0 ms and 97.31% of targets are below one second. The small targets amplify percentage errors in MAPE. The MLP reached its 24-epoch cap with a convergence warning. Its measured row remains in Table 3.

## What ran

| Artifact | Completed | Planned |
|---|---:|---:|
| [runs.csv](runs.csv) | 60 | 60 |
| [ablation.csv](ablation.csv) | 7 | 7 |
| [resource_runs.csv](resource_runs.csv) | 9 | 9 |
| [table_03_iat_metrics.csv](table_03_iat_metrics.csv) | 8 | 8 |
| [table_04_feature_ablation.csv](table_04_feature_ablation.csv) | 5 | 5 |

## Figures and tables

| Output | Data | Image |
|---|---|---|
| Figure 8 · Estimated overhead | [CSV](figure_08_cold_start_overhead.csv) | [PNG](figures/figure_08_cold_start_overhead.png) |
| Figure 9 · Cold-start ratio | [CSV](figure_09_cold_start_ratio.csv) | [PNG](figures/figure_09_cold_start_ratio.png) |
| Figure 10 · Training dynamics | [CSV](figure_10_training_dynamics.csv) | [PNG](figures/figure_10_training_dynamics.png) |
| Figure 11 · Seven-way ablation at 22 GB | [CSV](figure_11_reduced_ablation.csv) | [PNG](figures/figure_11_reduced_ablation.png) |
| Figure 11 supplement · The same ablation at 10 GB | [CSV](figure_11b_ablation_at_10gb.csv) | [PNG](figures/figure_11b_ablation_at_10gb.png) |
| Figure 12 · Expert routing | [CSV](figure_12_expert_routing.csv) | [PNG](figures/figure_12_expert_routing.png) |
| Figure 13 · Nine-policy worker RSS | [CSV](figure_13_worker_rss.csv) | [PNG](figures/figure_13_worker_rss.png) |
| Figure 14 · Nine-policy CPU time | [CSV](figure_14_policy_cpu.csv) | [PNG](figures/figure_14_policy_cpu.png) |
| Table 3 · Eight predictors | [CSV](table_03_iat_metrics.csv) | [PNG](figures/table_03_iat_metrics.png) |
| Table 4 · Feature ablations | [CSV](table_04_feature_ablation.csv) | [PNG](figures/table_04_feature_ablation.png) |

## Matched Representative results

The active footprint was 20.774 GB. Figure 11 retains the paper's 22 GB setting and the 10 GB supplement shows the same variants under memory pressure. The cold overhead uses the historical maximum-minus-average duration proxy.

### 22000 MB

| Policy | Cold starts (%) | Estimated overhead (%) |
|---|---:|---:|
| FaasCache | 0.00356 | 0.12351 |
| LACE (full) | 0.00356 | 0.12351 |
| wo_static | 0.00356 | 0.12351 |
| wo_statistical | 0.00356 | 0.12351 |
| wo_edcs | 0.00356 | 0.12351 |
| wo_deltas | 0.00356 | 0.12351 |
| MoE (wo_cls) | 0.00356 | 0.12351 |

### 10000 MB

| Policy | Cold starts (%) | Estimated overhead (%) |
|---|---:|---:|
| FaasCache | 0.07829 | 0.22047 |
| LACE (full) | 0.10779 | 4.82491 |
| wo_static | 0.08163 | 4.08485 |
| wo_statistical | 0.08847 | 4.29135 |
| wo_edcs | 0.08888 | 4.22687 |
| wo_deltas | 0.09843 | 4.50672 |
| MoE (wo_cls) | 0.12763 | 5.17430 |

### Worker resource diagnostics

| Policy | Peak RSS (MB) | Process CPU (ms/request) |
|---|---:|---:|
| LACE | 1640.95 | 0.653317 |
| Belady | 409.92 | 0.002077 |
| FaasCache | 347.38 | 0.002079 |
| TTL 10 min | 347.39 | 0.002294 |
| LRU | 347.45 | 0.002036 |
| HIST | 347.69 | 0.006448 |
| SIZE | 347.40 | 0.001828 |
| LND | 347.27 | 0.002245 |
| FREQ | 347.40 | 0.001948 |

[Worker timings](run_evidence.csv) · [Campaign status](campaign_status.csv) · [Workload manifest](workload_manifest.csv) · [Policy data](runs.csv)

Wall time through this report was 2:42:05. The monotonic experiment clock recorded 2:34:34.
