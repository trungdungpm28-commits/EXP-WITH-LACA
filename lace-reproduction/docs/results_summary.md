# LACE reproduction results

The primary experiment is the [longer-window campaign](../results/azure2019-expanded/). It completed 60 paired workload, capacity and policy runs on Debian 13 with an Intel i7-8550U and 24 GB installed RAM. The [initial campaign](../results/azure2019-reduced/) completed 36 additional runs. Its Representative slice lasted four minutes because three historically very busy functions accounted for most invocations. The primary campaign retained the original runs and froze a prospective cap of 25,000 historical invocations per candidate for Representative and Random. Both campaigns used the verified Microsoft Azure Functions 2019 trace with day 4 held out.

| Primary workload | Functions selected | Functions observed | User invocations | Window |
|---|---:|---:|---:|---|
| Representative | 400 | 324 | 99,956 | 18 h 40 min |
| Rare | 1,000 | 339 | 5,967 | 24 h |
| Random | 200 | 170 | 80,578 | 24 h |

The primary logical memory budgets were 1, 2, 4, 10 and 17.6 GB. The largest active working set was 19.471 GB, so 4 GB was the grid's middle point and replaced 22 GB for diagnostics. The run used 90,000 train and 10,000 validation examples. LACE used a reduced width-64, six-expert, top-three MoE with 18 offline training epochs and no online updates. The full model's selected checkpoint reached a validation MAE of 15.83 s and classification accuracy of 96.30%. In validation, 82.56% of targets fell in the shortest-IAT class.

## Figures 8–9

Each cell below is **cold-start ratio / estimated overhead**, both percentages, at 4 GB.

| Workload | LACE | FaasCache | TTL 10 min | LRU |
|---|---:|---:|---:|---:|
| Representative | 21.73 / 48.53 | 13.37 / 4.71 | 21.18 / 52.31 | 21.18 / 52.31 |
| Rare | 10.04 / 13.85 | 7.21 / 2.66 | 10.56 / 17.60 | 6.75 / 3.39 |
| Random | 4.25 / 35.53 | 1.65 / 2.08 | 2.76 / 30.34 | 1.63 / 19.64 |

FaasCache beat LACE on both measures at all five Representative capacities. LRU had the lowest Representative cold-start ratio at 10 GB (1.72%) and the same estimated overhead as FaasCache at 17.6 GB (0.66%). Rare's 4 GB results illustrate how per-function cost weighting can favor a policy with more cold requests. FaasCache served 430 cold requests compared with LRU's 403 and had lower estimated overhead. At the two capacities at or above 10 GB, LACE, FaasCache and LRU matched on Random because all 8.539 GB of its active footprint fit.

The reduced runs failed to reproduce the paper's consistent LACE advantage. They use minute-bin reconstruction, smaller history-conditioned cohorts, an offline reduced predictor, and a single reusable container per function. Overhead uses a historical maximum-minus-average execution-duration proxy. Microsoft's duration metadata excludes cold initialization, and the proxy can strongly favor retaining functions with variable runtimes. FaasCache explicitly weights this same proxy in GDSF. This gives a reason for divergence between the ratio and overhead rankings, without identifying the contribution of each design difference to the gap from the paper.

## Reduced ablation

At 4 GB, removing the classification task raised the Representative cold-start ratio to 30.99% and estimated overhead to 80.68%. The variant nevertheless had slightly lower held-out IAT MAE (53.64 s versus 54.98 s) on the same 99,632 labeled requests. The change in regression MAE and the change in cache results went in opposite directions.

## Resource measurements

At 4 GB on Representative, LACE used 373.35 MB peak worker RSS and 0.562 ms/request of process CPU time. The other workers each peaked at 41.56 MB and used 0.0020–0.0023 ms/request. LACE's measured per-request compute includes single-request inference and normalization. Feature history reconstruction happens during data preparation. These host resource results and the simulated latency outcomes use separate measurements. See [method mapping](method_mapping.md).

## Files

| Output | CSV | PNG |
|---|---|---|
| Figure 8, estimated cold-start overhead | [CSV](../results/azure2019-expanded/figure_08_cold_start_overhead.csv) | [PNG](../results/azure2019-expanded/figures/figure_08_cold_start_overhead.png) |
| Figure 9, cold-start ratio | [CSV](../results/azure2019-expanded/figure_09_cold_start_ratio.csv) | [PNG](../results/azure2019-expanded/figures/figure_09_cold_start_ratio.png) |
| Figure 10, training dynamics | [CSV](../results/azure2019-expanded/figure_10_training_dynamics.csv) | [PNG](../results/azure2019-expanded/figures/figure_10_training_dynamics.png) |
| Figure 11, reduced ablation | [CSV](../results/azure2019-expanded/figure_11_reduced_ablation.csv) | [PNG](../results/azure2019-expanded/figures/figure_11_reduced_ablation.png) |
| Figure 12, expert routing | [CSV](../results/azure2019-expanded/figure_12_expert_routing.csv) | [PNG](../results/azure2019-expanded/figures/figure_12_expert_routing.png) |
| Figure 13, worker RSS | [CSV](../results/azure2019-expanded/figure_13_worker_rss.csv) | [PNG](../results/azure2019-expanded/figures/figure_13_worker_rss.png) |
| Figure 14, worker CPU | [CSV](../results/azure2019-expanded/figure_14_policy_cpu.csv) | [PNG](../results/azure2019-expanded/figures/figure_14_policy_cpu.png) |
| Table 3-style matched IAT evaluation | [CSV](../results/azure2019-expanded/table_03_iat_metrics.csv) | |
| Per-run aggregates and provenance | [CSV](../results/azure2019-expanded/runs.csv) | |
| Deadline and worker evidence | [CSV](../results/azure2019-expanded/run_evidence.csv) | |
| Pre-run setup and recovered errors | [CSV](setup_evidence.csv) | |

The final campaign export completed at 18 min 3 s elapsed. The visual and numeric audit was finalized at 18 min 36 s elapsed. Both were within the shared six-hour budget. Initial dependency setup used bounded package requests before the shared supervisor started, as recorded in the setup evidence.
