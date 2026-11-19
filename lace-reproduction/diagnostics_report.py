"""Create an index and numerical handoff from completed diagnostic artifacts."""
import argparse
import datetime
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from common import atomic, sha256, write_csv, write_json
from export import read


def resource_rss(out):
    """Use samples from the launched replay process, excluding pre-exec high water."""
    from diagnostics_export import export

    resource = out / "resource_runs.csv"
    samples = {r["stage"]: r["peak_process_tree_rss_mb"]
               for r in read(out / "run_evidence.csv")
               if r["stage"].startswith("resource_") and r["status"] == "complete"}
    rows = read(resource)
    for row in rows:
        if row["status"] != "complete":
            continue
        row.setdefault("kernel_rusage_maxrss_mb", row["peak_worker_rss_mb"])
        row["peak_worker_rss_mb"] = samples[row["run_id"]]
        row["peak_worker_rss_method"] = "Maximum post-launch process RSS sampled by psutil"
    write_csv(resource, rows)
    export(out)


def report(out):
    resource_rss(out)
    workloads = json.loads((out / "workloads.json").read_text())["workloads"]
    audit = json.loads((out / "audit.json").read_text())
    resources = read(out / "resource_runs.csv")
    resource_scope = json.loads((out / "resources_scope.json").read_text())
    lines = ["# Azure Functions 2019 · Larger LACE reproduction", "",
             "The replays use observed Azure minute counts, deterministically expanded into events. "
             "Training and cohort selection use days 1–3. Replay uses the entire fourth day for all three cohorts. "
             "See [the diagnostic method](../../docs/diagnostics_method.md).", "",
             "| Workload | Functions selected | Active functions | Events |",
             "|---|---:|---:|---:|"]
    for name, s in workloads.items():
        lines.append(f"| {name} | {s['functions']:,} | {s['active_functions']:,} | {s['invocations']:,} |")
    training = json.loads((out / "training_inputs.json").read_text())
    sample = json.loads((out / "diagnostic_test.json").read_text())
    with np.load(out / "diagnostic_test.npz") as test:
        labels = test["y"]
        median_ms = 1000*np.median(labels)
        pct_subsecond = 100*np.mean(labels < 1)
    lines.extend(["",
        f"Full LACE trained on {training['train_examples']:,} examples and validated on "
        f"{training['val_examples']:,}. All model rows use the same {sample['test_examples']:,} held-out IAT targets. "
        "The primary grid uses four policies at 1, 4, 10, 22 and 40 GB. "
        f"Figures 13–14 use {resource_scope['event_count']:,} identical events across the nine policies at 22 GB.",
        "",
        f"The test IAT median is {median_ms:.1f} ms and {pct_subsecond:.2f}% of targets are below one second. "
        "The small targets amplify percentage errors in MAPE. "
        "The MLP reached its 24-epoch cap with a convergence warning. Its measured row remains in Table 3.",
        "",
        "## What ran", "",
        "| Artifact | Completed | Planned |",
        "|---|---:|---:|"])
    for name, s in audit.items():
        lines.append(f"| [{name}]({name}) | {s['completed']} | {s['planned']} |")
    checked = []
    items = [
        ("Figure 8 · Estimated overhead", "figure_08_cold_start_overhead"),
        ("Figure 9 · Cold-start ratio", "figure_09_cold_start_ratio"),
        ("Figure 10 · Training dynamics", "figure_10_training_dynamics"),
        ("Figure 11 · Seven-way ablation at 22 GB", "figure_11_reduced_ablation"),
        ("Figure 11 supplement · The same ablation at 10 GB", "figure_11b_ablation_at_10gb"),
        ("Figure 12 · Expert routing", "figure_12_expert_routing"),
        ("Figure 13 · Nine-policy worker RSS", "figure_13_worker_rss"),
        ("Figure 14 · Nine-policy CPU time", "figure_14_policy_cpu"),
        ("Table 3 · Eight predictors", "table_03_iat_metrics"),
        ("Table 4 · Feature ablations", "table_04_feature_ablation")]
    lines.extend(["", "## Figures and tables", "", "| Output | Data | Image |", "|---|---|---|"])
    for title, name in items:
        plot = out / "figures" / f"{name}.png"
        data = out / f"{name}.csv"
        csv_link = f"[CSV]({data.name})" if data.exists() else "Pending"
        image_link = "Pending"
        if plot.exists() and data.exists():
            image_link = f"[PNG](figures/{plot.name})"
            with Image.open(plot) as image:
                image.load()
                checked.append(dict(file=plot.name, width=image.width, height=image.height,
                                    sha256=sha256(plot), data_sha256=sha256(data)))
        lines.append(f"| {title} | {csv_link} | {image_link} |")
    lines.extend(["", "## Matched Representative results", "",
                  "The active footprint was 20.774 GB. Figure 11 retains the paper's 22 GB setting and the 10 GB supplement "
                  "shows the same variants under memory pressure. The cold overhead uses the "
                  "historical maximum-minus-average duration proxy."])
    for name in ("figure_11_reduced_ablation.csv", "figure_11b_ablation_at_10gb.csv"):
        source = out / name
        if not source.exists():
            continue
        rows = read(source)
        lines.extend(["", f"### {rows[0]['budget_mb']} MB", "", "| Policy | Cold starts (%) | Estimated overhead (%) |",
                      "|---|---:|---:|"])
        for row in rows:
            if row["status"] == "complete":
                lines.append(f"| {row['label']} | {float(row['cold_ratio_pct']):.5f} | {float(row['overhead_pct']):.5f} |")
            else:
                lines.append(f"| {row['label']} | Pending | Pending |")
    lines.extend(["", "### Worker resource diagnostics", "", "| Policy | Peak RSS (MB) | Process CPU (ms/request) |",
                  "|---|---:|---:|"])
    for row in resources:
        if row["status"] == "complete":
            lines.append(f"| {row['policy']} | {float(row['peak_worker_rss_mb']):.2f} | "
                         f"{float(row['policy_cpu_ms_per_request']):.6f} |")
        else:
            lines.append(f"| {row['policy']} | {row['status']} | {row['status']} |")
    lines.extend(["", "[Worker timings](run_evidence.csv) · [Campaign status](campaign_status.csv) · "
                  "[Workload manifest](workload_manifest.csv) · [Policy data](runs.csv)"])
    start_file = out / "start.monotonic"
    elapsed = time.monotonic() - float(start_file.read_text())
    wall_elapsed = time.time() - start_file.stat().st_mtime
    lines.extend(["", f"Wall time through this report was {datetime.timedelta(seconds=int(wall_elapsed))}. "
                  f"The monotonic experiment clock recorded {datetime.timedelta(seconds=int(elapsed))}.", ""])
    atomic(out / "REPORT.md", lambda path: path.write_text("\n".join(lines)))
    write_json(out / "report_verification.json", dict(elapsed_s=elapsed, wall_elapsed_s=wall_elapsed,
               checked_images=checked, audit=audit))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    report(p.parse_args().out)
