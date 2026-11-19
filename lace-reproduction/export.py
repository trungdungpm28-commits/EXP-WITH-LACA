import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from common import POLICIES, WORKLOADS, atomic, sha256, write_csv, write_json

COLORS = ("#282828", "#0072B2", "#D55E00", "#009E73")
MARKERS = ("o", "s", "^", "D")
NAMES = {
    8: ("cold_start_overhead", "Increase over warm-reference time (%)"),
    9: ("cold_start_ratio", "Cold invocations (%)"),
    10: ("training_dynamics", "Training dynamics"),
    11: ("reduced_ablation", "Reduced ablation"),
    12: ("expert_routing", "Expert routing"),
    13: ("worker_rss", "Peak worker RSS (MB)"),
    14: ("policy_cpu", "Mean policy CPU time (ms/request)"),
}


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def render(out, number, fig):
    path = out / "figures" / f"figure_{number:02}_{NAMES[number][0]}.png"
    atomic(path, lambda p: fig.savefig(p, format="png", dpi=180, facecolor="white"))
    plt.close(fig)
    with Image.open(path) as result:
        result.verify()
    return path


def csv_path(out, number):
    return out / f"figure_{number:02}_{NAMES[number][0]}.csv"


def main_figure(out, number, runs):
    metric = "overhead_pct" if number == 8 else "cold_ratio_pct"
    rows = [{**run, "budget_gb": float(run["budget_mb"]) / 1000} for run in runs]
    write_csv(csv_path(out, number), rows)
    rows = read(csv_path(out, number))
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4))
    for workload, ax in zip(WORKLOADS, axes):
        for policy, color, marker in zip(POLICIES, COLORS, MARKERS):
            chosen = sorted((r for r in rows if r["workload"] == workload and r["policy"] == policy),
                            key=lambda r: float(r["budget_gb"]))
            values = [float(r[metric]) if r["status"] == "complete" and r.get(metric) and
                      r.get("comparable") == "True" else float("nan") for r in chosen]
            ax.plot([float(r["budget_gb"]) for r in chosen], values, label=policy,
                    color=color, marker=marker, markersize=4, linewidth=1.6)
        ax.set(title=workload, xlabel="Logical cache budget (GB)")
        ax.set_ylim(bottom=0)
        ticks = sorted({float(r["budget_gb"]) for r in rows})
        ax.set_xticks(ticks, labels=[f"{tick:g}" for tick in ticks])
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel(NAMES[number][1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.52, .92))
    fig.suptitle(f"Figure {number} reproduction · Azure Functions 2019, held-out day 4")
    note = ("Cold penalty estimated from historical maximum minus average duration."
            if number == 8 else "Cold means no resident container at invocation time.")
    note += "  Reduced scale, panel-specific y-axes. Missing cells are gaps."
    fig.text(.50, .04, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(.02, .08, .99, .82))
    render(out, number, fig)


def diagnostics(out, numbers=(10, 11, 12, 13, 14)):
    if 10 in numbers and (out / "training_history.csv").exists():
        rows = [r for r in read(out / "training_history.csv") if r["variant"] == "full"]
        write_csv(csv_path(out, 10), rows)
        rows = read(csv_path(out, 10))
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout="constrained")
        fields = (("train_accuracy_pct", "val_accuracy_pct"), ("train_mae_s", "val_mae_s"),
                  ("sigma_reg", "sigma_cls"))
        for ax, columns, title, labels in zip(axes, fields, ("Classification accuracy (%)", "Regression MAE (s)",
                    "Learned uncertainty σ"), (("Train", "Validation"), ("Train", "Validation"), ("Regression", "Classification"))):
            for col, label, color in zip(columns, labels, COLORS):
                ax.plot([int(r["epoch"]) for r in rows], [float(r[col]) for r in rows], label=label, color=color)
            ax.set(xlabel="Epoch", title=title)
            ax.grid(axis="y", alpha=.2)
            ax.legend(frameon=False)
        fig.suptitle("Figure 10 reproduction · Reduced LACE training")
        render(out, 10, fig)
    for number, source, metrics, titles in [
        (11, "ablation.csv", ("overhead_pct", "cold_ratio_pct"), (NAMES[8][1], NAMES[9][1])),
        (13, "resource_runs.csv", ("peak_worker_rss_mb",), (NAMES[13][1],)),
        (14, "resource_runs.csv", ("policy_cpu_ms_per_request",), (NAMES[14][1],)),
    ]:
        if number not in numbers or not (out / source).exists():
            continue
        rows = read(out / source)
        if not rows:
            continue
        write_csv(csv_path(out, number), rows)
        rows = read(csv_path(out, number))
        fig, axes = plt.subplots(1, len(metrics), squeeze=False, figsize=(6.4 * len(metrics), 3.7), layout="constrained")
        for ax, field, title in zip(axes[0], metrics, titles):
            color_by_policy = dict(zip(POLICIES, COLORS))
            values = [float(r[field]) for r in rows]
            bars = ax.bar([r["label"] for r in rows], values,
                          color=[color_by_policy.get(r["label"], "#7C5D9B") for r in rows])
            ax.bar_label(bars, labels=[f"{v:.3g}" for v in values], padding=3, fontsize=9)
            ax.margins(y=.16)
            ax.set(ylabel=title)
            ax.grid(axis="y", alpha=.2)
        scope = "Reduced ablation, historical penalty proxy" if number == 11 else "Measured simulator worker"
        fig.suptitle(f"Figure {number} reproduction · {scope}\nRepresentative, {float(rows[0]['budget_mb'])/1000:g} GB")
        render(out, number, fig)
    routing = out / "routing.csv"
    if 12 in numbers and routing.exists():
        write_csv(csv_path(out, 12), read(routing))
        rows = read(csv_path(out, 12))
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True, layout="constrained")
        for workload, ax in zip(WORKLOADS, axes):
            rs = [r for r in rows if r["workload"] == workload]
            x = np.arange(len(rs))
            for offset, col, label, color in zip((-.24, 0, .24),
                    ("normalized_count", "normalized_summed_weight", "normalized_mean_weight"),
                    ("Count", "Summed weight", "Mean weight"), COLORS):
                ax.bar(x + offset, [float(r[col]) for r in rs], width=.22, label=label, color=color)
            ax.set(title=workload, xticks=x, xticklabels=[r["expert"] for r in rs], xlabel="Expert")
        axes[0].set_ylabel("Normalized across experts")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Figure 12 reproduction · Routing on held-out invocations")
        render(out, 12, fig)


def verify(out, runs):
    workloads = json.loads((out / "workloads.json").read_text())["workloads"]
    for workload, spec in workloads.items():
        assert sha256(out / spec["path"]) == spec["sha256"]
        assert sum(int(r["replay_invocations"]) for r in read(out / "workload_manifest.csv")
                   if r["workload"] == workload) == spec["invocations"]
        rows = [r for r in runs if r["workload"] == workload and r["status"] == "complete"]
        assert len({r["warm_reference_ms"] for r in rows}) <= 1
        for row in rows:
            n = int(row["request_count"])
            assert row["workload_sha256"] == spec["sha256"] and n == spec["invocations"]
            assert int(row["cold_invocations"]) + int(row["warm_invocations"]) == n
            assert int(row["peak_logical_mb"]) <= int(row["budget_mb"])
            assert math.isclose(100 * int(row["cold_invocations"]) / n, float(row["cold_ratio_pct"]))
            warm = float(row["warm_reference_ms"])
            if warm:
                delta = float(row["policy_e2e_ms"]) - warm
                assert math.isclose(100 * delta / warm, float(row["overhead_pct"]), abs_tol=1e-9)
    pngs = sorted(path.name for path in (out / "figures").glob("*.png"))
    write_json(out / "verification.json", dict(succeeded=True,
                completed_comparisons=sum(r["status"] == "complete" for r in runs),
                verified_pngs=pngs, exporter_sha256=sha256(Path(__file__))))


def export(out):
    runs = read(out / "runs.csv")
    for number in (8, 9):
        main_figure(out, number, runs)
    diagnostics(out)
    verify(out, runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    export(parser.parse_args().out)
