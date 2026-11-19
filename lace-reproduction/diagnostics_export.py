import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from common import atomic, sha256, write_csv, write_json
from diagnostic_policies import RESOURCE_POLICIES
from export import diagnostics, main_figure, read, render

ABLATION_ORDER = ("FaasCache", "LACE (full)", "wo_static", "wo_statistical", "wo_edcs", "wo_deltas", "MoE (wo_cls)")
FIELDS = ("mae_s", "rmse_s", "mape_pct", "r2", "mse_s2")


def table(out, filename, title):
    rows = read(out / f"{filename}.csv")
    cells = []
    for r in rows:
        if r["status"] == "complete":
            cells.append([r["model"], f"{float(r['mae_s']):,.2f}", f"{float(r['rmse_s']):,.2f}",
                          f"{float(r['mape_pct']):.2f}", f"{float(r['r2']):.4f}", f"{float(r['mse_s2']):.3g}"])
        else:
            cells.append([r["model"], *["—"]*5])
    fig, ax = plt.subplots(figsize=(9, 2.1 + .35*len(rows)))
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.text(.5, .02, "Matched held-out IAT targets from Representative · Azure Functions 2019",
            transform=ax.transAxes, ha="center", fontsize=9)
    tab = ax.table(cellText=cells, colLabels=["Model", "MAE (s) ↓", "RMSE (s) ↓", "MAPE (%) ↓", "R² ↑", "MSE (s²) ↓"],
                   colWidths=[.23, .15, .155, .15, .155, .16], loc="center", cellLoc="right", bbox=(.02, .12, .96, .82))
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    for (i, j), cell in tab.get_celld().items():
        cell.set_edgecolor("#d9dfe5")
        if i == 0:
            cell.set_facecolor("#17324D")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif i % 2:
            cell.set_facecolor("#f1f5f8")
        if j == 0:
            cell.get_text().set_ha("left")
    path = out / "figures" / f"{filename}.png"
    atomic(path, lambda p: fig.savefig(p, format="png", facecolor="white", dpi=180, bbox_inches="tight"))
    plt.close(fig)
    with Image.open(path) as image:
        image.verify()


def bars(ax, rows, label, field, ylabel, colors):
    x = np.arange(len(rows))
    values = [float(r[field]) if r["status"] == "complete" and r.get(field) else np.nan for r in rows]
    plots = ax.bar(x, values, color=colors[:len(rows)], edgecolor="#555", linewidth=.6)
    for index, (patch, value) in enumerate(zip(plots, values)):
        patch.set_hatch(("", "//", "xx", "..", "\\\\", "++", "oo", "**", "--")[index])
        show = f"{value:.3g}" if np.isfinite(value) else "Missing"
        ax.annotate(show, xy=(index, value if np.isfinite(value) else 0), ha="center",
                    xytext=(0, 4), textcoords="offset points", fontsize=9)
    ax.margins(y=.18)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, [r[label] for r in rows], rotation=24, ha="right")
    ax.grid(axis="y", alpha=.19)
    ax.set_axisbelow(True)


def export(out):
    plots = out / "figures"
    plots.mkdir(exist_ok=True)
    colors = ["#426E91", "#d9d9d9", "#6ea4c6", "#93bfbf", "#b8cce0", "#73a9a9", "#add2c8", "#c1c8ce", "#9db8d0"]
    if (out / "runs.csv").exists():
        runs = read(out / "runs.csv")
        for number in (8, 9):
            main_figure(out, number, runs)
    diagnostics(out, numbers=(10, 12))
    if (out / "ablation.csv").exists():
        rows = sorted(read(out / "ablation.csv"), key=lambda r: ABLATION_ORDER.index(r["label"]))
        export_rows = out / "figure_11_reduced_ablation.csv"
        write_csv(export_rows, rows)
        fig, axes = plt.subplots(2, 1, figsize=(8.9, 7.5), layout="constrained")
        for ax, field, label in zip(axes, ("overhead_pct", "cold_ratio_pct"),
                                     ("(a) Estimated overhead (%)", "(b) Cold starts (%)")):
            bars(ax, read(export_rows), "label", field, label, colors)
        fig.suptitle("Figure 11 · Feature and label ablations\nRepresentative at 22 GB")
        render(out, 11, fig)
    if (out / "resource_runs.csv").exists():
        rows = sorted(read(out / "resource_runs.csv"), key=lambda r: RESOURCE_POLICIES.index(r["policy"]))
        for number, stem, field, label in [
                (13, "worker_rss", "peak_worker_rss_mb", "Peak worker RSS (MB)"),
                (14, "policy_cpu", "policy_cpu_ms_per_request", "Process CPU time (ms/request)")]:
            data = out / f"figure_{number}_{stem}.csv"
            write_csv(data, rows)
            fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
            bars(ax, read(data), "policy", field, label, colors)
            detail = "Measured peak worker memory" if number == 13 else "Measured per-request policy time"
            fig.suptitle(f"Figure {number} · {detail}\nRepresentative at 22 GB · LACE includes live CPU inference")
            render(out, number, fig)
    for filename, title in [
            ("table_03_iat_metrics", "Table 3 · Inter-arrival prediction performance"),
            ("table_04_feature_ablation", "Table 4 · Inter-arrival prediction by feature ablation")]:
        if (out / f"{filename}.csv").exists():
            table(out, filename, title)
    write_json(out / "exports.json", {p.name: sha256(p) for p in sorted(plots.glob("*.png"))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    export(parser.parse_args().out)
