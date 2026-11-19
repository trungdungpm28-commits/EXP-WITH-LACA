"""Supplement Figure 11 at a capacity below the frozen active footprint."""
import argparse
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from PIL import Image

from campaign import Campaign
from common import atomic, write_csv
from diagnostic_campaign import simulate
from diagnostics_export import ABLATION_ORDER, bars
from export import read


def run(out):
    out = out.resolve()
    start = float((out / "start.monotonic").read_text())
    c = Campaign.__new__(Campaign)
    # Write separate evidence so the primary campaign ledger stays intact.
    c.out = out / "pressure_10gb"
    c.out.mkdir(exist_ok=False)
    c.start, c.deadline, c.heavy = start, start+10800, start+10500
    c.evidence, c.status = [], {}
    variants = ("wo_static", "wo_statistical", "wo_edcs", "wo_deltas", "wo_cls")
    all_rows = read(out / "runs.csv")
    rows = []
    for policy, label in (("FaasCache", "FaasCache"), ("LACE", "LACE (full)")):
        row = next(r for r in all_rows if r["policy"] == policy and r["workload"] == "Representative"
                   and int(r["budget_mb"]) == 10000)
        rows.append({**row, "label": label})
    for variant, label in zip(variants, ABLATION_ORDER[2:]):
        prediction = out / variant / "representative_predictions.npz"
        tag = "pressure_" + variant
        target = c.out / f"{tag}.json"
        evidence = c.worker(tag, ["replay.py", "--inputs", out / "representative.npz",
                                  "--policy", "LACE", "--budget", "10000", "--output", target,
                                  "--prediction", prediction], limit=300)
        from campaign import load
        row = load(target) if evidence["status"] == "complete" else {}
        rows.append(dict(label=label, variant=variant, status=evidence["status"], reason=evidence["reason"], **row))
    for r in rows:
        if r["status"] != "complete":
            continue
        assert r["workload_sha256"] == rows[0]["workload_sha256"]
        assert int(r["request_count"]) == int(rows[0]["request_count"])
        assert math.isclose(float(r["cold_ratio_pct"]), 100*int(r["cold_invocations"])/int(r["request_count"]))
    csv_path = out / "figure_11b_ablation_at_10gb.csv"
    write_csv(csv_path, rows)
    rows = read(csv_path)
    fig, axes = plt.subplots(2, 1, figsize=(8.9, 7.5), layout="constrained")
    colors = ["#d9d9d9", "#426E91", "#6ea4c6", "#93bfbf", "#b8cce0", "#73a9a9", "#add2c8"]
    for ax, field, title in zip(axes, ("overhead_pct", "cold_ratio_pct"),
                               ("(a) Estimated overhead (%)", "(b) Cold starts (%)")):
        bars(ax, rows, "label", field, title, colors)
    fig.suptitle("Figure 11 supplement · Feature and label ablations\nRepresentative under cache pressure at 10 GB")
    path = out / "figures" / "figure_11b_ablation_at_10gb.png"
    atomic(path, lambda p: fig.savefig(p, format="png", facecolor="white", dpi=180))
    plt.close(fig)
    with Image.open(path) as image:
        image.verify()
    c.mark("pressure_comparison", "complete" if all(r["status"] == "complete" for r in rows) else "partial")
    print(f"Supplement completed at {time.monotonic()-start:.1f} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
