"""Run the larger diagnostic reproduction with a three-hour deadline."""
import argparse
import csv
import json
import math
import platform
from pathlib import Path
import shutil
import time

import numpy as np

from campaign import Campaign, load, persist_runs, replay
from common import POLICIES, ROOT, SEED, WORKLOADS, sha256, write_csv, write_json
from diagnostic_policies import RESOURCE_POLICIES
from diagnostics_export import ABLATION_ORDER
from diagnostics_models import MODEL_NAMES, VARIANTS
from replay import Cache
from workloads import save_arrays

BUDGETS = (1000, 4000, 10000, 22000, 40000)
FIXED = 22000


class ResumableCampaign(Campaign):
    def worker(self, name, args, limit=1200, heavy=True):
        for row in reversed(self.evidence):
            if row["stage"] == name and row["status"] == "complete":
                return row
        log = self.out / "logs" / f"{name}.log"
        if log.exists():
            stamp = int(time.monotonic())
            archived = log.with_name(f"{name}.prior-{stamp}.log")
            log.replace(archived)
            self.evidence.append(dict(stage=name, status="interrupted",
                                      reason="prior shell terminated before saving stage completion",
                                      log=str(archived.relative_to(self.out))))
            write_csv(self.out / "run_evidence.csv", self.evidence)
        return super().worker(name, args, limit, heavy)


def build_predictions(out, variant):
    from model import Predictor, configure

    configure(1)
    model = Predictor(out / variant / "model.pt")
    routing = []
    for workload in WORKLOADS if variant == "full" else ("Representative",):
        stem = workload.lower()
        with np.load(out / f"{stem}.npz") as trace:
            x = trace["features"]
        predictions = np.empty(len(x), dtype=np.float32)
        counts = np.zeros(6, dtype=np.int64)
        sums = np.zeros(6)
        for i in range(0, len(x), 65536):
            pred, gate = model.predict(x[i:i+65536])
            predictions[i:i+len(pred)] = pred
            counts += (gate > 0).sum(0)
            sums += gate.sum(0, dtype=np.float64)
        assert np.isfinite(predictions).all() and (predictions >= 0).all()
        assert counts.sum() == 3*len(predictions)
        save_arrays(out / variant / f"{stem}_predictions.npz", pred_s=predictions)
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        for i in range(6):
            routing.append(dict(workload=workload, expert=i+1, normalized_count=counts[i]/counts.sum(),
                                normalized_summed_weight=sums[i]/sums.sum(),
                                normalized_mean_weight=means[i]/means.sum(), top_k_assignments=counts[i],
                                summed_weight=sums[i], mean_selected_weight=means[i], invocations=len(x)))
    if variant == "full":
        write_csv(out / "routing.csv", routing)


def preflight():
    from replay import toy
    toy()
    cache = Cache(1, "HIST")
    for i, t in enumerate((0, 600, 1200, 1800)):
        cache.request(i, t, 0, 1, 10., None)
    assert not cache.pool
    cache.histogram.advance(2339.)
    assert not cache.pool
    cache.histogram.advance(2340.)
    assert cache.pool and cache.histogram.prewarms == 1
    assert cache.request(4, 2400., 0, 1, 10., None)[0] == "warm"
    for policy, expected in (("Belady", 0), ("LND", 1), ("FREQ", 0), ("SIZE", 0)):
        cache = Cache(3, policy)
        cache.request(0, 0., 0, 2, 80., 50)
        cache.request(1, 1., 1, 1, 10., 5)
        if policy == "FREQ":
            cache.request(2, 2., 1, 1, 10., 6)
        assert cache.request(3, 3., 2, 1, 10., 7)[1][0] == expected


def resource_subset(out, n, subdir="resource_input"):
    source = out / "representative.npz"
    folder = out / subdir
    with np.load(source) as data:
        save_arrays(folder / source.name, events=data["events"][:n], features=data["features"][:n])
    spec = dict(path=source.name, sha256=sha256(folder / source.name), invocations=n)
    write_json(folder / "workloads.json", dict(workloads={"Representative": spec}))
    return folder


def simulate(c, tag, workload, policy, budget, variant="full", live=False, source=None, limit=None):
    out = c.out
    original_worker = c.worker
    if limit is not None:
        c.worker = lambda name, args: original_worker(name, args, limit=limit)
    try:
        checkpoint = out / variant / "model.pt"
        predicted = out / variant / f"{workload.lower()}_predictions.npz"
        row = replay(c, tag, source or out, workload, policy, budget, out / "replays" / f"{tag}.json",
                     model=checkpoint if policy == "LACE" and live else None,
                     predictions=predicted if policy == "LACE" and not live else None,
                     checkpoint_state=out / "replays" / "resource_LACE.checkpoint.json"
                     if tag == "resource_LACE" else None)
    finally:
        c.worker = original_worker
    if policy == "LACE":
        row.update(predictor=f"LACE_{variant}", model_sha256=sha256(checkpoint))
    c.mark(tag, row["status"], row["reason"])
    return row


def consolidate_models(c, statuses):
    out = c.out
    rows = {}
    for name in (*VARIANTS, *MODEL_NAMES[2:]):
        path = out / "method_scores" / f"{name}.json"
        label = {"full": "LACE", "wo_cls": "MoE"}.get(name, name)
        status = statuses.get(name, {})
        rows[label] = load(path) if path.exists() else dict(model=label, status=status.get("status", "planned"),
                      reason=status.get("reason", "worker has yet to run"))
    write_csv(out / "table_03_iat_metrics.csv", [rows[n] for n in MODEL_NAMES])
    write_csv(out / "table_04_feature_ablation.csv", [rows[n] for n in
              ("LACE", "wo_static", "wo_statistical", "wo_edcs", "wo_deltas")])


def audit(out):
    from export import read

    specs = load(out / "workloads.json")["workloads"]
    groups = [("runs.csv", list(read(out / "runs.csv"))),
              ("ablation.csv", list(read(out / "ablation.csv"))),
              ("resource_runs.csv", list(read(out / "resource_runs.csv")))]
    results = {}
    for file, rows in groups:
        complete = [r for r in rows if r["status"] == "complete"]
        totals = {}
        for row in complete:
            key = row["workload"]
            spec = specs[key] if file != "resource_runs.csv" or not (out / "resources_scope.json").exists() else load(out / "resources_scope.json")["workload"]
            n = int(row["request_count"])
            assert n == spec["invocations"] and row["workload_sha256"] == spec["sha256"]
            assert int(row["warm_invocations"]) + int(row["cold_invocations"]) == n
            assert int(row["peak_logical_mb"]) <= int(row["budget_mb"])
            warm = float(row["warm_reference_ms"])
            assert totals.setdefault(key, warm) == warm
            assert math.isclose(float(row["cold_ratio_pct"]), 100*int(row["cold_invocations"])/n)
            assert math.isclose(float(row["overhead_pct"]), 100*(float(row["policy_e2e_ms"])-warm)/warm, abs_tol=1e-9)
        results[file] = dict(completed=len(complete), planned=len(rows))
    target = sha256(out / "diagnostic_test.npz")
    for table in ("table_03_iat_metrics.csv", "table_04_feature_ablation.csv"):
        rows = read(out / table)
        complete = [r for r in rows if r["status"] == "complete"]
        assert all(r["test_sha256"] == target for r in complete)
        assert len({r["examples"] for r in complete}) <= 1
        results[table] = dict(completed=len(complete), planned=len(rows))
    write_json(out / "audit.json", results)


def campaign(out):
    start = float((out / "start.monotonic").read_text())
    c = ResumableCampaign.__new__(ResumableCampaign)
    c.out = out.resolve()
    out = c.out
    c.start, c.deadline, c.heavy = start, start+10800, start+10200
    c.evidence, c.status = [], {}
    if (out / "run_evidence.csv").exists():
        with (out / "run_evidence.csv").open() as stream:
            c.evidence = list(csv.DictReader(stream))
    if (out / "campaign_status.csv").exists():
        with (out / "campaign_status.csv").open() as stream:
            c.status = {row["stage"]: row for row in csv.DictReader(stream)}
    shutil.copy2(ROOT / "cache" / "data_validation.csv", out / "data_validation.csv")
    specs = load(out / "workloads.json")["workloads"]
    write_json(out / "configuration.json", dict(seed=SEED, platform=platform.platform(), width=96, epochs=24,
               budgets_mb=BUDGETS, fixed_diagnostic_mb=FIXED, policies=POLICIES,
               resource_policies=RESOURCE_POLICIES, workloads=specs,
               replays_use_azure_2019=True, monitoring_deadline_monotonic=c.deadline,
               sources={p.name: sha256(p) for p in ROOT.glob("*.py")}))
    c.mark("prepare_full_day4", "complete")
    c.require("toy", ["diagnostic_campaign.py", out, "--stage", "toy"])
    c.require("test_targets", ["diagnostics_models.py", "evaluation", out])
    c.require("train_full", ["diagnostics_models.py", "train", out, "full"], limit=1800)
    shutil.copy2(out / "full" / "training_history.csv", out / "training_history.csv")
    c.require("predict_full", ["diagnostic_campaign.py", out, "--stage", "predict", "--variant", "full"], limit=1200)

    n = specs["Representative"]["invocations"]
    pilot_n = 60000
    pdir = out / "pilot_input"
    if not (pdir / "workloads.json").exists():
        resource_subset(out, pilot_n, "pilot_input")
    pilot = simulate(c, "pilot_live", "Representative", "LACE", FIXED, live=True, source=pdir)
    if pilot["status"] != "complete":
        raise RuntimeError(pilot["reason"])
    estimated_live_s = pilot["runtime_s"] * n / pilot_n
    estimate = estimated_live_s + 1700
    write_json(out / "pilot_projection.json", dict(pilot=pilot, estimated_live_full_s=estimated_live_s,
              reserve_for_other_work_s=1700, estimated_remaining_s=estimate,
              absolute_deadline_s=c.deadline, event_count=n))
    print(f"Pilot forecast: full live LACE {estimated_live_s/60:.1f} min, remaining campaign {estimate/60:.1f} min", flush=True)

    grid = [(w,p,b) for w in WORKLOADS for b in BUDGETS for p in POLICIES]
    rows = [dict(workload=w, policy=p, budget_mb=b, status="planned", reason="pending") for w,p,b in grid]
    persist_runs(out, rows)
    for i,(w,p,b) in enumerate(grid):
        tag = f"{w.lower()}_{b}_{p.split()[0]}"
        rows[i] = simulate(c, tag, w, p, b)
        persist_runs(out, rows)
    c.require("export_main", ["diagnostics_export.py", out], heavy=False)

    scores = {}
    models = [*VARIANTS, *MODEL_NAMES[2:]]
    ablation = {}
    for policy, label in (("FaasCache", "FaasCache"), ("LACE", "LACE (full)")):
        ablation[label] = {**next(r for r in rows if r["workload"] == "Representative" and
                           r["policy"] == policy and r["budget_mb"] == FIXED), "label": label}
    for name in models:
        if name in VARIANTS:
            use_model = name == "full"
            if not use_model:
                fit = c.worker(f"train_{name}", ["diagnostics_models.py", "train", out, name], limit=1800)
                scores[name] = fit
                use_model = fit["status"] == "complete"
            if use_model:
                scores[name] = c.worker(f"score_{name}", ["diagnostics_models.py", "learned", out, name])
                if name != "full":
                    label = "MoE (wo_cls)" if name == "wo_cls" else name
                    pred = c.worker(f"predict_{name}", ["diagnostic_campaign.py", out, "--stage", "predict", "--variant", name])
                    if pred["status"] == "complete":
                        row = simulate(c, f"ablation_{name}", "Representative", "LACE", FIXED, variant=name)
                    else:
                        row = dict(status=pred["status"], reason=pred["reason"])
                    ablation[label] = {**row, "label": label}
            elif name != "full":
                label = "MoE (wo_cls)" if name == "wo_cls" else name
                ablation[label] = dict(label=label, status=scores[name]["status"], reason=scores[name]["reason"])
        else:
            scores[name] = c.worker(f"fit_score_{name}", ["diagnostics_models.py", "baseline", out, name], limit=1400)
        consolidate_models(c, scores)
        write_csv(out / "ablation.csv", [ablation.get(label, dict(label=label, status="planned",
                  reason="pending")) for label in ABLATION_ORDER])
    c.require("export_ablations_tables", ["diagnostics_export.py", out], heavy=False)

    # Freeze the diagnostic population for all nine policies using the live pilot rate.
    scope_file = out / "resources_scope.json"
    if scope_file.exists():
        scope = load(scope_file)
        approved_n = scope["event_count"]
        source = out if scope["full_workload"] else out / "resource_input"
        assert sha256(source / scope["workload"]["path"]) == scope["workload"]["sha256"]
    else:
        seconds = max(0, c.heavy-time.monotonic()-420)
        approved_n = min(n, int(seconds/(1.15*pilot["runtime_s"]/pilot_n)))
        if approved_n < 1:
            raise RuntimeError("No remaining time for resource replays")
        source = resource_subset(out, approved_n) if approved_n < n else out
        spec = load(source/"workloads.json")["workloads"]["Representative"]
        write_json(scope_file, dict(workload=spec, event_count=approved_n, pilot_events=pilot_n,
                   budget_mb=FIXED, full_workload=approved_n==n))
    resources = {p: dict(policy=p, status="planned", reason="pending", budget_mb=FIXED)
                 for p in RESOURCE_POLICIES}
    for p in (*RESOURCE_POLICIES[1:], "LACE"):
        name = "resource_" + p.split()[0]
        row = simulate(c, name, "Representative", p, FIXED, source=source, live=p=="LACE", limit=7600)
        resources[p] = row
        write_csv(out / "resource_runs.csv", [resources[p] for p in RESOURCE_POLICIES])
    c.require("export_final", ["diagnostics_export.py", out], heavy=False)
    audit(out)
    checks = load(out / "audit.json")
    c.mark("campaign", "complete" if all(v["completed"] == v["planned"] for v in checks.values()) else "partial")
    write_json(out / "elapsed.json", dict(elapsed_s=time.monotonic()-start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--stage", choices=("predict", "toy", "audit"))
    parser.add_argument("--variant", default="full")
    args = parser.parse_args()
    if args.stage == "predict":
        build_predictions(args.out, args.variant)
    elif args.stage == "toy":
        preflight()
    elif args.stage == "audit":
        audit(args.out)
    else:
        campaign(args.out)
