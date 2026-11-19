import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

import psutil

from common import POLICIES, ROOT, SEED, WORKLOADS, atomic, sha256, write_csv, write_json


def load(path):
    return json.loads(path.read_text())


class Campaign:
    def __init__(self, out, deadline):
        self.out = out.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.deadline = deadline
        self.heavy = deadline - 1800
        self.start = deadline - 21600
        self.evidence = []
        self.status = {}

    def mark(self, key, status, reason=""):
        self.status[key] = dict(stage=key, status=status, reason=reason)
        write_csv(self.out / "campaign_status.csv", self.status.values())

    def worker(self, name, args, limit=1200, heavy=True):
        stop = self.heavy if heavy else self.deadline
        before = time.monotonic()
        requested_stop = min(stop, before + limit)
        command = [sys.executable, str(ROOT / args[0]), *map(str, args[1:])]
        row = dict(stage=name, command=shlex.join(command), start_elapsed_s=before-self.start,
                   absolute_monotonic_deadline=self.deadline, worker_cutoff_monotonic=requested_stop)
        self.evidence.append(row)
        if before >= requested_stop:
            row.update(status="skipped", reason="deadline reserve reached", runtime_s=0.)
            write_csv(self.out / "run_evidence.csv", self.evidence)
            return row
        log = self.out / "logs" / f"{name}.log"
        log.parent.mkdir(exist_ok=True)
        disk_before = shutil.disk_usage(self.out).free
        available = psutil.virtual_memory().available
        peak = 0
        reason = ""
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "MPLCONFIGDIR": str(self.out / ".mpl"),
               "LACE_MONOTONIC_DEADLINE": str(self.deadline),
               "LACE_WORKER_CUTOFF": str(requested_stop)}
        with log.open("w") as stream:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True, env=env)
            try:
                while proc.poll() is None:
                    remaining = requested_stop - time.monotonic()
                    try:
                        root = psutil.Process(proc.pid)
                        procs = [root, *root.children(recursive=True)]
                        peak = max(peak, sum(p.memory_info().rss for p in procs if p.is_running()))
                    except psutil.Error:
                        pass
                    available = min(available, psutil.virtual_memory().available)
                    if remaining <= 0:
                        reason = "stage timeout or absolute cutoff"
                        break
                    if available < 4 * 2**30:
                        reason = "available system RAM dropped below 4 GiB"
                        break
                    if shutil.disk_usage(self.out).free < 2**30:
                        reason = "free disk dropped below 1 GiB"
                        break
                    try:
                        proc.wait(timeout=min(remaining, .25))
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                if proc.poll() is None:
                    # Kill the process group, including a download or any nested workers.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
        result = "complete" if proc.returncode == 0 else "failed"
        if proc.returncode and not reason:
            reason = f"worker exit {proc.returncode}; see {log.relative_to(self.out)}"
        row.update(status=result, reason=reason, exit_code=proc.returncode,
                   runtime_s=time.monotonic()-before, peak_process_tree_rss_mb=peak/1e6,
                   min_system_available_gib=available/2**30, free_disk_start_gib=disk_before/2**30,
                   free_disk_end_gib=shutil.disk_usage(self.out).free/2**30, log=str(log.relative_to(self.out)))
        write_csv(self.out / "run_evidence.csv", self.evidence)
        print(f"{name}: {result}, {row['runtime_s']:.2f} s {reason}", flush=True)
        return row

    def require(self, *args, **kwargs):
        result = self.worker(*args, **kwargs)
        if result["status"] != "complete":
            self.mark(args[0], result["status"], result["reason"])
            raise RuntimeError(result["reason"])
        self.mark(args[0], "complete")
        return result


def inputs(c, cache, nocache, archive):
    if not nocache and (cache / "data_validation.csv").exists() and all(
            (cache / f"day{day}.npz").exists() for day in range(1, 5)):
        # Recheck member digests on cache reuse. The sparse cache hashes are frozen below.
        evidence = c.require("data_verify", ["steps.py", "source", cache, *([archive] if archive else [])])
    else:
        evidence = c.require("data_parse", ["data.py", "--cache", cache,
                            *(["--archive", archive] if archive else [])], limit=1500)
    shutil.copy2(cache / "data_validation.csv", c.out / "data_validation.csv")
    write_json(c.out / "parsed_input_hashes.json",
               {f"day{day}.npz": sha256(cache / f"day{day}.npz") for day in range(1, 5)})
    return evidence


def budgets_for(out, expanded=False):
    import numpy as np

    workloads = load(out / "workloads.json")["workloads"]
    max_function = 0
    for spec in workloads.values():
        with np.load(out / spec["path"]) as data:
            events = data["events"]
            if len(events):
                max_function = max(max_function, int(events[:, 2].max()))
    working_sets = [v["working_set_mb"] for v in workloads.values() if v["working_set_mb"] > 0]
    middle = sorted(working_sets)[len(working_sets)//2]
    low = max(max_function, math.ceil(.14 * middle / 100) * 100)
    high = max(low + 200, math.ceil(.9 * max(working_sets) / 100) * 100)
    paper = [b for b in (1000, 4000, 10000, 22000, 40000, 79000) if low <= b <= high]
    if expanded:
        low = max(max_function, min(low, 1000))
        targets = np.geomspace(low, high, num=5)
        paper = [b for b in (1000, 4000, 10000, 22000, 40000, 79000) if low <= b <= high]
    else:
        targets = (low, math.sqrt(low * high), high)
    budgets = sorted(set(int(round(x / 100)) * 100 for x in targets))
    budgets[0] = max(max_function, budgets[0])
    # Retain a nearby paper point where it still spans the chosen working-set range.
    for i, target in enumerate(budgets.copy()):
        if paper:
            near = min(paper, key=lambda b: abs(b-target))
            if abs(near-target) <= .17 * target and near not in budgets:
                budgets[i] = near
    if low < 22000 < workloads["Representative"]["working_set_mb"]:
        if expanded:
            budgets.append(22000)
        elif 22000 <= high:
            budgets[-1] = 22000
    budgets = sorted(set(budgets))
    if len(budgets) < 3:
        budgets.append(max(budgets)+100)
    write_json(out / "budgets.json", dict(budgets_mb=budgets, max_footprint_mb=max_function,
              working_sets_mb=working_sets,
              method=("geometric grid from 1 GB (at least the largest footprint) to 90% of largest active set, "
                      "snap nearby paper points, add 22 GB when below Representative's active footprint"
                      if expanded else
                      "low: max footprint or 14% median active working set; high: 90% largest; geometric middle; snap nearby paper points"),
              diagnostic_budget_mb=22000 if 22000 in budgets else budgets[len(budgets)//2]))
    return budgets


def replay(c, stage, source, workload, policy, budget, target, model=None, predictions=None,
           checkpoint_state=None):
    meta = load(source / "workloads.json")["workloads"][workload]
    args = ["replay.py", "--inputs", source / meta["path"], "--policy", policy,
            "--budget", budget, "--output", target]
    if model is not None:
        args.extend(["--checkpoint", model])
    if predictions is not None:
        args.extend(["--prediction", predictions])
    if checkpoint_state is not None:
        args.extend(["--checkpoint-state", checkpoint_state])
    evidence = c.worker(stage, args)
    settings = dict(run_id=stage, workload=workload, policy=policy, budget_mb=budget,
                    scenario="historical_max_minus_average_ms", seed=SEED, status=evidence["status"],
                    reason=evidence["reason"], workload_sha256=meta["sha256"],
                    predictor="none", prediction_sha256="")
    if predictions is not None:
        settings.update(predictor="offline_reduced_LACE_full", prediction_sha256=sha256(predictions))
    if model is not None:
        variant = load(model.with_name("model.json"))["variant"]
        settings.update(predictor=f"offline_reduced_LACE_{variant}", model_sha256=sha256(model))
    return {**settings, **(load(target) if evidence["status"] == "complete" else {})}


def pilot(c, cache, parsed):
    source = c.out / "pilot"
    prep = c.require("pilot_prepare", ["steps.py", "prepare", cache, source, 3000, 5000, ".12"])
    fit = c.require("pilot_train", ["steps.py", "train", source, source / "full", 2, "full"])
    inference = c.require("pilot_predict", ["steps.py", "predict", source, source / "full" / "model.pt"])
    c.require("toy_checks", ["replay.py", "--toy"])
    info = load(source / "workloads.json")["workloads"]["Representative"]
    budget = max(350, budgets_for(source)[0])
    rows = []
    for policy in POLICIES:
        target = source / f"smoke_{policy.split()[0]}.json"
        pred = source / "representative_predictions.npz" if policy == "LACE" else None
        row = replay(c, f"smoke_{policy.split()[0]}", source, "Representative", policy, budget, target,
                     predictions=pred)
        assert row["status"] == "complete"
        assert row["request_count"] == info["invocations"] and row["workload_sha256"] == info["sha256"]
        assert row["cold_invocations"] + row["warm_invocations"] == row["request_count"]
        assert row["peak_logical_mb"] <= budget
        rows.append(row)
    target = source / "smoke_live_LACE.json"
    live = replay(c, "smoke_live_LACE", source, "Representative", "LACE", budget, target,
                  model=source / "full" / "model.pt")
    assert live["status"] == "complete" and live["cold_invocations"] == rows[0]["cold_invocations"]
    assert (target.with_suffix(".decisions.csv")).exists()
    write_csv(source / "smoke_runs.csv", rows + [live])
    replay_s = sum(row["runtime_s"] for row in rows)
    projection = parsed["runtime_s"] + 4*prep["runtime_s"] + 150*fit["runtime_s"] + 12*inference["runtime_s"] + 170*replay_s + 24*live["runtime_s"] + 180
    remaining = c.heavy - time.monotonic()
    write_json(c.out / "pilot_projection.json", dict(
               parsed_or_verified_s=parsed["runtime_s"], pilot_prep_s=prep["runtime_s"],
               pilot_training_s=fit["runtime_s"], pilot_batched_inference_s=inference["runtime_s"],
               pilot_replay_s=replay_s, pilot_live_lace_replay_s=live["runtime_s"],
               projected_conservative_full_s=projection, remaining_heavy_s=remaining,
               notes="Inflated scaling from pilot phase times includes process startup and optional diagnostics"))
    c.mark("pilot", "complete")
    return projection, remaining


def persist_runs(out, rows):
    write_csv(out / "runs.csv", rows)
    accounting = ("run_id", "workload", "policy", "budget_mb", "scenario", "status", "request_count",
                  "warm_invocations", "cold_invocations", "prewarm_events", "unadmitted", "oversized",
                  "excluded", "unresolved", "peak_logical_mb", "workload_sha256")
    write_csv(out / "replay_counts.csv", [{k: row.get(k, "") for k in accounting} for row in rows])


def measured(c, cache, profile, expanded=False):
    examples, cap, factor, epochs = profile
    out = c.out
    c.require("measured_prepare", ["steps.py", "prepare", cache, out, cap, examples, factor,
                                  *([25000] if expanded else [])])
    budgets = budgets_for(out, expanded)
    c.require("train_full", ["steps.py", "train", out, out / "full", epochs, "full"], limit=5400)
    shutil.copy2(out / "full" / "training_history.csv", out / "training_history.csv")
    checkpoint = out / "full" / "model.pt"
    c.require("predict_full", ["steps.py", "predict", out, checkpoint])
    combinations = [(w, p, b) for w in WORKLOADS for b in budgets for p in POLICIES]
    planned = [dict(workload=w, policy=p, budget_mb=b, status="planned",
                    reason="run not started", scenario="historical_max_minus_average_ms", seed=SEED)
               for w, p, b in combinations]
    persist_runs(out, planned)
    for i, (workload, policy, budget) in enumerate(combinations):
        tag = f"{workload.lower()}_{budget}_{policy.split()[0]}"
        pred = out / f"{workload.lower()}_predictions.npz" if policy == "LACE" else None
        row = replay(c, tag, out, workload, policy, budget, out / "replays" / f"{tag}.json", predictions=pred)
        if pred:
            row["model_sha256"] = sha256(checkpoint)
        planned[i] = row
        persist_runs(out, planned)
        c.mark(tag, row["status"], row["reason"])
    for figure in (8, 9, 10, 12):
        c.mark(f"figure_{figure:02}", "pending_export")
    c.require("export_main", ["export.py", out], heavy=False, limit=180)
    for figure in (8, 9, 10, 12):
        c.mark(f"figure_{figure:02}", "complete")
    diagnostic_budget = load(out / "budgets.json")["diagnostic_budget_mb"]
    resources = []
    for policy in POLICIES:
        name = "resource_" + policy.split()[0]
        row = replay(c, name, out, "Representative", policy, diagnostic_budget,
                     out / "replays" / f"{name}.json", model=checkpoint if policy == "LACE" else None)
        if row["status"] == "complete" and row["comparable"]:
            expected = next(r for r in planned if r["workload"] == "Representative" and
                            r["policy"] == policy and r["budget_mb"] == diagnostic_budget)
            assert row["cold_invocations"] == expected["cold_invocations"]
            resources.append({**row, "label": policy})
    if resources:
        write_csv(out / "resource_runs.csv", resources)
    for figure in (13, 14):
        c.mark(f"figure_{figure:02}", "complete" if len(resources) == len(POLICIES) else "partial",
               "" if len(resources) == len(POLICIES) else "See resource worker statuses")
    variant = "wo_cls"
    variant_path = out / variant
    ablation_fit = c.worker("train_wo_cls", ["steps.py", "train", out, variant_path, epochs, variant], limit=5400)
    if ablation_fit["status"] == "complete":
        ablation = replay(c, "ablation_wo_cls", out, "Representative", "LACE", diagnostic_budget,
                           out / "replays" / "ablation_wo_cls.json", model=variant_path / "model.pt")
        matched = [r for r in planned if r["workload"] == "Representative" and r["budget_mb"] == diagnostic_budget
                   and r["policy"] in ("FaasCache", "LACE") and r["status"] == "complete"]
        if len(matched) == 2 and ablation["status"] == "complete":
            write_csv(out / "ablation.csv", [{**r, "label": r["policy"]} for r in matched] +
                      [{**ablation, "label": "LACE wo_cls"}])
            c.mark("figure_11", "complete")
        else:
            c.mark("figure_11", "skipped", "A matched replay was unavailable")
    else:
        c.mark("figure_11", "skipped", ablation_fit["reason"])
    table = c.worker("iat_metrics", ["steps.py", "table", out], limit=180)
    c.mark("table_03", table["status"], table["reason"])
    c.require("export_final", ["export.py", out], heavy=False, limit=180)
    c.mark("campaign", "complete" if all(r["status"] == "complete" for r in planned) else "partial")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--reparse", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--expanded", action="store_true",
                        help="After the pilot, use the longer-window sensitivity campaign when time permits")
    args = parser.parse_args()
    deadline = args.deadline if args.deadline is not None else time.monotonic() + 21600
    c = Campaign(args.output, deadline)
    sources = [*ROOT.glob("*.py"), ROOT / "requirements.txt", ROOT / "vendor/faascache/gdsf.py"]
    write_json(c.out / "provenance.json", dict(python=sys.version, seed=SEED, command=shlex.join(sys.argv),
               host=os.uname().sysname, machine=os.uname().machine, clock_start_monotonic=c.start,
               heavy_cutoff_monotonic=c.heavy, deadline_monotonic=deadline,
               source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sources}))
    try:
        c.mark("HIST", "substituted", "Selected LRU before comparisons; no verified drop-in HIST implementation")
        parsed = inputs(c, args.cache, args.reparse, args.archive)
        estimate, remaining = pilot(c, args.cache, parsed)
        if not args.smoke:
            profile = (50000, 30000, 1., 18) if estimate < .65*remaining else (20000, 12000, .5, 8)
            expanded = args.expanded and 2.5*estimate < .65*remaining
            if expanded:
                profile = (100000, 100000, 1., 18)
            write_json(c.out / "profile.json", dict(example_cap=profile[0], request_cap=profile[1],
                      function_cap_factor=profile[2], epochs=profile[3], train_threads=4,
                      policies=POLICIES, prewarm=False, online_training=False,
                      expanded_campaign=expanded,
                      historical_activity_limit=25000 if expanded else None))
            measured(c, args.cache, profile, expanded)
    except Exception as err:
        c.mark("campaign", "partial", str(err))
        if (c.out / "runs.csv").exists():
            c.worker("export_recovery", ["export.py", c.out], heavy=False)
        raise
    finally:
        write_json(c.out / "elapsed.json", dict(elapsed_s=time.monotonic() - c.start))


if __name__ == "__main__":
    main()
