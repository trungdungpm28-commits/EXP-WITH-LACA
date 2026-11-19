"""Independently check campaign accounting, causality and replay reproducibility."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from common import DAY, POLICIES, ROOT, WORKLOADS, sha256, write_json
from data import TraceDay, read_stats
from replay import run
from workloads import SPLIT, features, historical, static_features


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def audit(out, cache):
    metadata = json.loads((out / "workloads.json").read_text())["workloads"]
    manifest = rows(out / "workload_manifest.csv")
    runs = rows(out / "runs.csv")
    histories = [TraceDay(cache, day) for day in range(1, 4)]
    stats = [read_stats(cache, day) for day in range(1, 4)]
    _, durations, sizes, _ = historical(histories, stats)
    results = {}
    for workload in WORKLOADS:
        spec = metadata[workload]
        events_file = out / spec["path"]
        assert sha256(events_file) == spec["sha256"]
        with np.load(events_file) as data:
            events, model_input, target = data["events"], data["features"], data["target_s"]
        assert np.all(events[:, 0] >= 3*DAY)
        assert np.all(events[:, 0] < 3*DAY + 60*spec["window_end_minute"])
        assert np.lexsort((events[:, 1], events[:, 0])).tolist() == list(range(len(events)))
        records = [r for r in manifest if r["workload"] == workload]
        # Check every manifest count and within-function label, plus timestamp-prefix features
        # on a bounded deterministic sample spanning the historical activity ranks.
        active = []
        for row in records:
            positions = np.flatnonzero(events[:, 1] == int(row["function_index"]))
            assert len(positions) == int(row["replay_invocations"])
            if len(positions):
                active.append((row, positions))
                assert np.all(events[positions, 2] == int(row["memory_mb"]))
                assert np.array_equal(target[positions[:-1]], np.diff(events[positions, 0]))
                assert np.isnan(target[positions[-1]])
        sampled = 0
        for choice in np.linspace(0, len(active)-1, min(10, len(active)), dtype=int):
            row, positions = active[choice]
            key = tuple(row["function_id"].split("/"))
            check = positions[(len(positions)-1)//2]
            prefix = np.concatenate([*(t.arrivals(key) for t in histories),
                                     events[positions[positions <= check], 0]])
            history_source = next(t for t in reversed(histories) if key in t.index)
            expected = features(prefix, np.array([len(prefix)-1]),
                                static_features(history_source, key, durations[key], sizes[key]))
            np.testing.assert_array_equal(expected[0], model_input[check])
            sampled += 1
        completed = [r for r in runs if r["workload"] == workload and r["status"] == "complete"]
        for row in completed:
            assert row["workload_sha256"] == spec["sha256"]
            assert int(row["request_count"]) == len(events)
            assert int(row["warm_invocations"]) + int(row["cold_invocations"]) == len(events)
            assert math.isclose(float(row["warm_reference_ms"]), float(events[:, 3].sum()))
            assert int(row["peak_logical_mb"]) <= int(row["budget_mb"])
        results[workload] = dict(manifest_entries=len(records), event_rows=len(events),
                                 timestamp_prefix_feature_checks=sampled, completed_runs=len(completed))
    with np.load(out / "fit.npz") as fit:
        assert np.max(fit["train_t"] + fit["train_y"]) < SPLIT
        assert np.min(fit["val_t"]) >= SPLIT
        assert np.max(fit["val_t"] + fit["val_y"]) < 3*DAY
    # Rerun each Representative policy at the smallest saved capacity and compare
    # both result aggregates and sampled decision order with the actual saved runs.
    cap = min(int(r["budget_mb"]) for r in runs)
    verification_fields = ("cold_invocations", "warm_invocations", "peak_logical_mb",
                           "capacity_evictions", "ttl_evictions", "warm_reference_ms", "policy_e2e_ms")
    repeats = 0
    for policy in POLICIES:
        existing = next(r for r in runs if r["workload"] == "Representative" and
                        r["policy"] == policy and int(r["budget_mb"]) == cap)
        assert existing["status"] == "complete"
        result, decisions = run(out / "representative.npz", policy, cap, sample=20,
                                prediction=out / "representative_predictions.npz" if policy == "LACE" else None)
        for field in verification_fields:
            assert float(existing[field]) == result[field]
        saved = rows(out / "replays" / f"{existing['run_id']}.decisions.csv")
        assert len(saved) == len(decisions)
        for old, new in zip(saved, decisions):
            assert int(old["index"]) == new["index"]
            assert old["event"] == new["event"] and old["evictions"] == new["evictions"]
            assert int(old["memory_mb"]) == new["memory_mb"]
        repeats += 1
    figures = []
    for path in sorted((out / "figures").glob("*.png")):
        assert (out / f"{path.stem}.csv").is_file()
        with Image.open(path) as im:
            im.load()
            figures.append(dict(file=path.name, width=im.width, height=im.height, sha256=sha256(path)))
    write_json(out / "audit.json", dict(status="passed", workloads=results,
                training_labels_within_splits=True, deterministic_replays=repeats,
                deterministic_replay_budget_mb=cap, figures=figures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    args = parser.parse_args()
    audit(args.out, args.cache)
