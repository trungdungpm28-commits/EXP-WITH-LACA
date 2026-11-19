from collections import Counter
import math

import numpy as np

from common import DAY, FEATURES, SEED, WORKLOADS, atomic, sha256, write_csv, write_json
from data import TraceDay, read_stats

TRIGGERS = ("http", "timer", "queue", "event", "storage", "orchestration", "other")
SPLIT = int(2.7 * DAY)
FUNCTION_CAPS = dict(zip(WORKLOADS, (400, 1000, 200)))
HALF_LIVES = 2.0 ** np.arange(9, 19)
CLASS_EDGES = np.array([60., 600., 3600.])


def static_features(trace, key, duration, memory):
    trigger = str(trace.kind(key)).lower()
    category = trigger if trigger in TRIGGERS else "other"
    average, maximum = duration
    return np.r_[np.log1p([average, maximum - average, memory]),
                 [float(t == category) for t in TRIGGERS]]


def features(times, indexes, static):
    """Observe every invocation in order, returning features at selected positions."""
    gaps = np.r_[np.inf, np.diff(times)]
    result = np.empty((len(indexes), FEATURES), dtype=np.float32)
    if not len(indexes):
        return result
    for lag in range(8):
        available = indexes >= lag + 1
        result[:, lag] = -1
        result[available, lag] = np.log1p(gaps[indexes[available] - lag])
    decay = np.exp2(-gaps[:, None] / HALF_LIVES)
    counters = np.ones(len(HALF_LIVES))
    selected = 0
    for i, multiplier in enumerate(decay):
        counters = 1 + counters * multiplier
        if i == indexes[selected]:
            result[selected, 8:18] = np.log1p(counters)
            selected += 1
            if selected == len(indexes):
                break
    result[:, 18:28] = static
    recent = indexes + 1 - np.searchsorted(times, times[indexes] - 1800)
    result[:, 28:] = np.c_[np.log1p(indexes + 1), np.log1p(recent), recent / (indexes + 1)]
    return result


def save_arrays(path, **arrays):
    def save(p):
        with p.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
    atomic(path, save)


def historical(cohorts, stats):
    counts = Counter()
    for trace in cohorts:
        counts.update(trace.totals)
    population = {key: count for key, count in counts.items() if count >= 2}
    by_app = Counter(key[:2] for key in population)
    durations = {}
    memory = {}
    for durs, mem in stats:
        durations.update(durs)
        memory.update(mem)
    eligible = {key: count for key, count in population.items()
                if key in durations and key[:2] in memory}
    sizes = {key: math.ceil(memory[key[:2]] / by_app[key[:2]]) for key in eligible}
    return eligible, durations, sizes, {"eligible_before_metadata": len(population),
                                       "excluded_missing_historical_metadata": len(population) - len(eligible)}


def choose(eligible, seed, factor, historical_activity_limit=None):
    rng = np.random.default_rng(seed)
    ranked = sorted(eligible, key=lambda key: (eligible[key], key))
    capped = [key for key in ranked if historical_activity_limit is None or
              eligible[key] <= historical_activity_limit]

    def sample(candidates, n):
        positions = rng.choice(len(candidates), min(len(candidates), n), replace=False)
        return [candidates[i] for i in positions]

    sizes = {k: max(4, int(v * factor)) for k, v in FUNCTION_CAPS.items()}
    groups = {}
    strata = {}
    for index, indexes in enumerate(np.array_split(np.arange(len(capped)), 4), 1):
        cohort = sample([capped[i] for i in indexes], sizes["Representative"] // 4)
        strata.update({key: index for key in cohort})
        groups.setdefault("Representative", []).extend(cohort)
    groups["Rare"] = ranked[:sizes["Rare"]]
    groups["Random"] = sample(capped, sizes["Random"])
    return groups, strata


def build_training(traces, stats, out, max_examples, seed):
    # Day 1 is the metadata and history calibration period. Sample functions from day 1 only.
    eligible, duration, sizes, _ = historical(traces[:1], stats[:1])
    candidates = sorted(key for key, n in eligible.items() if n <= 25000)
    rng = np.random.default_rng(seed)
    choose_idx = rng.choice(len(candidates), min(1000, len(candidates)), replace=False)
    keys = [candidates[i] for i in choose_idx]
    histories = []
    counts = [0, 0]
    cap_start = DAY
    for key in keys:
        days = [t.arrivals(key) for t in traces[:3]]
        times = np.concatenate(days)
        start = np.searchsorted(times, cap_start)
        split = np.searchsorted(times, SPLIT)
        bounds = ((start, max(start, split - 1)), (split, max(split, len(times) - 1)))
        counts = [n + hi - lo for n, (lo, hi) in zip(counts, bounds)]
        if any(hi > lo for lo, hi in bounds):
            histories.append((key, times, bounds))
    ntrain = int(max_examples * .9)
    selected = [np.sort(rng.choice(n, min(n, cap), replace=False)) for n, cap in
                zip(counts, [ntrain, max_examples - ntrain])]
    arrays = [[], []]
    labels = [[], []]
    example_times = [[], []]
    offsets = [0, 0]
    for key, times, bounds in histories:
        positions = []
        for split, (lo, hi) in enumerate(bounds):
            old = offsets[split]
            offsets[split] += hi - lo
            i, j = selected[split].searchsorted([old, offsets[split]])
            positions.append(selected[split][i:j] - old + lo)
        inputs = features(times, np.concatenate(positions), static_features(traces[0], key, duration[key], sizes[key]))
        for split, inputs_split in enumerate(np.split(inputs, [len(positions[0])])):
            indexes = positions[split]
            arrays[split].append(inputs_split)
            labels[split].append(times[indexes + 1] - times[indexes])
            example_times[split].append(times[indexes])
    joined_x = [np.concatenate(items) for items in arrays]
    joined_y = [np.concatenate(items) for items in labels]
    joined_t = [np.concatenate(items) for items in example_times]
    assert (joined_t[0] + joined_y[0] < SPLIT).all()
    assert (joined_t[1] >= SPLIT).all() and (joined_t[1] + joined_y[1] < 3 * DAY).all()
    save_arrays(out / "fit.npz", train_x=joined_x[0], val_x=joined_x[1],
                train_y=joined_y[0], val_y=joined_y[1], train_t=joined_t[0], val_t=joined_t[1],
                thresholds=CLASS_EDGES)
    write_json(out / "training_inputs.json", dict(seed=seed, training_functions=len(histories),
               candidate_examples=list(map(int, counts)), train_examples=len(joined_y[0]), val_examples=len(joined_y[1]),
               train_start_s=DAY, validation_start_s=SPLIT, validation_end_s=3 * DAY,
               training_file_sha256=sha256(out / "fit.npz")))


def build(cache, out, invocations, examples, seed=SEED, factor=1., historical_activity_limit=None):
    traces = [TraceDay(cache, day) for day in range(1, 5)]
    stats = [read_stats(cache, day) for day in range(1, 5)]
    build_training(traces, stats, out, examples, seed)
    eligible, durations, sizes, eligibility = historical(traces[:3], stats[:3])
    groups, strata = choose(eligible, seed, factor, historical_activity_limit)
    manifest = []
    workload_info = {}
    for workload, keys in groups.items():
        keys = sorted(keys)
        held = traces[3]
        counts = sum((held.minute_counts(key) for key in keys), start=np.zeros(1440, dtype=np.int64))
        end_minute = int(np.searchsorted(counts.cumsum(), invocations, side="right"))
        if end_minute < 1:
            raise ValueError(f"{workload} exceeds invocation cap in the first minute")
        with_requests = 0
        working_set = 0
        parts = []
        fit_parts = []
        targets = []
        for i, key in enumerate(keys):
            window = held.arrivals(key, end_minute)
            if len(window):
                with_requests += 1
                working_set += sizes[key]
                history = [t.arrivals(key) for t in traces[:3]]
                times = np.concatenate([*history, window])
                indexes = np.arange(len(times) - len(window), len(times))
                model_input = features(times, indexes, static_features(
                    next(t for t in reversed(traces[:3]) if key in t.index), key, durations[key], sizes[key]))
                eval_avg, _ = stats[3][0].get(key, durations[key])
                # FaasCache cost and model static inputs use historical metadata. Figure 8 uses
                # the same historical penalty across policies; day 4 average is outcome only.
                prior_avg, prior_max = durations[key]
                parts.append(np.c_[window, np.full(len(window), i), np.full(len(window), sizes[key]),
                                   np.full(len(window), eval_avg), np.full(len(window), prior_max - prior_avg)])
                fit_parts.append(model_input)
                targets.append(np.r_[np.diff(window), np.nan])
            manifest.append(dict(workload=workload, function_id="/".join(key), function_index=i,
                                 historical_invocations=eligible[key],
                                 frequency_stratum=strata.get(key, "") if workload == "Representative" else "",
                                 origin_day=4, window_start_s=3 * DAY, window_end_s=3 * DAY + 60 * end_minute,
                                 day4_total=held.totals.get(key, 0), replay_invocations=len(window),
                                 memory_mb=sizes[key], seed=seed, feature_prehistory="days 1–3",
                                 cache_initial_state="empty",
                                 duration_source="day 4" if key in stats[3][0] else "historical fallback"))
        events = np.concatenate(parts) if parts else np.empty((0, 5))
        inputs = np.concatenate(fit_parts) if parts else np.empty((0, FEATURES), dtype=np.float32)
        target = np.concatenate(targets) if parts else np.empty(0)
        order = np.lexsort((events[:, 1], events[:, 0]))
        path = out / f"{workload.lower()}.npz"
        save_arrays(path, events=events[order], features=inputs[order], target_s=target[order])
        assert len(events) == counts[:end_minute].sum() <= invocations
        workload_info[workload] = dict(path=path.name, sha256=sha256(path), functions=len(keys),
                                      active_functions=with_requests, invocations=len(events),
                                      working_set_mb=working_set, window_end_minute=end_minute)
    for row in manifest:
        row["replay_sha256"] = workload_info[row["workload"]]["sha256"]
    write_csv(out / "workload_manifest.csv", manifest)
    write_json(out / "workloads.json", dict(workloads=workload_info,
               representative_random_historical_activity_limit=historical_activity_limit, **eligibility))
