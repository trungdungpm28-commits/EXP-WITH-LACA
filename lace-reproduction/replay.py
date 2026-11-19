import argparse
from collections import Counter
import heapq
import json
import resource
import time
from pathlib import Path

import numpy as np

from common import POLICIES, sha256, write_csv, write_json
from diagnostic_policies import Histogram, POLICIES_EXTRA, oracle_next
from vendor.faascache.gdsf import priority as gdsf_priority


class Cache:
    def __init__(self, budget, policy):
        self.budget = budget
        self.policy = policy
        self.pool = {}
        self.queue = []
        self.clock = 0.
        self.used = 0
        self.peak = 0
        self.evictions = Counter()
        self.histogram = Histogram(self) if policy == "HIST" else None

    def remove(self, function, reason):
        size, *_ = self.pool.pop(function)
        self.used -= size
        self.evictions[reason] += 1

    def current_min(self):
        while self.queue:
            priority, serial, function = self.queue[0]
            entry = self.pool.get(function)
            if entry is not None and entry[3] == serial:
                return priority, serial, function
            heapq.heappop(self.queue)
        return None

    def request(self, serial, ts, function, size, penalty, predicted):
        if self.histogram:
            self.histogram.advance(ts)
        if self.policy == "TTL 10 min":
            head = self.current_min()
            while head and ts - head[0] > 600:
                self.remove(head[2], "expired")
                head = self.current_min()
        if size > self.budget:
            return "oversized", (), None
        old = self.pool.get(function)
        hit = old is not None
        victims = []
        if not hit:
            while self.used + size > self.budget:
                head = self.current_min()
                if head is None:
                    raise AssertionError("Pool has no victim despite insufficient capacity")
                priority, _, victim = head
                heapq.heappop(self.queue)
                self.remove(victim, "capacity")
                if self.policy in ("FaasCache", "LND"):
                    self.clock = priority
                victims.append(victim)
            self.used += size
        elif old[0] != size:
            raise ValueError("Footprint changed within a replay")
        frequency = old[1] + 1 if hit else 1
        if self.policy == "LACE":
            priority = 1. / (1. + float(predicted))
        elif self.policy == "FaasCache":
            priority = gdsf_priority(self.clock, frequency, penalty, 0., size)
        elif self.policy == "LND":
            priority = self.clock + penalty / size
        elif self.policy == "FREQ":
            priority = frequency
        elif self.policy == "SIZE":
            priority = -size
        elif self.policy == "Belady":
            priority = -int(predicted)
        else:
            priority = ts
        self.pool[function] = (size, frequency, priority, serial)
        heapq.heappush(self.queue, (priority, serial, function))
        if len(self.queue) > 4 * len(self.pool) + 64:
            self.queue = [(entry[2], entry[3], key) for key, entry in self.pool.items()]
            heapq.heapify(self.queue)
        self.peak = max(self.peak, self.used)
        if self.histogram:
            self.histogram.observe(serial, ts, function, size, penalty)
        assert 0 <= self.used <= self.budget
        return "warm" if hit else "cold", victims, priority


def run(inputs, policy, budget, model=None, prediction=None, sample=0, cost_multiplier=1.,
        checkpoint_state=None, model_hash=""):
    with np.load(inputs) as data:
        events = data["events"]
        x = data["features"] if model is not None else None
    cached = None
    if prediction:
        with np.load(prediction) as data:
            cached = data["pred_s"]
        if len(cached) != len(events) or not np.isfinite(cached).all():
            raise ValueError("Invalid cached predictions")
    if policy == "Belady":
        cached = oracle_next(events)
    if policy == "LACE" and cached is None and model is None:
        raise ValueError("LACE requires a trained predictor")
    cache = Cache(budget, policy)
    counts = Counter()
    samples = []
    cpu_ns = 0
    warm_reference_ms = cold_penalty_ms = 0.
    start = time.monotonic()
    previous_time = -1
    first = 0
    elapsed_replayed_s = 0.
    peak_previous_rss_mb = 0.
    trace_hash = sha256(inputs)
    resume_key = [trace_hash, model_hash, policy, budget, sample, cost_multiplier]
    if checkpoint_state is not None:
        if model is None or policy != "LACE":
            raise ValueError("Resume is supported for live LACE resource replays")
        if checkpoint_state.exists():
            state = json.loads(checkpoint_state.read_text())
            if state["resume_key"] != resume_key:
                raise ValueError("Replay checkpoint does not match inputs and settings")
            first = state["next_event"]
            counts.update(state["counts"])
            cache.pool = {int(k): tuple(v) for k, v in state["pool"].items()}
            cache.queue = [(e[2], e[3], k) for k, e in cache.pool.items()]
            heapq.heapify(cache.queue)
            cache.used, cache.peak = state["used"], state["peak"]
            cache.evictions.update(state["evictions"])
            samples = state["samples"]
            cpu_ns = state["cpu_ns"]
            elapsed_replayed_s = state["elapsed_replayed_s"]
            peak_previous_rss_mb = state["peak_previous_rss_mb"]
            warm_reference_ms, cold_penalty_ms = state["warm_reference_ms"], state["cold_penalty_ms"]
            previous_time = events[first - 1, 0]
            print(f"Resuming after {first:,}/{len(events):,} requests", flush=True)
    start = time.monotonic()
    for i, (ts, key, size, avg, penalty) in enumerate(events[first:], start=first):
        if ts < previous_time:
            raise ValueError("Trace is out of event order")
        previous_time = ts
        key, size = int(key), int(size)
        penalty *= cost_multiplier
        begin = time.process_time_ns()
        predicted = None
        if cached is not None:
            predicted = cached[i]
        elif model is not None:
            predicted = model.predict(x[i:i+1])[0][0]
        status, victims, priority = cache.request(i, ts, key, size, penalty, predicted)
        cpu_ns += time.process_time_ns() - begin
        counts[status] += 1
        warm_reference_ms += avg
        if status != "warm":
            cold_penalty_ms += penalty
        if i < sample or (len(samples) < 2 * sample and victims):
            samples.append(dict(index=i, timestamp_s=ts, function_index=key, event=status,
                                memory_mb=cache.used, priority=priority,
                                predicted_iat_s=float(predicted) if predicted is not None else None,
                                evictions="|".join(map(str, victims))))
        if checkpoint_state is not None and (i + 1) % 100000 == 0:
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e6
            write_json(checkpoint_state, dict(resume_key=resume_key, next_event=i+1, pool=cache.pool,
                       used=cache.used, peak=cache.peak, evictions=cache.evictions, counts=counts,
                       samples=samples, cpu_ns=cpu_ns,
                       elapsed_replayed_s=elapsed_replayed_s+time.monotonic()-start,
                       peak_previous_rss_mb=max(peak_previous_rss_mb, rss_mb),
                       warm_reference_ms=warm_reference_ms, cold_penalty_ms=cold_penalty_ms))
        if model is not None and (i + 1) % 250000 == 0:
            print(f"Live inference replay: {i + 1:,}/{len(events):,} "
                  f"requests, {time.monotonic() - start:.1f} s", flush=True)
    assert sum(counts.values()) == len(events)
    assert sum(entry[0] for entry in cache.pool.values()) == cache.used
    n = len(events)
    cold = counts["cold"] + counts["oversized"]
    hist = cache.histogram
    return dict(policy=policy, request_count=n, cold_invocations=cold, warm_invocations=counts["warm"],
                oversized=counts["oversized"], unadmitted=counts["oversized"], prewarm_events=hist.prewarms if hist else 0,
                prewarm_cost_ms=hist.charge if hist else 0., prewarm_rejected=hist.rejected if hist else 0, excluded=0,
                unresolved=0, capacity_evictions=cache.evictions["capacity"], ttl_evictions=cache.evictions["expired"],
                budget_mb=budget, peak_logical_mb=cache.peak, cold_penalty_ms=cold_penalty_ms,
                warm_reference_ms=warm_reference_ms, policy_e2e_ms=warm_reference_ms + cold_penalty_ms,
                cold_ratio_pct=100 * cold / n if n else None,
                overhead_pct=100 * cold_penalty_ms / warm_reference_ms if warm_reference_ms else None,
                overhead_missing_reason="" if warm_reference_ms else "zero warm reference",
                comparable=counts["oversized"] == 0 and n > 0,
                runtime_s=elapsed_replayed_s + time.monotonic() - start,
                policy_cpu_ms_per_request=cpu_ns / n / 1e6 if n else None,
                peak_worker_rss_mb=max(peak_previous_rss_mb, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e6),
                workload_sha256=trace_hash), samples


def toy():
    cache = Cache(4, "FaasCache")
    assert cache.request(0, 0, 0, 2, 10, None)[2] == 5
    assert cache.request(1, 1, 1, 2, 8, None)[2] == 4
    assert cache.request(2, 2, 0, 2, 10, None)[2] == 10
    status, victims, priority = cache.request(3, 3, 2, 3, 9, None)
    assert (status, victims, priority) == ("cold", [1, 0], 13)
    assert cache.used == 3
    assert cache.request(4, 4, 3, 5, 10, None)[0] == "oversized"
    assert cache.used == 3
    cache = Cache(2, "TTL 10 min")
    assert cache.request(0, 0, 0, 1, 1, None)[0] == "cold"
    assert cache.request(1, 600, 0, 1, 1, None)[0] == "warm"
    assert cache.request(2, 1201, 0, 1, 1, None)[0] == "cold"
    assert cache.evictions["expired"] == 1
    cache = Cache(2, "LACE")
    cache.request(0, 0, 0, 1, 1, 10)
    cache.request(1, 1, 1, 1, 1, 30)
    assert cache.request(2, 2, 2, 1, 1, 2)[1] == [1]
    cache.request(3, 3, 0, 1, 1, 60)
    assert cache.request(4, 4, 1, 1, 1, 5)[1] == [0]
    cache = Cache(2, "LRU")
    cache.request(0, 0, 0, 1, 1, None)
    cache.request(1, 0, 1, 1, 1, None)
    assert cache.request(2, 0, 2, 1, 1, None)[1] == [0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--policy", choices=(*POLICIES, *POLICIES_EXTRA))
    parser.add_argument("--budget", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--checkpoint-state", type=Path)
    parser.add_argument("--sample", type=int, default=20)
    args = parser.parse_args()
    if args.toy:
        toy()
        print("Toy replay passed")
    else:
        model = None
        if args.checkpoint:
            from model import Predictor, configure
            configure(1)
            model = Predictor(args.checkpoint)
        state_path = args.checkpoint_state
        if state_path is None and args.checkpoint is not None:
            state_path = args.output.with_suffix(".checkpoint.json")
        result, decisions = run(args.inputs, args.policy, args.budget, model, args.prediction, args.sample,
                                checkpoint_state=state_path,
                                model_hash=sha256(args.checkpoint) if state_path else "")
        write_csv(args.output.with_suffix(".decisions.csv"), decisions)
        write_json(args.output, result)
