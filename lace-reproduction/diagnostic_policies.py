"""Additional replay policies for Figures 13–14."""
from collections import Counter, defaultdict
import heapq
import math

POLICIES_EXTRA = ("Belady", "HIST", "SIZE", "LND", "FREQ")
RESOURCE_POLICIES = ("LACE", "Belady", "FaasCache", "TTL 10 min", "LRU", "HIST", "SIZE", "LND", "FREQ")


class Histogram:
    def __init__(self, cache):
        self.cache = cache
        self.states = {}
        self.scheduled = []
        self.expirations = []
        self.charge = 0.
        self.prewarms = self.rejected = 0

    def advance(self, ts):
        while self.scheduled or self.expirations:
            expiration = self.expirations[0] if self.expirations else (math.inf,)
            warming = self.scheduled[0] if self.scheduled else (math.inf,)
            if expiration[0] < ts and expiration[0] <= warming[0]:
                _, serial, key = heapq.heappop(self.expirations)
                entry = self.cache.pool.get(key)
                if entry is not None and entry[3] == serial:
                    self.cache.remove(key, "expired")
            elif warming[0] <= ts:
                at, serial, key, size, penalty, expires = heapq.heappop(self.scheduled)
                if self.states[key][0] != serial or key in self.cache.pool:
                    continue
                if self.cache.used + size > self.cache.budget:
                    self.rejected += 1
                    continue
                self.cache.used += size
                self.cache.peak = max(self.cache.peak, self.cache.used)
                # Separate token from the request that scheduled this event.
                token = -serial-1
                self.cache.pool[key] = (size, 0, at, token)
                heapq.heappush(self.cache.queue, (at, token, key))
                heapq.heappush(self.expirations, (expires, token, key))
                self.prewarms += 1
                self.charge += penalty
            else:
                break

    def observe(self, serial, ts, key, size, penalty):
        state = self.states.get(key)
        hist, count, total, total_sq = Counter(), 0, 0, 0
        if state is not None:
            _, prev, hist, count, total, total_sq = state
            minute = int((ts-prev)//60)
            if minute < 240:
                hist[minute] += 1
                count += 1
                total += minute
                total_sq += minute**2
        self.states[key] = serial, ts, hist, count, total, total_sq
        if key not in self.cache.pool:
            return
        ttl = 7200.
        mean = total/count if count else 0.
        std = math.sqrt(max(0., total_sq/count-mean**2)) if count else math.inf
        if count >= 3 and (mean == 0 or std/mean <= 2):
            seen, head, tail = 0, None, None
            for minute, mass in sorted(hist.items()):
                seen += mass
                if head is None and seen >= .05*count:
                    head = minute
                if seen >= .99*count:
                    tail = minute
                    break
            ttl = max(60., 1.1*(tail+1)*60)
            prewarm = .9*head*60
            if prewarm > 0:
                self.cache.remove(key, "prewarm_release")
                heapq.heappush(self.scheduled, (ts+prewarm, serial, key, size, penalty, ts+ttl))
                return
        heapq.heappush(self.expirations, (ts+ttl, serial, key))
        # Bound stale expiration/prewarm records on bursty streams.
        if len(self.expirations) > 4*len(self.cache.pool)+512:
            self.expirations = [e for e in self.expirations
                                if e[2] in self.cache.pool and self.cache.pool[e[2]][3] == e[1]]
            heapq.heapify(self.expirations)


def oracle_next(events):
    import numpy as np
    positions = np.empty(len(events), dtype=np.int64)
    next_seen = {}
    for i in range(len(events)-1, -1, -1):
        key = int(events[i, 1])
        positions[i] = next_seen.get(key, len(events))
        next_seen[key] = i
    return positions
