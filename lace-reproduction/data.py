import argparse
import math
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import DAY, ROOT, atomic, sha256, write_csv, write_json

SOURCE = "https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md"
URL = "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz"
DIGEST = "aff8b3ca7240a41a109e4ee598e0a96e45fcb92e7b8395ac19cb3748cd260d89"
ARCHIVE = URL.rsplit("/", 1)[1]
KINDS = ("invocations_per_function_md", "function_durations_percentiles", "app_memory_percentiles")
KEY = ["HashOwner", "HashApp", "HashFunction"]
BUCKETS = [str(i) for i in range(1, 1441)]


def name(kind, day):
    return f"{kind}.anon.d{day:02}.csv"


def acquire(cache, archive):
    needed = {name(kind, day) for kind in KINDS for day in range(1, 5)}
    source = cache / "source"
    source.mkdir(parents=True, exist_ok=True)
    if archive is None:
        nearby = ROOT.parent.parent / "DATASET"
        archive = next((p for p in [cache / ARCHIVE, *nearby.glob("*.tar.xz")]
                        if p.is_file() and p.stat().st_size == 142968140 and sha256(p) == DIGEST), None)
    if archive is None:
        archive = cache / ARCHIVE
        partial = archive.with_suffix(".partial")
        subprocess.run(["curl", "--fail", "--location", "--retry", "2",
                        "--connect-timeout", "15", "--max-time", "600",
                        "--continue-at", "-", "--output", str(partial), URL], check=True)
        if sha256(partial) != DIGEST:
            raise ValueError(f"Microsoft archive checksum mismatch: {partial}")
        partial.replace(archive)
    if sha256(archive) != DIGEST:
        raise ValueError(f"Microsoft archive checksum mismatch: {archive}")
    receipt = source / "sha256.json"
    if receipt.is_file():
        import json
        hashes = json.loads(receipt.read_text())
        if needed <= set(hashes) and all((source / f).is_file() and
                                         sha256(source / f) == hashes[f] for f in needed):
            return archive, hashes
    # Read only known regular files. Never recreate archived paths or permissions.
    with tarfile.open(archive, "r|xz") as tar:
        for member in tar:
            if member.name in needed and member.isfile():
                def save(p):
                    with tar.extractfile(member) as src, p.open("wb") as out:
                        shutil.copyfileobj(src, out)
                atomic(source / member.name, save)
                needed.remove(member.name)
                if not needed:
                    break
    if needed:
        raise ValueError(f"Missing archive members: {sorted(needed)}")
    hashes = {name(k, d): sha256(source / name(k, d)) for k in KINDS for d in range(1, 5)}
    write_json(receipt, hashes)
    return archive, hashes


def parse(cache, archive=None):
    archive, hashes = acquire(cache, archive)
    evidence = []
    for day in range(1, 5):
        for kind in KINDS:
            fn = name(kind, day)
            path = cache / "source" / fn
            receipt = dict(day=day, file=fn, sha256=hashes[fn], archive_sha256=DIGEST,
                           archive_path=str(archive.resolve()), source=SOURCE, download=URL)
            if kind == KINDS[0]:
                sparse(cache, path, receipt)
            else:
                metadata(cache, path, kind, receipt)
            evidence.append(receipt)
    write_csv(cache / "data_validation.csv", evidence)


def sparse(cache, path, receipt):
    keys = []
    trigger = []
    cols = []
    counts = []
    indptr = [np.array([0], dtype=np.int64)]
    seen = set()
    invalid = duplicates = rows = 0
    schema = KEY + ["Trigger"] + BUCKETS
    if pd.read_csv(path, nrows=0).columns.tolist() != schema:
        raise ValueError(f"Unexpected invocation schema: {path}")
    types = dict.fromkeys(BUCKETS, "int64")
    for chunk in pd.read_csv(path, dtype=types, chunksize=512):
        rows += len(chunk)
        matrix = chunk[BUCKETS].to_numpy()
        valid = (matrix >= 0).all(axis=1) & chunk[KEY + ["Trigger"]].notna().all(axis=1).to_numpy()
        invalid += int((~valid).sum())
        unique = valid.copy()
        for index, values in enumerate(chunk[KEY].itertuples(index=False, name=None)):
            if not valid[index]:
                continue
            if values in seen:
                duplicates += 1
                unique[index] = False
            else:
                seen.add(values)
                keys.append(values)
        trigger.extend(chunk["Trigger"].to_numpy()[unique])
        matrix = matrix[unique]
        row, col = np.nonzero(matrix)
        cols.append(col.astype(np.int16))
        counts.append(matrix[row, col])
        indptr.append(indptr[-1][-1] + np.cumsum(np.bincount(row, minlength=len(matrix))))
    receipt.update(rows=rows, invalid=invalid, duplicates=duplicates, accepted=len(keys),
                   schema="HashOwner,HashApp,HashFunction,Trigger,1..1440 integer counts")
    target = cache / f"day{receipt['day']}.npz"

    def save(p):
        with p.open("wb") as stream:
            np.savez_compressed(stream, keys=np.asarray(keys, dtype="U64"),
                                trigger=np.asarray(trigger, dtype="U24"),
                                cols=np.concatenate(cols), counts=np.concatenate(counts),
                                ptr=np.concatenate(indptr))

    atomic(target, save)


def metadata(cache, path, kind, receipt):
    columns = KEY + ["Average", "Maximum"] if kind == KINDS[1] else KEY[:2] + ["AverageAllocatedMb"]
    frame = pd.read_csv(path, usecols=columns)[columns]
    valid = frame.notna().all(axis=1)
    for col in columns:
        if col not in KEY:
            valid &= np.isfinite(frame[col]) & (frame[col] >= 0)
    if kind == KINDS[1]:
        valid &= frame.Maximum >= frame.Average
    else:
        valid &= frame.AverageAllocatedMb > 0
    clean = frame[valid].drop_duplicates(columns[:-2] if kind == KINDS[1] else columns[:-1])
    receipt.update(rows=len(frame), accepted=len(clean), invalid=int((~valid).sum()),
                   duplicates=int(valid.sum()) - len(clean), schema=",".join(columns))
    atomic(cache / path.name, lambda p: clean.to_csv(p, index=False))


class TraceDay:
    def __init__(self, cache, day):
        with np.load(cache / f"day{day}.npz", allow_pickle=False) as data:
            keys = [tuple(row) for row in data["keys"]]
            self.index = {key: i for i, key in enumerate(keys)}
            self.trigger = data["trigger"]
            self.minutes = data["cols"]
            self.counts = data["counts"]
            self.ptr = data["ptr"]
        cumulative = np.r_[0, self.counts.cumsum()]
        self.totals = dict(zip(keys, np.diff(cumulative[self.ptr]).tolist()))
        self.day = day

    def arrivals(self, key, end_minute=1440, start_minute=0):
        i = self.index.get(key)
        if i is None:
            return np.empty(0)
        lo, hi = self.ptr[i:i+2]
        mins = self.minutes[lo:hi]
        keep = (mins >= start_minute) & (mins < end_minute)
        starts = (self.day - 1) * DAY + mins[keep].astype(np.float64) * 60
        n = self.counts[lo:hi][keep]
        stops = n.cumsum()
        starts_idx = stops - n
        count = int(stops[-1]) if len(stops) else 0
        return np.repeat(starts, n) + 60 * (np.arange(count) - np.repeat(starts_idx, n)) / np.repeat(n, n)

    def minute_counts(self, key):
        i = self.index.get(key)
        if i is None:
            return np.zeros(1440, dtype=np.int64)
        lo, hi = self.ptr[i:i+2]
        out = np.zeros(1440, dtype=np.int64)
        out[self.minutes[lo:hi]] = self.counts[lo:hi]
        return out

    def kind(self, key):
        i = self.index.get(key)
        return self.trigger[i] if i is not None else "other"


def read_stats(cache, day):
    durations = pd.read_csv(cache / name(KINDS[1], day))
    memory = pd.read_csv(cache / name(KINDS[2], day))
    return ({tuple(row[:3]): tuple(row[3:]) for row in durations.itertuples(name=None, index=False)},
            {tuple(row[:2]): row[2] for row in memory.itertuples(name=None, index=False)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--archive", type=Path)
    options = parser.parse_args()
    parse(options.cache, options.archive)
