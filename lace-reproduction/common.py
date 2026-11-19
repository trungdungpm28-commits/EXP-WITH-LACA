import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 17291
DAY = 86400
MINUTE = 60
FEATURES = 8 + 10 + 3 + 7 + 3
WORKLOADS = ("Representative", "Rare", "Random")
POLICIES = ("LACE", "FaasCache", "TTL 10 min", "LRU")


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, write):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        pending = Path(tmp.name)
    try:
        write(pending)
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def write_json(path, value):
    atomic(path, lambda p: p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))

    def save(p):
        with p.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    atomic(path, save)
