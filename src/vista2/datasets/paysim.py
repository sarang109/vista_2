"""PaySim mobile-money log (E. Lopez-Rojas, A. Elmir and S. Axelsson, 2016): download, verify, cache, load.

The public log has 6,362,620 rows and 8,213 labelled frauds. `build_cache` makes one streaming pass over the
CSV, refuses a file that does not match the published statistics, drops CASH_IN (not a payment) and writes
memory-mapped .npy columns for the remaining 4,963,336 TRANSFER, CASH_OUT, PAYMENT and DEBIT rows, so E3 can
read any slice without loading the CSV.

Build the cache from the repository root with

    python -m vista2.datasets.paysim                 # downloads the CSV (494 MB, no account needed)
    python -m vista2.datasets.paysim --csv PATH      # uses a copy you already have

The default download is a Hugging Face mirror of the Kaggle file PS_20174392719_1491204439457_log.csv
(dataset "Synthetic Financial Datasets For Fraud Detection"); the file is checked against its SHA-256 and the
published row, fraud and type counts either way.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request
from collections import Counter

import numpy as np

from ..util import sha256_file
from . import DATA

URL = ("https://huggingface.co/datasets/vitaliy-sharandin/synthetic-fraud-detection/resolve/main/"
       "PS_20174392719_1491204439457_log.csv")
PINNED_SHA256 = "16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b"
EXPECTED = {"rows": 6_362_620, "fraud": 8_213, "flagged": 16, "steps": 743,
            "types": {"CASH_OUT": 2_237_500, "PAYMENT": 2_151_495, "CASH_IN": 1_399_284,
                      "TRANSFER": 532_909, "DEBIT": 41_432}}
CACHE_VERSION = 1
TYPE_CODE = {"TRANSFER": 0, "CASH_OUT": 1, "PAYMENT": 2, "DEBIT": 3}  # CASH_IN is not a payment
COLS = ("step", "type", "amount_minor", "orig", "dest", "dest_m", "fraud")
_DT = {"step": np.int16, "type": np.int8, "amount_minor": np.int64, "orig": np.int64, "dest": np.int64,
       "dest_m": np.int8, "fraud": np.int8}
LOAD_COLS = ("step", "type", "amount_minor", "orig", "dest", "fraud")


class PaySimMismatch(ValueError):
    pass


def default_paths() -> tuple[str, str]:
    return os.path.join(DATA, "paysim.csv"), os.path.join(DATA, "paysim_cache")


def download(path: str, url: str = URL) -> str:
    """Stream the CSV to `path` through a temporary file (atomic rename)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "vista2-artifact/2.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, time.time()
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if time.time() - last > 10:
                print(f"  downloaded {done / 1e6:.0f} of {total / 1e6:.0f} MB", flush=True)
                last = time.time()
    os.replace(tmp, path)
    return path


def build_cache(csv_path: str, cache_dir: str, expected: dict | None = EXPECTED) -> dict:
    """One streaming pass over the CSV; verifies the published statistics and writes the column cache."""
    size = os.path.getsize(csv_path)
    sha = sha256_file(csv_path)
    parts: dict[str, list] = {k: [] for k in COLS}
    buf: dict[str, list] = {k: [] for k in COLS}
    stats = {"rows": 0, "fraud": 0, "flagged": 0, "types": Counter(), "max_step": 0}
    t0 = time.time()

    def flush():
        for k in COLS:
            parts[k].append(np.asarray(buf[k], dtype=_DT[k]))
            buf[k].clear()

    with open(csv_path, newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        ix = {h: i for i, h in enumerate(head)}
        need = ("step", "type", "amount", "nameOrig", "nameDest", "isFraud", "isFlaggedFraud")
        if any(n not in ix for n in need):
            raise PaySimMismatch(f"not a PaySim file; header = {head}")
        for row in rd:
            stats["rows"] += 1
            t = row[ix["type"]]
            stats["types"][t] += 1
            step = int(row[ix["step"]])
            stats["max_step"] = max(stats["max_step"], step)
            fr = row[ix["isFraud"]] == "1"
            stats["fraud"] += fr
            stats["flagged"] += row[ix["isFlaggedFraud"]] == "1"
            if t in TYPE_CODE:
                nd = row[ix["nameDest"]]
                buf["step"].append(step)
                buf["type"].append(TYPE_CODE[t])
                buf["amount_minor"].append(max(1, int(round(float(row[ix["amount"]]) * 100))))
                buf["orig"].append(int(row[ix["nameOrig"]][1:]))
                buf["dest"].append(int(nd[1:]))
                buf["dest_m"].append(1 if nd[0] == "M" else 0)
                buf["fraud"].append(1 if fr else 0)
            if stats["rows"] % 500_000 == 0:
                flush()
    flush()
    if expected:
        got = {"rows": stats["rows"], "fraud": stats["fraud"], "flagged": stats["flagged"],
               "steps": stats["max_step"], "types": dict(stats["types"])}
        if got != expected:
            raise PaySimMismatch(f"file does not match the published PaySim statistics.\n expected {expected}\n got      {got}")
    os.makedirs(cache_dir, exist_ok=True)
    n_use = 0
    for k in COLS:
        arr = np.concatenate(parts[k])
        np.save(os.path.join(cache_dir, k + ".npy"), arr)
        n_use = len(arr)
    meta = {"version": CACHE_VERSION, "source_sha256": sha, "source_bytes": size,
            "sha256_matches_pinned_hf_copy": sha == PINNED_SHA256, "rows_total": stats["rows"],
            "rows_usable": n_use, "rows_excluded_cash_in": stats["rows"] - n_use,
            "fraud_total": stats["fraud"], "fraud_usable": int(np.concatenate(parts["fraud"]).sum()),
            "flagged_total": stats["flagged"], "types": dict(stats["types"]), "max_step": stats["max_step"],
            "verified_against_published_statistics": bool(expected), "parse_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(cache_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def ensure_cache(csv_path: str | None = None, cache_dir: str | None = None, url: str = URL,
                 allow_download: bool = True) -> dict:
    """Download (if needed), verify and cache. Idempotent: a valid cache for the same file is reused."""
    d_csv, d_cache = default_paths()
    csv_path, cache_dir = csv_path or d_csv, cache_dir or d_cache
    meta_p = os.path.join(cache_dir, "meta.json")
    complete = os.path.exists(meta_p) and all(os.path.exists(os.path.join(cache_dir, k + ".npy")) for k in COLS)
    if not os.path.exists(csv_path) and complete:
        with open(meta_p) as fh:
            meta = json.load(fh)
        if meta.get("version") == CACHE_VERSION and meta.get("verified_against_published_statistics"):
            return meta  # verified when it was built; the CSV is only needed to rebuild it
    if not os.path.exists(csv_path):
        if not allow_download:
            raise FileNotFoundError(csv_path)
        download(csv_path, url)
    if os.path.exists(meta_p):
        with open(meta_p) as fh:
            meta = json.load(fh)
        ok = (meta.get("version") == CACHE_VERSION and meta.get("source_bytes") == os.path.getsize(csv_path)
              and all(os.path.exists(os.path.join(cache_dir, k + ".npy")) for k in COLS))
        if ok and meta.get("source_sha256") == sha256_file(csv_path):
            return meta
    return build_cache(csv_path, cache_dir, EXPECTED)


def load(cache_dir: str | None = None) -> dict:
    d = cache_dir or default_paths()[1]
    out = {c: np.load(os.path.join(d, c + ".npy"), mmap_mode="r") for c in LOAD_COLS}
    with open(os.path.join(d, "meta.json")) as fh:
        out["meta"] = json.load(fh)
    return out


def window_spend(key: np.ndarray, window: np.ndarray, amount: np.ndarray) -> np.ndarray:
    """Cumulative spend per (key, window) in row order, including the current row (the issuer counter)."""
    order = np.lexsort((np.arange(len(key)), window, key))
    k, w, a = key[order], window[order], amount[order].astype(float)
    new = np.ones(len(k), dtype=bool)
    new[1:] = (k[1:] != k[:-1]) | (w[1:] != w[:-1])
    cs = np.cumsum(a)
    start = np.maximum.accumulate(np.where(new, np.arange(len(k)), 0))
    base = np.where(start > 0, cs[start - 1], 0.0)
    res = np.empty(len(k))
    res[order] = cs - base
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download, verify and cache the PaySim log for E3.")
    ap.add_argument("--csv", help="path to an existing PS_20174392719_1491204439457_log.csv")
    ap.add_argument("--cache", help="cache directory (default data/paysim_cache)")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args(argv)
    meta = ensure_cache(args.csv, args.cache, allow_download=not args.no_download)
    print(json.dumps({k: meta[k] for k in ("rows_total", "rows_usable", "fraud_total",
                                           "sha256_matches_pinned_hf_copy", "verified_against_published_statistics")}))


if __name__ == "__main__":
    main()
