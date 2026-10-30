"""PaySim mobile-money log (E. Lopez-Rojas et al.), read from the column cache built by VISTA v1.

The cache holds the 4,963,336 TRANSFER, CASH_OUT, PAYMENT and DEBIT rows of the published 6,362,620-row
log (CASH_IN is not a payment). See data/README.md for how to rebuild it from the public CSV.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import DATA

COLS = ("step", "type", "amount_minor", "orig", "dest", "fraud")


def load(cache_dir: str | None = None) -> dict:
    d = cache_dir or os.path.join(DATA, "paysim_cache")
    out = {c: np.load(os.path.join(d, c + ".npy"), mmap_mode="r") for c in COLS}
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
