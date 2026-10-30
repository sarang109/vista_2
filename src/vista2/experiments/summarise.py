"""Aggregate per-seed rows into mean and standard deviation per group."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

DEFAULT_METRICS = ("auc", "auc_under_attack", "legit_held", "fraud_accepted_truthful", "fraud_accepted_attack",
                   "gap", "share_frauds_gaining", "fraud_value_accepted_truthful", "fraud_value_accepted_attack",
                   "exposure_attack", "exposure_truthful", "attack_s", "fit_s", "theorem2_violations")


def by_group(rows: list[dict], keys: tuple, metrics=DEFAULT_METRICS) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in keys)].append(r)
    out = []
    for g, rs in groups.items():
        row = dict(zip(keys, g))
        row["seeds"] = len(rs)
        for m in metrics:
            vals = [float(r[m]) for r in rs if m in r and r[m] not in ("", None) and not
                    (isinstance(r[m], float) and math.isnan(r[m]))]
            if vals:
                row[m] = float(np.mean(vals))
                row[m + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(row)
    return out
