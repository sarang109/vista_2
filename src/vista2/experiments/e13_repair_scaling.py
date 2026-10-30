"""E13: how does the cost of repairing a black-box model grow with the claim lattice?

Algorithm 2b enumerates the up-set of the shown claims, so its cost per payment is the product of
(L_i - w_i) over the claims: 4^n for a holder that shows nothing. The box repair for tree ensembles costs
trees x leaves per payment whatever the lattice. For n = 3..9 four-level claims (64 to 262,144 lattice
points) a LightGBM model is trained on a synthetic task and both repairs are timed on random payments and
on the worst case (nothing shown). The box repair's excess risk over the exact repair is reported where
the exact repair was run.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from ..control import FeatureSpec, TypedTable
from ..learners import LGBModel
from ..repair import ExactRepair, TreeBoxRepair
from ..util import write_csv


def _task(n_claims: int, rows: int, rng):
    W = rng.integers(0, 4, (rows, n_claims))
    T = rng.normal(0, 1, (rows, 3))
    beta = rng.uniform(0.2, 0.8, n_claims)
    inter = 0.3 * (W[:, 0] == 0) * (W[:, 1 % n_claims] >= 2)
    logit = -3.0 - W @ beta * 0.5 + 0.8 * T[:, 0] + inter + 0.4 * (W[:, -1] == 0)
    y = (rng.random(rows) < 1 / (1 + np.exp(-logit))).astype(int)
    X = pd.DataFrame(np.column_stack([W, T]).astype(np.float32),
                     columns=[f"w{i}" for i in range(n_claims)] + ["t0", "t1", "t2"])
    specs = {f"w{i}": FeatureSpec(f"w{i}", "W", "chain") for i in range(n_claims)}
    specs.update({f"t{i}": FeatureSpec(f"t{i}", "T") for i in range(3)})
    return TypedTable(f"lattice{n_claims}", X, y, np.zeros(rows, dtype=np.int8), specs), X.columns.tolist()


def run(cfg, out, log=print):
    rng = np.random.default_rng(cfg.get("seed", 1) + 13)
    rows = []
    for n_claims in cfg.get("e13_claims", [3, 4, 5, 6, 7, 8, 9]):
        t, cols = _task(n_claims, 60_000, rng)
        m = LGBModel(seed=1, rounds=150, leaves=31).fit(t.X, t.y)
        ex = ExactRepair(m.risk, t, cols, ("W",))
        bx = TreeBoxRepair(m, t, cols, ("W",))
        q_n = cfg.get("e13_queries", 100) if n_claims <= 8 else max(10, cfg.get("e13_queries", 100) // 10)
        q = t.X.sample(q_n, random_state=n_claims)
        worst = q.copy()
        worst[[f"w{i}" for i in range(n_claims)]] = 0.0
        for label, Q in (("random", q), ("nothing_shown", worst.iloc[: max(3, q_n // 10)])):
            t0 = time.perf_counter()
            calls0 = ex.calls
            re = ex.risk(Q)
            t_ex = (time.perf_counter() - t0) * 1e3 / len(Q)
            t0 = time.perf_counter()
            rb = bx.risk(Q)
            t_bx = (time.perf_counter() - t0) * 1e3 / len(Q)
            rows.append({"claims": n_claims, "lattice_points": 4 ** n_claims, "queries": label, "n": len(Q),
                         "exact_ms_per_payment": t_ex, "exact_model_calls_per_payment": (ex.calls - calls0) / len(Q),
                         "box_ms_per_payment": t_bx, "box_minus_exact_mean": float((rb - re).mean()),
                         "box_minus_exact_max": float((rb - re).max()), "box_below_exact": int((rb < re - 1e-9).sum()),
                         "trees": len(bx.trees)})
        log(f"  e13 n={n_claims}: exact {rows[-2]['exact_ms_per_payment']:.1f} ms random, "
            f"{rows[-1]['exact_ms_per_payment']:.1f} ms worst; box {rows[-2]['box_ms_per_payment']:.2f} ms")
    return [write_csv(os.path.join(out, "e13_repair_scaling.csv"), rows)]
