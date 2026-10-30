"""E1: how large is the evaluation gap (Theorem 2) on real data?

Conventional receiver models (no monotone constraints, all conventional features) are trained with three
learners and several seeds. For each, fraud acceptance at a 5% legitimate-hold threshold is reported on
truthful test inputs (what a standard evaluation measures) and under three attackers: withholding claims
only, manipulating F fields only, and both. The gap realized - evaluated is the quantity of Theorem 2.
"""
from __future__ import annotations

import os

from .. import datasets
from ..attack import F_ONLY, FULL, WITHHOLD_ONLY
from ..control import variants
from ..util import write_csv
from .common import evaluate_trained, fit_variant
from .summarise import by_group


def run(cfg, out, log=print):
    rows = []
    for name in cfg.get("e1_datasets", ["ieee_cis", "baf", "synthetic"]):
        if not datasets.available(name):
            log(f"  e1: {name} not available, skipped")
            continue
        t = datasets.load(name, cfg.get("synthetic_size"))
        has_cap = bool(t.capability_cols)
        v = variants(t, ("unconstrained",) if has_cap else ("naive",))[0]
        for kind in cfg.get("e1_learners", ["lightgbm", "xgboost", "mlp"]):
            for seed in cfg.get("seeds", [1]):
                train_rows = cfg.get("mlp_train_rows") if kind == "mlp" else None
                tr = fit_variant(t, v, kind, seed=seed, train_rows=train_rows)
                res = evaluate_trained(t, tr, [WITHHOLD_ONLY, F_ONLY, FULL], seed=seed,
                                       max_legit=cfg.get("max_legit_eval"),
                                       max_fraud_attack=cfg.get("max_fraud_attack"))
                for r in res:
                    r.update(dataset=name, learner=kind, seed=seed, model=v.name)
                    rows.append(r)
                log(f"  e1 {name} {kind} seed {seed}: truthful {res[0]['fraud_accepted_truthful']:.3f} "
                    f"full {res[-1]['fraud_accepted_attack']:.3f}")
    files = [write_csv(os.path.join(out, "e1_gap_runs.csv"), rows)]
    files.append(write_csv(os.path.join(out, "e1_gap.csv"), by_group(rows, ("dataset", "learner", "attack"))))
    return files
