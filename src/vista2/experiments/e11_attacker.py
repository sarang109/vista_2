"""E11: how strong is the attacker, and does the attack need white-box access?

Synthetic workload. (a) Search strength: coordinate search with 1, 2 and 3 rounds, and exhaustive
enumeration of the 768-point claim lattice followed by two rounds over the F fields. If more search finds
little more, the coordinate-search numbers elsewhere are close to the true best response. (b) Knowledge:
a transfer attacker optimises against its own surrogate (a LightGBM trained on a disjoint 30% of the
training period with another seed, or an XGBoost model) and submits the result to the receiver's model,
falling back to the truth when the surrogate's choice would do worse.
"""
from __future__ import annotations

import os

import numpy as np

from .. import datasets
from ..attack import AttackConfig
from ..control import variants
from ..util import write_csv
from .common import Trained, evaluate_trained, fit_variant


def run(cfg, out, log=print):
    t = datasets.load("synthetic", cfg.get("synthetic_size"))
    seed = cfg.get("seeds", [1])[0]
    strengths = [AttackConfig(rounds=1, name="coord_1_round"), AttackConfig(rounds=2, name="coord_2_rounds"),
                 AttackConfig(rounds=3, name="coord_3_rounds"),
                 AttackConfig(rounds=2, exhaustive_w_limit=1024, name="exhaustive_w_plus_2_rounds")]
    rows = []
    tr_idx = np.flatnonzero(t.split == 0)
    rng = np.random.default_rng(seed + 100)
    for v in variants(t, ("naive", "unconstrained", "monotone_w", "control_typed")):
        target = fit_variant(t, v, "lightgbm", seed=seed)
        for r in evaluate_trained(t, target, strengths, seed=seed, max_fraud_attack=cfg.get("max_fraud_attack")):
            r.update(variant=v.name, knowledge="white_box")
            rows.append(r)
        # transfer attackers
        sub = np.zeros(len(t.X), dtype=bool)
        sub[rng.choice(tr_idx, int(0.3 * len(tr_idx)), replace=False)] = True
        t_sur = t.subset(sub | (t.split > 0))
        for label, kind in (("surrogate_lightgbm_30pct", "lightgbm"), ("surrogate_xgboost_30pct", "xgboost")):
            sur = fit_variant(t_sur, v, kind, seed=seed + 7)
            res = evaluate_trained(t, target, [AttackConfig(rounds=2, name=label)], seed=seed,
                                   max_fraud_attack=cfg.get("max_fraud_attack"), surrogate=sur)
            r = res[-1]
            r.update(variant=v.name, knowledge="transfer")
            rows.append(r)
        log(f"  e11 {v.name}: " + ", ".join(f"{r['attack']} {r.get('fraud_accepted_attack', r['fraud_accepted_truthful']):.3f}"
                                            for r in rows if r["variant"] == v.name))
    return [write_csv(os.path.join(out, "e11_attacker_strength.csv"), rows)]
