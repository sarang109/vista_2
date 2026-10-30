"""E15: are the headline numbers tight? Two robustness checks on the main results.

(a) Attack tightness on IEEE-CIS (1,500 attacked frauds per model). Coordinate search with 1 to 4 rounds,
    and searches restarted from 5 and 10 random reachable presentations (best over the plain search and
    every restart; the 5-restart figure is read off the same sequence of starts). If more search finds
    little more, the attack numbers reported elsewhere are close to the true best response.
(b) Exact hold rate. Thresholds elsewhere are set on the validation period, so the share of legitimate
    test payments held drifts from 5% (to 5.7-7% on the real data). Here the same models are also scored
    at a threshold that holds exactly 5% of legitimate test payments.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .. import datasets
from ..attack import FULL, WITHHOLD_ONLY, AttackConfig, best_response_restarts
from ..control import variants
from ..evaluate import hold_threshold
from ..util import write_csv
from .common import evaluate_trained, fit_variant, sample_rows
from .summarise import by_group


def tightness(cfg, log) -> list[dict]:
    rows = []
    t = datasets.load("ieee_cis")
    for v in variants(t, ("naive", "monotone_w", "control_typed")):
        for seed in cfg.get("seeds_e15", [1, 2, 3]):
            tr = fit_variant(t, v, "lightgbm", seed=seed)
            va, te = sample_rows(t, 1, None, seed), sample_rows(t, 2, None, seed)
            tau = hold_threshold(tr.risk(t.X.iloc[va])[t.y[va] == 0], 0.05)
            fr = te[t.y[te] == 1]
            n_att = cfg.get("e15_fraud_attack", 1500)
            if len(fr) > n_att:
                fr = np.sort(np.random.default_rng(seed).choice(fr, n_att, replace=False))
            Xf = t.X.iloc[fr]
            truthful = float((tr.risk(Xf) <= tau).mean())
            line = {}
            searches = {}
            for label, rounds in (("coord_1_round", 1), ("coord_2_rounds", 2), ("coord_3_rounds", 3),
                                  ("coord_4_rounds", 4)):
                t0 = time.perf_counter()
                searches[label] = (best_response_restarts(tr.risk, Xf, t, tr.columns, AttackConfig(rounds=rounds), 0,
                                                          seed), time.perf_counter() - t0)
            t0 = time.perf_counter()
            snaps = best_response_restarts(tr.risk, Xf, t, tr.columns, AttackConfig(rounds=2), 10, seed,
                                           checkpoints=(5, 10))
            el = time.perf_counter() - t0
            searches["restarts_5"] = (snaps[5], el / 2)
            searches["restarts_10"] = (snaps[10], el)
            for label, (r, sec) in searches.items():
                acc = float((r <= tau).mean())
                rows.append({"dataset": "ieee_cis", "variant": v.name, "seed": seed, "search": label,
                             "fraud_accepted_truthful": truthful, "fraud_accepted_attack": acc,
                             "mean_attacked_risk": float(r.mean()), "seconds": sec})
                line[label] = acc
            log(f"  e15a {v.name} seed {seed}: truthful {truthful:.3f}, " +
                ", ".join(f"{k} {x:.3f}" for k, x in line.items()))
    return rows


def exact_hold(cfg, log) -> list[dict]:
    rows = []
    for name in cfg.get("e15_datasets", ["ieee_cis", "baf", "synthetic"]):
        if not datasets.available(name):
            continue
        t = datasets.load(name, cfg.get("synthetic_size"))
        which = [x for x in ("naive", "unconstrained", "monotone_w", "control_typed")
                 if not (x == "unconstrained" and not t.capability_cols)]
        for v in variants(t, which):
            for seed in cfg.get("seeds", [1]):
                tr = fit_variant(t, v, "lightgbm", seed=seed)
                for r in evaluate_trained(t, tr, [WITHHOLD_ONLY, FULL], seed=seed,
                                          max_legit=cfg.get("max_legit_eval"),
                                          max_fraud_attack=cfg.get("max_fraud_attack")):
                    r.update(dataset=name, variant=v.name, seed=seed)
                    rows.append(r)
            log(f"  e15b {name} {v.name}: done")
    return rows


def run(cfg, out, log=print):
    files = []
    if datasets.available("ieee_cis"):
        rows = tightness(cfg, log)
        files += [write_csv(os.path.join(out, "e15_attack_tightness_runs.csv"), rows),
                  write_csv(os.path.join(out, "e15_attack_tightness.csv"),
                            by_group(rows, ("dataset", "variant", "search"),
                                     ("fraud_accepted_truthful", "fraud_accepted_attack", "mean_attacked_risk", "seconds")))]
    rows = exact_hold(cfg, log)
    files += [write_csv(os.path.join(out, "e15_exact_hold_runs.csv"), rows),
              write_csv(os.path.join(out, "e15_exact_hold.csv"),
                        by_group(rows, ("dataset", "variant", "attack"),
                                 ("auc", "legit_held", "fraud_accepted_truthful", "fraud_accepted_attack",
                                  "fraud_accepted_truthful_test_hold", "fraud_accepted_attack_test_hold")))]
    return files
