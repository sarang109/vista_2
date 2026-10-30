"""E2: what does each defence cost?  (and E12: operating curves at several hold rates)

Variants (control.variants): naive (no registry capability flags), unconstrained (with flags), monotone in
W, W-repaired (Algorithm 2b on the claim lattice), fully repaired (W and F; box repair for trees), and
control-typed training. Each is trained with LightGBM over several seeds and evaluated truthfully and under
the withholding-only and full attackers. Reported: AUC, legitimate payments held, fraud accepted truthful
and under attack, value accepted.

The full repair of a tree model uses the box relaxation, which is an upper bound on the exact repair. On
the synthetic data the exact full repair is also computed on a subsample (`e2_box_vs_exact.csv`) to show
how much of its accuracy loss is the relaxation and how much is inherent to repairing an F-dependent model.

E12 (`e12_operating_curves.csv`) reuses the same models and attacked presentations but sets each threshold
on the TEST legitimate scores (evaluate.curve) for hold rates of 0.5% to 20%. It is an oracle sensitivity
analysis of the hold budget, not a deployable evaluation, and its 5% row is not the E2 estimate: E2 sets
its threshold on the validation period.
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np

from .. import datasets
from ..attack import FULL, WITHHOLD_ONLY
from ..control import DEFAULT_VARIANTS, variants
from ..evaluate import auc, hold_threshold
from ..repair import ExactRepair
from ..util import write_csv
from .common import evaluate_trained, fit_variant, sample_rows
from .summarise import by_group


def box_vs_exact(t, tr, seed: int, n_legit: int, n_fraud: int) -> dict:
    """Exact full repair of the unconstrained model on a small sample, against the box repair."""
    rng = np.random.default_rng(seed)
    out = {}
    # an 8-point amount grid keeps exact enumeration affordable; the box repair only uses the grid maximum
    specs = dict(t.specs)
    a = specs[t.amount]
    specs[t.amount] = replace(a, values=np.unique(np.quantile(a.values, np.linspace(0, 1, 8))))
    exact = ExactRepair(tr.model.risk, replace(t, specs=specs), tr.columns, ("W", "F"))
    for part, name in ((1, "val"), (2, "test")):
        idx = np.flatnonzero(t.split == part)
        leg = rng.choice(idx[t.y[idx] == 0], n_legit, replace=False)
        fr = rng.choice(idx[t.y[idx] == 1], min(n_fraud, int((t.y[idx] == 1).sum())), replace=False)
        sel = np.concatenate([leg, fr])
        X, y = t.X.iloc[sel], t.y[sel]
        out[name] = (y, exact.risk(X), tr.risk(X), tr.model.risk(X))
    rows = []
    for label, k in (("exact_full_repair", 1), ("box_full_repair", 2), ("unrepaired", 3)):
        yv, rv = out["val"][0], out["val"][k]
        tau = hold_threshold(rv[yv == 0], 0.05)
        yt, rt = out["test"][0], out["test"][k]
        rows.append({"repair": label, "auc": auc(yt, rt), "legit_held": float((rt[yt == 0] > tau).mean()),
                     "fraud_accepted": float((rt[yt == 1] <= tau).mean()), "seed": seed,
                     "n_legit": n_legit, "n_fraud": int((yt == 1).sum()), "model_calls": exact.calls})
    return rows


def run(cfg, out, log=print):
    rows, curves, bve = [], [], []
    for name in cfg.get("e2_datasets", ["synthetic", "ieee_cis", "baf"]):
        if not datasets.available(name):
            log(f"  e2: {name} not available, skipped")
            continue
        t = datasets.load(name, cfg.get("synthetic_size"))
        which = [v for v in cfg.get("e2_variants", DEFAULT_VARIANTS)
                 if not (v == "unconstrained" and not t.capability_cols)]
        for v in variants(t, which):
            for seed in cfg.get("seeds", [1]):
                tr = fit_variant(t, v, "lightgbm", seed=seed)
                cur = []
                res = evaluate_trained(t, tr, [WITHHOLD_ONLY, FULL], seed=seed,
                                       max_legit=cfg.get("max_legit_repair") if v.repair else cfg.get("max_legit_eval"),
                                       max_fraud_attack=cfg.get("max_fraud_attack_repair", 300) if v.repair
                                       else cfg.get("max_fraud_attack"), curves=cur)
                for r in res:
                    r.update(dataset=name, variant=v.name, seed=seed)
                    rows.append(r)
                for c in cur:
                    c.update(dataset=name, variant=v.name, seed=seed)
                    curves.append(c)
                log(f"  e2 {name} {v.name} seed {seed}: auc {res[0]['auc']:.3f} truthful "
                    f"{res[0]['fraud_accepted_truthful']:.3f} full {res[-1]['fraud_accepted_attack']:.3f}")
                if name == "synthetic" and v.name == "repaired_full" and seed == cfg.get("seeds", [1])[0]:
                    bve += box_vs_exact(t, tr, seed, cfg.get("e2_exact_legit", 400), cfg.get("e2_exact_fraud", 150))
    files = [write_csv(os.path.join(out, "e2_defences_runs.csv"), rows),
             write_csv(os.path.join(out, "e2_defences.csv"), by_group(rows, ("dataset", "variant", "attack"))),
             write_csv(os.path.join(out, "e12_operating_curves.csv"),
                       by_group(curves, ("dataset", "variant", "attack", "hold"),
                                ("fraud_accepted_truthful", "fraud_accepted_attack")))]
    if bve:
        files.append(write_csv(os.path.join(out, "e2_box_vs_exact.csv"), bve))
    return files
