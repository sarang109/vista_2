"""E14: does the evaluation gap survive a bounded attacker?

Elsewhere the attacker may move every freely chosen (F) field to any value on its grid, which a reviewer
may find generous. Here the attacker is limited in two ways, separately and together:

  field budget   at most b F fields change per payment (b = 0 is withholding only), chosen greedily as
                 the fields that lower that payment's risk most
  value window   a moved field may only take values within a fraction delta of its grid (by rank) around
                 its true value, e.g. delta = 0.05 lets the hour move by about one hour and an amount move
                 to a nearby quantile
  combined       b = 2 fields within delta = 0.1

Withholding of attested claims stays free in every setting, since it is the holder's legitimate choice.
Models: the conventional model (unconstrained, or naive where there are no registry flags), the W-monotone
model and the control-typed model, three seeds each.
"""
from __future__ import annotations

import os

from .. import datasets
from ..attack import AttackConfig
from ..control import variants
from ..util import write_csv
from .common import evaluate_trained, fit_variant
from .summarise import by_group

BUDGETS = [0, 1, 2, 3, 5]
WINDOWS = [0.02, 0.05, 0.1, 0.25, 0.5]


def configs(cfg) -> list[AttackConfig]:
    out = [AttackConfig(max_f_fields=b, name=f"fields_{b}") for b in cfg.get("e14_budgets", BUDGETS)]
    out += [AttackConfig(window=w, name=f"window_{w}") for w in cfg.get("e14_windows", WINDOWS)]
    out.append(AttackConfig(max_f_fields=2, window=0.1, name="fields_2_window_0.1"))
    out.append(AttackConfig(name="unbounded"))
    return out


def run(cfg, out, log=print):
    rows = []
    attacks = configs(cfg)
    for name in cfg.get("e14_datasets", ["ieee_cis", "baf", "synthetic"]):
        if not datasets.available(name):
            log(f"  e14: {name} not available, skipped")
            continue
        t = datasets.load(name, cfg.get("synthetic_size"))
        conv = "unconstrained" if t.capability_cols else "naive"
        for v in variants(t, (conv, "monotone_w", "control_typed")):
            for seed in cfg.get("seeds_e14", [1, 2, 3]):
                tr = fit_variant(t, v, "lightgbm", seed=seed)
                res = evaluate_trained(t, tr, attacks, seed=seed, max_legit=cfg.get("max_legit_eval"),
                                       max_fraud_attack=cfg.get("max_fraud_attack"))
                for r in res:
                    r.update(dataset=name, variant=v.name if v.name != conv else "conventional", seed=seed)
                    rows.append(r)
                log(f"  e14 {name} {v.name} seed {seed}: " + ", ".join(
                    f"{r['attack']} {r.get('fraud_accepted_attack', r['fraud_accepted_truthful']):.3f}"
                    for r in res if r["attack"] in ("truthful", "fields_1", "window_0.05", "fields_2_window_0.1",
                                                    "unbounded")))
    return [write_csv(os.path.join(out, "e14_bounded_runs.csv"), rows),
            write_csv(os.path.join(out, "e14_bounded.csv"), by_group(rows, ("dataset", "variant", "attack")))]
