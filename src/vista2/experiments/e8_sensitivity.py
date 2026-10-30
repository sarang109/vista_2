"""E8: how sensitive are the real-data results to the W/F/T typing?

Column semantics are partly anonymised (IEEE-CIS) or ambiguous (BAF: verified or self-reported?), so the
typing is a modelling assumption. Each uncertain feature group is moved to another type and the
control-typed model is retrained:

  pessimistic   T -> F (complete): the holder could in fact change it; the control-typed model drops it and
                the conventional model gains a new attack surface
  optimistic    F -> T: a third party in fact verifies it; the control-typed model may use it freely

The "verified upgrade" rows ask the design question behind Gate A: how much accuracy would the receiver
regain if the sending bank attested the fields that are self-reported today.
"""
from __future__ import annotations

import os

from .. import datasets
from ..attack import FULL
from ..control import variants
from ..util import write_csv
from .common import evaluate_trained, fit_variant

MOVES = {
    "baf": {
        "customer_age": ("T", "fixed"), "current_address_months_count": ("T", "fixed"),
        "prev_address_months_count": ("T", "fixed"), "days_since_request": ("T", "fixed"),
        "device_distinct_emails_8w": ("F", "complete"), "device_fraud_count": ("F", "complete"),
        "foreign_request": ("F", "complete"), "has_other_cards": ("F", "complete"),
    },
    "ieee_cis": {
        "group:D": ("F", "complete"), "group:V": ("F", "complete"), "group:id_num": ("F", "complete"),
        "ProductCD": ("F", "complete"), "group:dist": ("T", "fixed"), "group:M": ("T", "fixed"),
        "group:id_flags": ("T", "fixed"),
    },
}
UPGRADES = {
    "baf": {"verify_income_employment": ["income", "employment_status"],
            "verify_identity_fields": ["customer_age", "current_address_months_count", "prev_address_months_count",
                                       "housing_status"],
            "verify_all_self_reported": ["income", "employment_status", "customer_age", "current_address_months_count",
                                         "prev_address_months_count", "housing_status", "name_email_similarity",
                                         "email_is_free"]},
    "ieee_cis": {"verify_email_and_address": ["P_emaildomain", "R_emaildomain", "addr1", "addr2"],
                 "verify_device": ["DeviceType", "DeviceInfo", "id_30", "id_31", "id_33"]},
}


def _expand(t, name):
    if not name.startswith("group:"):
        return [name]
    g = name.split(":", 1)[1]
    cols = list(t.specs)
    if g == "D":
        return [c for c in cols if c.startswith("D") and c[1:].isdigit()]
    if g == "V":
        return [c for c in cols if c.startswith("V") and c[1:].isdigit()]
    if g == "id_num":
        return [c for c in cols if c.startswith("id_") and t.specs[c].ctype == "T"]
    if g == "dist":
        return ["dist1", "dist2"]
    if g == "M":
        return [f"M{i}" for i in range(1, 10)]
    if g == "id_flags":
        return [c for c in cols if c.startswith("id_") and t.specs[c].ctype == "W"]
    raise KeyError(name)


def _eval(t, label, kind, cfg, seed):
    out = {}
    for v in variants(t, ("naive", "control_typed")):
        tr = fit_variant(t, v, "lightgbm", seed=seed)
        res = evaluate_trained(t, tr, [FULL], seed=seed, max_legit=cfg.get("max_legit_eval"),
                               max_fraud_attack=cfg.get("e8_fraud_attack", 1500))
        out[v.name] = res
    ct, nv = out["control_typed"], out["naive"]
    return {"dataset": t.name, "change": label, "kind": kind, "ct_auc": ct[0]["auc"],
            "ct_fraud_accepted": ct[0]["fraud_accepted_truthful"], "ct_realized": ct[1]["fraud_accepted_attack"],
            "naive_auc": nv[0]["auc"], "naive_fraud_accepted": nv[0]["fraud_accepted_truthful"],
            "naive_realized": nv[1]["fraud_accepted_attack"], "auc_cost": nv[0]["auc"] - ct[0]["auc"]}


def run(cfg, out, log=print):
    rows = []
    seed = cfg.get("seeds", [1])[0]
    for name in cfg.get("e8_datasets", ["baf", "ieee_cis"]):
        if not datasets.available(name):
            log(f"  e8: {name} not available, skipped")
            continue
        t = datasets.load(name)
        rows.append(_eval(t, "baseline typing", "baseline", cfg, seed))
        log(f"  e8 {name} baseline: ct auc {rows[-1]['ct_auc']:.3f}, cost {rows[-1]['auc_cost']:.3f}")
        for label, (ct, reach) in MOVES.get(name, {}).items():
            cols = [c for c in _expand(t, label) if c in t.specs]
            if not cols:
                continue
            kind = "pessimistic" if ct == "F" else "optimistic"
            rows.append(_eval(t.retyped({c: (ct, reach) for c in cols}), f"{label} -> {ct}", kind, cfg, seed))
            log(f"  e8 {name} {label} -> {ct}: ct auc {rows[-1]['ct_auc']:.3f}, cost {rows[-1]['auc_cost']:.3f}")
        for label, cols in UPGRADES.get(name, {}).items():
            cols = [c for c in cols if c in t.specs]
            rows.append(_eval(t.retyped({c: ("T", "fixed") for c in cols}), label, "verified_upgrade", cfg, seed))
            log(f"  e8 {name} {label}: ct auc {rows[-1]['ct_auc']:.3f}, cost {rows[-1]['auc_cost']:.3f}")
    return [write_csv(os.path.join(out, "e8_typing_sensitivity.csv"), rows)]
