"""Shared pipeline: train a receiver model variant, set its hold threshold, attack the test frauds."""
from __future__ import annotations

import time

import numpy as np

from .. import learners
from ..attack import AttackConfig, _Evaluator, best_response
from ..control import TypedTable, Variant
from ..evaluate import curve, hold_threshold, summary
from ..repair import ExactRepair, TreeBoxRepair


class Trained:
    """A fitted variant: risk() applies the repair when the variant asks for one."""

    def __init__(self, variant: Variant, model, repaired=None, fit_s: float = 0.0):
        self.variant, self.model, self.repaired, self.fit_s = variant, model, repaired, fit_s

    @property
    def columns(self):
        return self.variant.columns

    def risk(self, X):
        if self.repaired is not None:
            return self.repaired.risk(X[self.columns])
        return self.model.risk(X[self.columns])


def make_repair(t: TypedTable, v: Variant, m, kind: str, exact_limit: int = 4096):
    """W-only repairs are exact when the claim lattice is small; otherwise (and for full repairs of tree
    models) the box repair is used."""
    types = ("W",) if v.repair == "w" else ("W", "F")
    size = 1
    for c in v.columns:
        s = t.specs[c]
        if s.ctype in types:
            size *= (int(t.X[c].max()) + 1) if s.ctype == "W" else len(s.values)
    if kind == "lightgbm" and size > exact_limit:
        return TreeBoxRepair(m, t, v.columns, types)
    return ExactRepair(m.risk, t, v.columns, types)


def fit_variant(t: TypedTable, v: Variant, kind: str = "lightgbm", seed: int = 0, train_rows=None,
                **kw) -> Trained:
    Xtr, ytr = t.part(0)
    Xva, yva = t.part(1)
    if train_rows is not None and len(Xtr) > train_rows:
        idx = np.random.default_rng(seed).choice(len(Xtr), train_rows, replace=False)
        Xtr, ytr = Xtr.iloc[np.sort(idx)], ytr[np.sort(idx)]
    if kind == "mlp" and v.monotone:
        kind = "mono_mlp"
    m = learners.make(kind, monotone=v.monotone, seed=seed, **kw)
    t0 = time.perf_counter()
    if kind in ("lightgbm", "xgboost"):
        m.fit(Xtr[v.columns], ytr, Xva[v.columns], yva)
    else:
        m.fit(Xtr[v.columns], ytr)
    fit_s = time.perf_counter() - t0
    rep = None
    if v.repair:
        rep = make_repair(t, v, m, kind)
    return Trained(v, m, rep, fit_s)


def sample_rows(t: TypedTable, part: int, max_legit: int | None, seed: int):
    """Indices (into t.X) of one split, keeping every fraud and at most max_legit legitimate rows."""
    idx = np.flatnonzero(t.split == part)
    if max_legit is None:
        return idx
    y = t.y[idx]
    leg = idx[y == 0]
    if len(leg) > max_legit:
        leg = np.sort(np.random.default_rng(seed + part).choice(leg, max_legit, replace=False))
    return np.sort(np.concatenate([idx[y == 1], leg]))


def evaluate_trained(t: TypedTable, tr: Trained, attacks: list[AttackConfig], hold: float = 0.05,
                     max_legit: int | None = None, max_fraud_attack: int | None = None, seed: int = 0,
                     surrogate=None, curves: list | None = None) -> list[dict]:
    """One row per attack (plus the truthful row). `surrogate`, if given, is a Trained model the attacker
    optimises against instead of the target (transfer attack). If `curves` is a list, operating points
    (fraud accepted at several hold rates, truthful and under each attack) are appended to it."""
    va = sample_rows(t, 1, max_legit, seed)
    te = sample_rows(t, 2, max_legit, seed)
    rv = tr.risk(t.X.iloc[va])
    tau = hold_threshold(rv[t.y[va] == 0], hold)
    Xte, yte = t.X.iloc[te], t.y[te]
    rt = tr.risk(Xte)
    fr = np.flatnonzero(yte == 1)
    if max_fraud_attack is not None and len(fr) > max_fraud_attack:
        fr = np.sort(np.random.default_rng(seed).choice(fr, max_fraud_attack, replace=False))
    amounts = Xte[t.amount].to_numpy(float)[fr] if t.amount else None
    base = summary(yte[np.concatenate([np.flatnonzero(yte == 0), fr])],
                   np.concatenate([rt[yte == 0], rt[fr]]), tau, None, amounts)
    rows = [dict(base, attack="truthful", fit_s=tr.fit_s, n_test=len(te), n_fraud_attacked=len(fr))]
    for cfg in attacks:
        t0 = time.perf_counter()
        target = surrogate if surrogate is not None else tr
        res = best_response(target.risk, Xte.iloc[fr], t, tr.columns, cfg)
        if surrogate is not None:
            # score the surrogate's chosen presentation (including its split) with the real model; the
            # holder can always fall back to the truth
            names = list(res.X.columns)
            ev = _Evaluator(tr.risk, t, names)
            amt = res.X[t.amount].to_numpy(float) if t.amount in names else np.zeros(len(fr))
            cum = res.X[t.cumulative].to_numpy(float) if t.cumulative in names else None
            attacked = np.minimum(ev(res.X.to_numpy(float), res.split_parts, amt, cum), rt[fr])
        else:
            attacked = res.risk
        s = summary(yte[np.concatenate([np.flatnonzero(yte == 0), fr])],
                    np.concatenate([rt[yte == 0], rt[fr]]), tau, attacked, amounts)
        s.update(attack=cfg.name, attack_s=time.perf_counter() - t0, fit_s=tr.fit_s, n_test=len(te),
                 n_fraud_attacked=len(fr), mean_split_parts=float(res.split_parts.mean()))
        acc = attacked <= tau
        if curves is not None:
            yy = np.concatenate([np.zeros((yte == 0).sum()), np.ones(len(fr))])
            rr = np.concatenate([rt[yte == 0], rt[fr]])
            for pt in curve(yy, rr, attacked):
                curves.append(dict(pt, attack=cfg.name))
        s["moved"] = ";".join(f"{c}:{int(k)}" for c, k in res.moved.items() if k)
        if amounts is not None:
            parts = res.split_parts
            s["exposure_attack"] = float(amounts[acc].sum())
            s["exposure_truthful"] = float(amounts[rt[fr] <= tau].sum())
            s["max_parts_used"] = float(parts[acc].max()) if acc.any() else 0.0
        rows.append(s)
    return rows
