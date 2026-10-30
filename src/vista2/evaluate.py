"""Thresholds and metrics: evaluated versus realized fraud acceptance (Theorem 2).

The receiver accepts a payment iff risk <= tau, with tau set on truthful validation data so that a fixed
share of legitimate payments is held (5% by default). "Evaluated" numbers score truthful inputs, which is
what every standard fraud-model evaluation reports; "realized" numbers score the attacker's best
presentation of each fraud, which is what the deployed rule faces.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score


def hold_threshold(legit_risk: np.ndarray, hold: float = 0.05) -> float:
    """Largest tau such that at most `hold` of these legitimate payments have risk > tau."""
    r = np.sort(np.asarray(legit_risk, dtype=float))
    if len(r) == 0:
        return math.inf
    k = int(math.ceil((1.0 - hold) * len(r))) - 1
    return float(r[min(max(k, 0), len(r) - 1)])


def auc(y, risk) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return math.nan
    return float(roc_auc_score(y, risk))


def acceptance(risk, tau) -> np.ndarray:
    return np.asarray(risk) <= tau


def summary(y_test, risk_test, tau, fraud_attacked_risk=None, fraud_amount=None) -> dict:
    """Metrics on a test set. fraud_attacked_risk is aligned with the fraud rows of y_test (in order)."""
    y = np.asarray(y_test)
    r = np.asarray(risk_test)
    fr = y == 1
    acc_t = acceptance(r[fr], tau)
    out = {"auc": auc(y, r), "tau": tau, "legit_held": float(1 - acceptance(r[~fr], tau).mean()),
           "fraud_accepted_truthful": float(acc_t.mean()) if fr.any() else math.nan}
    if fraud_amount is not None:
        a = np.asarray(fraud_amount, dtype=float)
        out["fraud_value_accepted_truthful"] = float(a[acc_t].sum() / a.sum())
    if fraud_attacked_risk is not None:
        ra = np.asarray(fraud_attacked_risk)
        acc_a = acceptance(ra, tau)
        out["fraud_accepted_attack"] = float(acc_a.mean())
        out["gap"] = out["fraud_accepted_attack"] - out["fraud_accepted_truthful"]
        out["share_frauds_gaining"] = float((acc_a & ~acc_t).mean())
        out["theorem2_violations"] = int((ra > r[fr] + 1e-9).sum())  # attacked risk above truthful
        r2 = r.copy()
        r2[fr] = ra
        out["auc_under_attack"] = auc(y, r2)
        if fraud_amount is not None:
            out["fraud_value_accepted_attack"] = float(a[acc_a].sum() / a.sum())
    return out


def curve(y, risk, attacked=None, holds=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2)) -> list[dict]:
    """Operating points: for each legitimate-hold rate, fraud accepted truthful and under attack.

    The threshold for each rate is set on the legitimate scores passed in, which in E12 are the test
    payments themselves. These curves are therefore an oracle sensitivity analysis of how the conclusions
    move with the hold budget, not operating points a receiver could deploy; the deployable thresholds of E2
    are set on the validation period (`hold_threshold` on validation scores)."""
    y = np.asarray(y)
    r = np.asarray(risk)
    rows = []
    for h in holds:
        tau = hold_threshold(r[y == 0], h)
        row = {"hold": h, "fraud_accepted_truthful": float((r[y == 1] <= tau).mean())}
        if attacked is not None:
            row["fraud_accepted_attack"] = float((np.asarray(attacked) <= tau).mean())
        rows.append(row)
    return rows
