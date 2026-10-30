"""E10: the designs of the property matrix, instantiated as receiver configurations on one workload.

Agentic workload (human and agent payments), the same test payments and the same attacker for every
design. Each design is calibrated to hold 5% of legitimate validation payments where it has a threshold.

  raw_evidence       3-D Secure-style data elements: the receiver scores raw evidence with its own model;
                     absent fields are readable signals (no registry flags)
  opaque_score       the sending bank scores the payment with its own model on the TRUE evidence and the
                     receiver thresholds that score (the receiver keeps no policy of its own)
  policy_token       the sending bank evaluates the receiver's published amount-tiered monotone policy on
                     the true claims and signs one bit
  sdjwt_unconstrained selective disclosure; the receiver runs an unconstrained model with registry flags
  ap2_mandate_plus_model  AP2-style mandate caps (a payment carrying a mandate must stay within the
                     per-payment cap) on top of the receiver's existing unconstrained model; a mandates-only
                     receiver cannot decide payments without a mandate (human payments, agents on platforms
                     outside the rollout), so the model handles those
  vista_v1           the receiver evaluates its amount-tiered monotone policy on the disclosed claims
                     (thresholds are discrete, so its hold rate can fall below 5%)
  vista_v1_model     v1 policy AND an unconstrained risk model (what a receiver adding ML to v1 would run)
  vista_v2           control-typed model with the ledger-signed cumulative claim

Attackers: withholding, F manipulation, and both (split up to 4 parts, hour, merchant, text). For designs
evaluated by the sender on true evidence, withholding cannot change the input but F fields still can.
`disclosed_bits` is the plug-in entropy of what the receiver learns about the holder's claims.

Why vista_v1 and policy_token report identical acceptance. Both apply the same amount-tiered monotone policy
with the same calibrated offset; they differ only in who evaluates it. The token is computed by the sender
on the true claims; v1 evaluates the policy on the claims the holder discloses. Because the policy is
non-decreasing in every claim, a holder can never raise v1's decision by withholding (its best response is
to disclose everything it has), so v1's decision on any presentation equals the token's decision on the
truth: the two are the same function of (true claims, amount tier). The attack moves both identically,
through the freely chosen amount tier (splitting) and nothing else. The designs differ in what the receiver
learns (`disclosed_bits`: one bit plus the tier for the token, the disclosed claims for v1), not in fraud
acceptance.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .. import datasets
from ..attack import F_ONLY, FULL, WITHHOLD_ONLY, best_response
from ..control import Variant, variants
from ..datasets.agentic import CHAINS, HUMAN
from ..evaluate import hold_threshold
from ..util import write_csv
from .common import Trained, fit_variant
from .summarise import by_group

TIER_CUTS = (50.0, 200.0, 800.0)


def _entropy(rows: np.ndarray) -> float:
    _, c = np.unique(rows, axis=0, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def claim_score(X: pd.DataFrame) -> np.ndarray:
    """Monotone evidence score used by the policy-based designs (higher = stronger)."""
    s = sum(X[c].to_numpy(float) / d for c, d in zip(CHAINS, (2, 3, 2, 2, 2)))
    h = X["auth_strength"].to_numpy(float) / 3 * 3 + X["device_continuity"].to_numpy(float) / 2 * 2
    return np.where(X["is_agent"].to_numpy() > 0, s, h)


class RuleDesign(Trained):
    """A rule-based design with a single calibrated offset; risk <= tau means accept."""

    def __init__(self, name: str, columns: list, risk_fn):
        super().__init__(Variant(name, columns, {}), None)
        self._risk = risk_fn

    def risk(self, X):
        return self._risk(X)


def policy_risk(offset: float, step: float = 0.35):
    def f(X):
        tier = np.searchsorted(TIER_CUTS, X["amount"].to_numpy(float))
        return (offset + step * tier) - claim_score(X)
    return f


def ap2_risk(cap: float, model, tau: float):
    """AP2-style mandate caps on top of the receiver's existing model: a payment that carries a mandate
    must stay within the per-payment cap; everything is also scored by the model."""
    def f(X):
        mandate = X["mandate_kind"].to_numpy() >= 1
        over_cap = mandate & (X["amount"].to_numpy(float) > cap)
        return np.maximum(over_cap, model.risk(X) > tau).astype(float)
    return f


def calibrate_pair(hold_fn, target: float = 0.05) -> float:
    """Common quantile q for two combined components so that the joint hold on validation is `target`."""
    lo, hi = 0.0, target
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if hold_fn(mid) > target:
            hi = mid
        else:
            lo = mid
    return lo


def run(cfg, out, log=print):
    t = datasets.load("agentic", cfg.get("agentic_size"))
    W = t.cols("W")
    va = t.split == 1
    Xva_leg = t.X[va & (t.y == 0)]
    rows = []
    for seed in cfg.get("seeds_agentic", cfg.get("seeds", [1])):
        designs = {}
        vs = {v.name: v for v in variants(t, ("naive", "unconstrained", "control_typed"))}
        designs["raw_evidence"] = fit_variant(t, Variant("raw_evidence", vs["naive"].columns, {}), "lightgbm", seed=seed)
        designs["sdjwt_unconstrained"] = fit_variant(t, vs["unconstrained"], "lightgbm", seed=seed)
        designs["vista_v2"] = fit_variant(t, vs["control_typed"], "lightgbm", seed=seed)
        # opaque sender score: the sender's model on true evidence; the holder cannot withhold from its own bank
        designs["opaque_score"] = fit_variant(t, Variant("opaque_score", vs["naive"].columns, {}), "lightgbm",
                                              seed=seed + 1000)
        cols_rule = W + ["amount", "is_agent"]
        off = np.quantile(claim_score(Xva_leg) - 0.35 * np.searchsorted(TIER_CUTS, Xva_leg["amount"].to_numpy()), 0.05)
        designs["policy_token"] = RuleDesign("policy_token", cols_rule, policy_risk(off))
        designs["vista_v1"] = RuleDesign("vista_v1", cols_rule, policy_risk(off))
        un = designs["sdjwt_unconstrained"]
        r_un = un.risk(Xva_leg)
        base_pol = claim_score(Xva_leg) - 0.35 * np.searchsorted(TIER_CUTS, Xva_leg["amount"].to_numpy())
        amt_m = Xva_leg.loc[Xva_leg["mandate_kind"] >= 1, "amount"].to_numpy(float)

        def v1_parts(q):
            return policy_risk(np.quantile(base_pol, q)), hold_threshold(r_un, q)

        def v1_hold(q):
            pol, tau_q = v1_parts(q)
            return float(((pol(Xva_leg) > 0) | (r_un > tau_q)).mean())

        q1 = calibrate_pair(v1_hold)
        pol, tau_un = v1_parts(q1)
        designs["vista_v1_model"] = RuleDesign("vista_v1_model", list(dict.fromkeys(un.columns + cols_rule)),
                                               lambda X, pol=pol, tau_un=tau_un: np.maximum(pol(X) > 0, un.risk(X) > tau_un).astype(float))

        def ap2_hold(q):
            return float(ap2_risk(np.quantile(amt_m, 1 - q), un, hold_threshold(r_un, q))(Xva_leg).mean())

        q2 = calibrate_pair(ap2_hold)
        designs["ap2_mandate_plus_model"] = RuleDesign("ap2_mandate_plus_model", list(dict.fromkeys(un.columns + cols_rule)),
                                                       ap2_risk(float(np.quantile(amt_m, 1 - q2)), un, hold_threshold(r_un, q2)))
        sender_side = {"opaque_score", "policy_token"}
        te = t.split == 2
        Xte, yte = t.X[te].reset_index(drop=True), t.y[te]
        fr = np.flatnonzero(yte == 1)
        amounts = Xte["amount"].to_numpy(float)[fr]
        for name, d in designs.items():
            if name in ("vista_v1_model", "ap2_mandate_plus_model"):
                tau = 0.5
            elif isinstance(d, RuleDesign):
                tau = 0.0
            else:
                tau = hold_threshold(d.risk(Xva_leg), 0.05)
            r_te = d.risk(Xte)
            held = float((r_te[yte == 0] > tau).mean())
            acc_t = r_te[fr] <= tau
            seen = {"raw_evidence": W, "sdjwt_unconstrained": W, "vista_v2": W, "vista_v1": W, "vista_v1_model": W,
                    "ap2_mandate_plus_model": W}.get(name)
            bits = _entropy(Xte[seen].to_numpy()) if seen else (
                _entropy(np.column_stack([(r_te <= tau), np.searchsorted(TIER_CUTS, Xte["amount"])]))
                if name == "policy_token" else _entropy(np.round(r_te, 3)[:, None]))
            for att in (WITHHOLD_ONLY, F_ONLY, FULL):
                cfg_att = att
                if name in sender_side:
                    from dataclasses import replace
                    cfg_att = replace(att, withhold=False)
                res = best_response(d.risk, Xte.iloc[fr], t, d.columns, cfg_att)
                acc_a = res.risk <= tau
                rows.append({"design": name, "seed": seed, "attack": att.name, "legit_held": held,
                             "fraud_accepted_truthful": float(acc_t.mean()), "fraud_accepted_attack": float(acc_a.mean()),
                             "gap": float(acc_a.mean() - acc_t.mean()),
                             "fraud_value_accepted_attack": float(amounts[acc_a].sum() / amounts.sum()),
                             "exposure_attack": float(amounts[acc_a].sum()), "exposure_truthful": float(amounts[acc_t].sum()),
                             "disclosed_bits": bits})
            log(f"  e10 seed {seed} {name}: held {held:.3f} truthful {acc_t.mean():.3f} full {rows[-1]['fraud_accepted_attack']:.3f}")
    summ = by_group(rows, ("design", "attack"), ("legit_held", "fraud_accepted_truthful", "fraud_accepted_attack", "gap",
                                               "fraud_value_accepted_attack", "exposure_attack", "disclosed_bits"))
    return [write_csv(os.path.join(out, "e10_comparison_runs.csv"), rows),
            write_csv(os.path.join(out, "e10_comparison.csv"), summ)]
