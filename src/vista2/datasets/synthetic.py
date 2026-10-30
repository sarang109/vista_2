"""Synthetic real-time payment workload with attested claims during a partial rollout.

Each payment is sent by one of `n_banks` sending institutions (Zipf market shares). A share
`nonparticipating` of the banks issues no claims at all; participating banks can issue each claim with
probability `claim_support`, and the trust registry publishes which claims each bank can issue (the T-type
capability flags). Issued claims occasionally fail to arrive (`issuance_failure`), so "nothing shown" also
occurs honestly in the training data.

Scenarios (latent; the label is fraud = ato or scam):
  legit  ordinary customers with a spread of evidence strength
  ato    account takeover: recent credential reset, new device, weaker authentication, night-skewed hours,
         larger amounts, new beneficiaries, little prior spend in the window
  scam   a deceived customer who authenticates strongly on their own device but pays a new beneficiary a
         large amount in an investment or crypto category

Columns
  W  auth_strength, device_continuity, credential_age, phishing_resistant, dynamic_binding, account_age
  F  hour (complete), amount (split), category (unsigned purpose, step relabel)
  T  cap_<claim> registry flags, signed_mcc (acquirer-signed category, -1 for person-to-person),
     beneficiary_new and beneficiary_inflow_24h (observed by the receiving bank),
     cum_spend_24h (issuer-signed cumulative spend including this payment)
All of this is synthetic and labelled so; it is a controlled testbed, not evidence about real fraud rates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..claims import PAYMENT_SCHEMA
from ..control import FeatureSpec, TypedTable

CLAIMS = PAYMENT_SCHEMA.names
CATEGORIES = ("bills", "merchant", "p2p", "shopping", "investment", "crypto")  # ordered by typical risk
MERCHANT_CATS = {0, 1, 3, 4, 5}


def _bank_shares(n, rng):
    w = 1.0 / np.arange(1, n + 1) ** 0.9
    return rng.permutation(w / w.sum())


def generate(n: int = 240_000, n_banks: int = 60, nonparticipating: float = 0.3, claim_support: float = 0.85,
             issuance_failure: float = 0.02, p_ato: float = 0.02, p_scam: float = 0.008, seed: int = 1) -> TypedTable:
    rng = np.random.default_rng(seed)
    bank = rng.choice(n_banks, size=n, p=_bank_shares(n_banks, rng))
    participating = rng.random(n_banks) >= nonparticipating
    # make sure the requested share holds exactly in bank count
    k = int(round(nonparticipating * n_banks))
    order = rng.permutation(n_banks)
    participating[:] = True
    participating[order[:k]] = False
    cap = (rng.random((n_banks, len(CLAIMS))) < claim_support) & participating[:, None]

    u = rng.random(n)
    scen = np.where(u < p_ato, 1, np.where(u < p_ato + p_scam, 2, 0))  # 0 legit, 1 ato, 2 scam
    ato, scam, legit = scen == 1, scen == 2, scen == 0

    def pick(probs_legit, probs_ato, probs_scam):
        out = np.empty(n, dtype=np.int16)
        for m, pr in ((legit, probs_legit), (ato, probs_ato), (scam, probs_scam)):
            out[m] = rng.choice(len(pr), size=int(m.sum()), p=pr)
        return out

    # true claim levels (1..d for chains; 0/1 for booleans)
    x = {
        "auth_strength": 1 + pick([0.30, 0.45, 0.25], [0.45, 0.40, 0.15], [0.20, 0.45, 0.35]),
        "device_continuity": 1 + pick([0.20, 0.80], [0.55, 0.45], [0.15, 0.85]),
        "credential_age": 1 + pick([0.15, 0.25, 0.60], [0.45, 0.25, 0.30], [0.12, 0.23, 0.65]),
        "phishing_resistant": pick([0.65, 0.35], [0.80, 0.20], [0.60, 0.40]),
        "dynamic_binding": pick([0.30, 0.70], [0.45, 0.55], [0.30, 0.70]),
        "account_age": 1 + pick([0.20, 0.25, 0.55], [0.30, 0.25, 0.45], [0.30, 0.25, 0.45]),
    }
    shown = {}
    for j, c in enumerate(CLAIMS):
        ok = cap[bank, j] & (rng.random(n) >= issuance_failure)
        shown[c] = np.where(ok, x[c], 0).astype(np.int16)

    category = pick([0.22, 0.30, 0.25, 0.15, 0.05, 0.03], [0.06, 0.18, 0.32, 0.22, 0.10, 0.12],
                    [0.02, 0.05, 0.25, 0.10, 0.35, 0.23])
    base_amt = np.array([80.0, 45.0, 120.0, 150.0, 900.0, 500.0])[category]
    mult = np.where(ato, 2.2, np.where(scam, 3.0, 1.0))
    amount = np.round(base_amt * mult * rng.lognormal(0.0, 0.9, n), 2).clip(1.0, 50_000.0)
    day_curve = np.array([1, .6, .4, .3, .3, .5, 1.2, 2.5, 4, 4.5, 4.5, 4.6, 5, 4.8, 4.5, 4.4, 4.6, 5, 5.2, 5,
                          4.2, 3.2, 2.2, 1.5])
    night_curve = day_curve * np.where(np.arange(24) < 6, 3.0, 1.0)
    hour = np.empty(n, dtype=np.int16)
    for m, cv in ((legit | scam, day_curve), (ato, night_curve)):
        hour[m] = rng.choice(24, size=int(m.sum()), p=cv / cv.sum())
    signed_mcc = np.where(np.isin(category, list(MERCHANT_CATS)) & (rng.random(n) < 0.9), category, -1)
    beneficiary_new = np.where(legit, rng.random(n) < 0.15, rng.random(n) < np.where(ato, 0.55, 0.65)).astype(np.int8)
    inflow = rng.poisson(np.where(legit, 2.0, np.where(ato, 3.0, 2.6)))
    prior_spend = np.where(ato, rng.exponential(60.0, n), rng.lognormal(4.0, 1.3, n) * (rng.random(n) < 0.6))
    cum = np.round(prior_spend + amount, 2)

    X = pd.DataFrame({c: shown[c] for c in CLAIMS})
    X["hour"] = hour
    X["amount"] = amount
    X["category"] = category
    for j, c in enumerate(CLAIMS):
        X[f"cap_{c}"] = cap[bank, j].astype(np.int8)
    X["signed_mcc"] = signed_mcc
    X["beneficiary_new"] = beneficiary_new
    X["beneficiary_inflow_24h"] = inflow
    X["cum_spend_24h"] = cum
    X = X.astype(np.float32)

    y = (scen > 0).astype(np.int8)
    split = np.zeros(n, dtype=np.int8)  # payments are in time order; 60 / 20 / 20
    split[int(0.6 * n):int(0.8 * n)] = 1
    split[int(0.8 * n):] = 2

    specs = {c: FeatureSpec(c, "W", "chain", capability=f"cap_{c}", note="issuer-signed threshold claim")
             for c in CLAIMS}
    specs["hour"] = FeatureSpec("hour", "F", "complete", np.arange(24, dtype=float), note="sender picks the time")
    amt_grid = np.unique(np.quantile(amount, np.linspace(0, 1, 24)).round(2))
    specs["amount"] = FeatureSpec("amount", "F", "split", amt_grid, counterpart="cum_spend_24h",
                                  note="per-payment amount, split-closed")
    specs["category"] = FeatureSpec("category", "F", "step", np.arange(len(CATEGORIES), dtype=float),
                                    note="unsigned purpose code; relabelled one step")
    for c in CLAIMS:
        specs[f"cap_{c}"] = FeatureSpec(f"cap_{c}", "T", note="registry: bank can issue this claim")
    specs["signed_mcc"] = FeatureSpec("signed_mcc", "T", note="acquirer-signed merchant category")
    specs["beneficiary_new"] = FeatureSpec("beneficiary_new", "T", note="receiver-observed")
    specs["beneficiary_inflow_24h"] = FeatureSpec("beneficiary_inflow_24h", "T", note="receiver-observed")
    specs["cum_spend_24h"] = FeatureSpec("cum_spend_24h", "T", note="issuer-signed cumulative claim")
    meta = {"scenario": scen, "bank": bank, "participating_banks": int(participating.sum()), "n_banks": n_banks,
            "nonparticipating": nonparticipating, "seed": seed}
    return TypedTable("synthetic", X, y, split, specs, amount="amount", cumulative="cum_spend_24h",
                      baseline_exclude=["signed_mcc", "cum_spend_24h"],
                      capability_cols=[f"cap_{c}" for c in CLAIMS], meta=meta)
