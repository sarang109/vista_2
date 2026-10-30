"""E3: does the cumulative claim bound exposure under splitting (Theorem 3)?

PaySim payments, window = 24 simulated hours. Four receiver rules, each calibrated to hold 5% of
legitimate validation payments:

  stateless_cap      accept iff amount <= A
  stateless_model    LightGBM on (amount, type, hour)
  cumulative_cap     accept iff c <= B, c = issuer-signed spend of the payer in the window incl. this payment
  cumulative_model   LightGBM on (c, type), risk monotone non-decreasing in c (control-typed)

Each test fraud drains its amount a. The attacker splits a into k <= K equal parts (and, against the model
that reads the hour, also picks the best hour). Stateless rules see identical parts; under the cumulative
claim part j carries c = prior + j a / k and the issuer releases the reservation of a rejected part.
Reported per rule and K: share of fraud value accepted, mean and maximum accepted total per attacker
window, and legitimate payments held. `e3_bound_sweep.csv` traces the price of the bound (legitimate
holds against the exposure cap B) and `e3_atomicity.csv` runs Algorithm 3 with real signatures under
parallel requests, with and without an atomic counter.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import pandas as pd

from .. import cumulative
from ..crypto.signed import SigningKey
from ..datasets import available, paysim
from ..evaluate import hold_threshold
from ..learners import LGBModel
from ..trust import TrustRegistry
from ..util import percentiles, write_csv

KS = (1, 2, 4, 8, 16, 64)


def _frame(ps, idx):
    amt = ps["amount_minor"][idx] / 100.0
    return pd.DataFrame({"amount": amt, "type": ps["type"][idx].astype(float),
                         "hour": (ps["step"][idx] % 24).astype(float)})


def run(cfg, out, log=print):
    if not available("paysim"):
        log("  e3: PaySim cache not available, skipped")
        return []
    rng = np.random.default_rng(cfg.get("seed", 1))
    ps = paysim.load()
    n = len(ps["step"])
    step = np.asarray(ps["step"])
    fraud = np.asarray(ps["fraud"]).astype(bool)
    orig = np.asarray(ps["orig"])
    amount = np.asarray(ps["amount_minor"]) / 100.0
    window = step // 24
    c_all = paysim.window_spend(orig, window, amount)
    cut1, cut2 = np.quantile(step, [0.6, 0.8])
    split = np.where(step <= cut1, 0, np.where(step <= cut2, 1, 2))
    sub = cfg.get("e3_train_rows", 1_500_000)
    tr_idx = np.flatnonzero(split == 0)
    if len(tr_idx) > sub:
        tr_idx = np.sort(rng.choice(tr_idx, sub, replace=False))
    va = np.flatnonzero(split == 1)
    te = np.flatnonzero(split == 2)
    va_leg = va[~fraud[va]]
    te_leg = te[~fraud[te]]
    te_fr = te[fraud[te]]
    log(f"  e3: {n} payments, test frauds {len(te_fr)}")

    rules = {}
    A = hold_threshold(amount[va_leg], 0.05)
    rules["stateless_cap"] = {"tau": A}
    B = hold_threshold(c_all[va_leg], 0.05)
    rules["cumulative_cap"] = {"tau": B}
    X_tr = _frame(ps, tr_idx)
    m_st = LGBModel(seed=1, rounds=200).fit(X_tr, fraud[tr_idx].astype(int))
    Xv = _frame(ps, va_leg)
    rules["stateless_model"] = {"model": m_st, "tau": hold_threshold(m_st.risk(Xv), 0.05)}
    Xc_tr = pd.DataFrame({"c": c_all[tr_idx], "type": X_tr["type"]})
    m_cu = LGBModel(monotone={"c": +1}, seed=1, rounds=200).fit(Xc_tr, fraud[tr_idx].astype(int))
    Xcv = pd.DataFrame({"c": c_all[va_leg], "type": Xv["type"]})
    rules["cumulative_model"] = {"model": m_cu, "tau": hold_threshold(m_cu.risk(Xcv), 0.05)}

    fa = amount[te_fr]
    ftype = np.asarray(ps["type"])[te_fr].astype(float)
    fprior = c_all[te_fr] - fa  # spend already in the window before the fraud
    rows = []
    for name, r in rules.items():
        tau = r["tau"]
        # legitimate holds (truthful)
        if name == "stateless_cap":
            held = float((amount[te_leg] > tau).mean())
        elif name == "cumulative_cap":
            held = float((c_all[te_leg] > tau).mean())
        elif name == "stateless_model":
            held = float((r["model"].risk(_frame(ps, te_leg)) > tau).mean())
        else:
            held = float((r["model"].risk(pd.DataFrame({"c": c_all[te_leg], "type": amount[te_leg] * 0 +
                                                        np.asarray(ps["type"])[te_leg]})) > tau).mean())
        for K in KS:
            best = np.zeros(len(fa))
            parts_used = np.ones(len(fa))
            for k in range(1, K + 1):
                part = fa / k
                if name == "stateless_cap":
                    ok = part <= tau
                    tot = np.where(ok, fa, 0.0)
                elif name == "stateless_model":
                    ok = np.zeros(len(fa), dtype=bool)
                    for h in range(24):
                        ok |= r["model"].risk(pd.DataFrame({"amount": part, "type": ftype,
                                                            "hour": np.full(len(fa), float(h))})) <= tau
                    tot = np.where(ok, fa, 0.0)
                else:
                    # parts arrive in order; counter = prior + accepted so far + current part
                    acc = np.zeros(len(fa))
                    for j in range(k):
                        c_now = fprior + acc + part
                        if name == "cumulative_cap":
                            ok = c_now <= tau
                        else:
                            ok = r["model"].risk(pd.DataFrame({"c": c_now, "type": ftype})) <= tau
                        acc = acc + np.where(ok, part, 0.0)
                    tot = acc
                better = tot > best
                parts_used = np.where(better, k, parts_used)
                best = np.maximum(best, tot)
            rows.append({"rule": name, "K": K, "tau": tau, "legit_held": held,
                         "fraud_value_accepted": float(best.sum() / fa.sum()),
                         "frauds_fully_drained": float((best >= fa - 1e-6).mean()),
                         "mean_exposure": float(best.mean()), "max_exposure": float(best.max()),
                         "p99_exposure": float(np.quantile(best, 0.99)), "mean_parts_used": float(parts_used.mean()),
                         "bound_B": tau if name == "cumulative_cap" else "",
                         "bound_holds": bool(best.max() <= tau + 1e-6) if name == "cumulative_cap" else ""})
        log(f"  e3 {name}: K=1 value {rows[-len(KS)]['fraud_value_accepted']:.3f} "
            f"K=64 value {rows[-1]['fraud_value_accepted']:.3f}, max exposure {rows[-1]['max_exposure']:.0f}")
    files = [write_csv(os.path.join(out, "e3_splitting.csv"), rows)]

    sweep = []
    for q in (0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999):
        Bq = float(np.quantile(c_all[va_leg], q))
        held_c = float((c_all[te_leg] > Bq).mean())
        Aq = float(np.quantile(amount[va_leg], q))
        held_a = float((amount[te_leg] > Aq).mean())
        # attacker with K = 64 against each cap
        st_val = float(np.where(fa / 64 <= Aq, fa, 0).sum() / fa.sum())
        cu = np.minimum(fa, np.maximum(Bq - fprior, 0))
        sweep.append({"quantile": q, "B": Bq, "legit_held_cumulative": held_c, "max_exposure_cumulative": float(cu.max()),
                      "fraud_value_accepted_cumulative": float(cu.sum() / fa.sum()), "A": Aq,
                      "legit_held_stateless": held_a, "fraud_value_accepted_stateless_K64": st_val})
    files.append(write_csv(os.path.join(out, "e3_bound_sweep.csv"), sweep))
    files.append(write_csv(os.path.join(out, "e3_atomicity.csv"), atomicity(cfg)))
    return files


def atomicity(cfg) -> list[dict]:
    """Algorithm 3 end to end: issuer reserves, signs a PID-bound claim, receiver verifies and accepts iff
    c <= B. 32 parallel parts of 1,000 against B = 10,000, repeated."""
    trust = TrustRegistry()
    key = SigningKey()
    trust.register("BANK1", key, "issuer", 0, 10 ** 10)
    rows = []
    for mode, cls in (("atomic", cumulative.SpendCounter), ("racy", cumulative.RacyCounter)):
        over, worst, lat = 0, 0, []
        trials = cfg.get("e3_atomic_trials", 40)
        for t in range(trials):
            ctr = cls()
            acc = []
            lock = threading.Lock()
            now = 1_000_000

            def go(j):
                t0 = time.perf_counter()
                c = ctr.reserve("mandate-1", cumulative.window_of(now, 86_400), 1_000)
                pid, nonce = f"pid-{t}-{j}", f"n-{t}-{j}"
                claim = cumulative.issue_claim(key, "BANK1", "mandate-1", cumulative.window_of(now, 86_400), c, pid, nonce)
                got = cumulative.verify_claim(claim, trust, now, pid, nonce, "BANK1", 86_400)
                if got <= 10_000:
                    with lock:
                        acc.append(1_000)
                else:
                    ctr.release("mandate-1", cumulative.window_of(now, 86_400), 1_000)
                with lock:
                    lat.append((time.perf_counter() - t0) * 1e3)

            ts = [threading.Thread(target=go, args=(j,)) for j in range(32)]
            for th in ts:
                th.start()
            for th in ts:
                th.join()
            tot = sum(acc)
            over += tot > 10_000
            worst = max(worst, tot)
        p = percentiles(lat)
        rows.append({"counter": mode, "trials": trials, "parallel_parts": 32, "B": 10_000,
                     "trials_over_bound": over, "worst_accepted_total": worst,
                     "claim_ms_p50": p["p50"], "claim_ms_p99": p["p99"]})
    return rows
