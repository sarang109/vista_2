"""E0: numerical verification of the theory (Theorems 1-5, Corollary 1, Algorithms 2b and 3, box repair).

These are checks, not proofs: the proofs are in docs/theory.md. Each row of e0_verification.csv states
what was checked, on how many instances, and how many failed.
"""
from __future__ import annotations

import itertools
import math
import os
import threading

import numpy as np

from .. import game
from ..cumulative import RacyCounter, SpendCounter
from ..disclosure import repeated
from ..disclosure.signature import SignatureSpace
from ..repair import lattice_dp, lattice_dp_reference
from ..util import write_csv


def _random_preorder(rng, n):
    R = rng.random((n, n)) < rng.uniform(0.05, 0.35)
    return game.transitive_closure(R)


def check_theorem1(rng, instances: int, brute_max: int = 6) -> list[dict]:
    f = {"dstar_ge_d": 0, "dstar_monotone": 0, "sp_iff_fixed_point": 0, "sp_iff_monotone": 0,
         "repair_sp": 0, "repair_below_d": 0, "repair_largest": 0, "repair_fixes_sp": 0, "closure_preorder": 0,
         "dstar_idempotent": 0}
    brute = 0
    for _ in range(instances):
        n = int(rng.integers(2, 10))
        R = _random_preorder(rng, n)
        f["closure_preorder"] += not game.is_preorder(R)
        d = rng.integers(0, 3, n)
        ds = game.realized(d, R)
        f["dstar_ge_d"] += not (ds >= d).all()
        f["dstar_monotone"] += not game.is_monotone(ds, R)
        f["dstar_idempotent"] += not (game.realized(ds, R) == ds).all()
        sp = game.is_sp(d, R)
        f["sp_iff_fixed_point"] += sp != bool((ds == d).all())
        f["sp_iff_monotone"] += sp != game.is_monotone(d, R)
        rep = game.repair(d, R)
        f["repair_sp"] += not game.is_sp(rep, R)
        f["repair_below_d"] += not (rep <= d).all()
        f["repair_fixes_sp"] += sp and not (rep == d).all()
        if n <= brute_max:
            brute += 1
            for g in game.all_sp_rules_below(d, R, 3):
                if not (g <= rep).all():
                    f["repair_largest"] += 1
                    break
    rows = [{"id": "T1." + k, "check": k, "instances": instances, "failures": v} for k, v in f.items()]
    rows[[r["check"] for r in rows].index("repair_largest")]["instances"] = brute
    return rows


def check_corollary1() -> dict:
    """Space W chain(3) x F complete(3) x T fixed(2), binary decisions: enumerate all 2^18 rules and check
    that SP holds exactly for rules non-decreasing in W, constant in F and arbitrary in T."""
    R = game.product_reach([game.chain_down(3), game.complete(3), game.fixed(2)])
    states = list(itertools.product(range(3), range(3), range(2)))
    idx = {s: i for i, s in enumerate(states)}
    mism = 0
    total = 0
    for bits in range(1 << len(states)):
        d = np.array([(bits >> i) & 1 for i in range(len(states))])
        sp = game.is_sp(d, R)
        cond = True
        for w, f, t in states:
            if w > 0 and d[idx[(w, f, t)]] < d[idx[(w - 1, f, t)]]:
                cond = False
            if d[idx[(w, f, t)]] != d[idx[(w, 0, t)]]:
                cond = False
        mism += sp != cond
        total += 1
    return {"id": "C1.enumeration", "check": "SP iff monotone in W, constant in complete F, free in T",
            "instances": total, "failures": mism}


def check_theorem2(rng, instances: int) -> list[dict]:
    neg = eq = 0
    for _ in range(instances):
        n = int(rng.integers(2, 10))
        R = _random_preorder(rng, n)
        d = rng.integers(0, 3, n)
        p = rng.dirichlet(np.ones(n))
        g = game.evaluation_gap(d, R, p, 2)
        neg += g["gap"] < -1e-12
        # equality iff no supported input below "accept" can reach an accepted presentation
        p[rng.random(n) < 0.3] = 0.0
        p = p / p.sum() if p.sum() > 0 else np.full(n, 1.0 / n)
        g = game.evaluation_gap(d, R, p, 2)
        gains = (p > 0) & (d < 2) & (game.realized(d, R) >= 2)
        eq += (abs(g["gap"]) < 1e-12) != (not gains.any())
    return [{"id": "T2.gap_nonnegative", "check": "realized >= evaluated acceptance", "instances": instances,
             "failures": neg},
            {"id": "T2.equality", "check": "gap = 0 iff no supported non-accepted input reaches an accepted presentation",
             "instances": instances, "failures": eq}]


def check_theorem3(rng, instances: int) -> list[dict]:
    bad_imp = bad_suf = 0
    for _ in range(instances):
        a = int(rng.integers(1_000, 100_000))
        a_small = int(rng.integers(1, a))
        k = a // a_small + 1
        tot, parts = game.split_exposure_stateless(lambda x: x <= a_small, k * a_small, k)
        bad_imp += not (tot == k * a_small and tot > a and parts == k)
        B = int(rng.integers(1_000, 50_000))
        seq = list(rng.integers(1, 5_000, int(rng.integers(1, 60))))
        prior = int(rng.integers(0, 2_000))
        bad_suf += game.split_exposure_cumulative(B, seq, prior) > max(B - prior, 0)
    return [{"id": "T3.1.impossibility", "check": "k parts of an accepted amount move more than a rejected amount",
             "instances": instances, "failures": bad_imp},
            {"id": "T3.2.sufficiency", "check": "accepted total per window <= B - prior spend",
             "instances": instances, "failures": bad_suf}]


def check_atomicity(rng, trials: int, threads: int = 16, bound: int = 10_000, part: int = 1_000) -> list[dict]:
    """Algorithm 3: parallel requests against one key; the receiver accepts iff c <= bound."""
    rows = []
    for name, cls in (("atomic", SpendCounter), ("racy", RacyCounter)):
        over = worst = 0
        for t in range(trials):
            ctr = cls()
            accepted = []
            lock = threading.Lock()

            def go():
                c = ctr.reserve("K", 0, part)
                if c <= bound:
                    with lock:
                        accepted.append(part)
                else:
                    ctr.release("K", 0, part)

            ts = [threading.Thread(target=go) for _ in range(threads)]
            for th in ts:
                th.start()
            for th in ts:
                th.join()
            tot = sum(accepted)
            over += tot > bound
            worst = max(worst, tot)
        rows.append({"id": f"A3.{name}", "check": f"{name} counter: accepted total <= B under {threads} parallel parts",
                     "instances": trials, "failures": over, "max_error": worst - bound,
                     "note": f"B={bound}, part={part}; worst accepted total {worst}"})
    return rows


def check_lattice_dp(rng, instances: int) -> dict:
    bad = 0
    err = 0.0
    for _ in range(instances):
        dims = tuple(int(v) for v in rng.integers(2, 5, int(rng.integers(1, 5))))
        s = rng.random(dims)
        a, b = lattice_dp(s), lattice_dp_reference(s)
        brute = np.empty(dims)
        for w in itertools.product(*[range(d) for d in dims]):
            brute[w] = s[tuple(slice(i, None) for i in w)].max()
        e = max(np.abs(a - b).max(), np.abs(a - brute).max())
        err = max(err, float(e))
        bad += e > 1e-12
    return {"id": "A2b.dp", "check": "lattice DP = recursion = max over up-set", "instances": instances,
            "failures": bad, "max_error": err}


def check_box_repair(rng, n_rows: int = 300) -> list[dict]:
    from ..control import variants
    from ..datasets import synthetic
    from ..experiments.common import fit_variant
    from ..repair import ExactRepair, TreeBoxRepair
    t = synthetic.generate(n=40_000, seed=11)
    v = [x for x in variants(t, ("unconstrained",))][0]
    tr = fit_variant(t, v, "lightgbm", seed=0, rounds=60)
    X = t.X.iloc[rng.choice(len(t.X), n_rows, replace=False)]
    m = tr.model
    rows = []
    point = TreeBoxRepair(m, t, v.columns, types=())
    e = float(np.abs(point.raw(X) - m.raw(X)).max())
    rows.append({"id": "BOX.point", "check": "box repair with no moving coordinates equals the model",
                 "instances": n_rows, "failures": int(e > 1e-6), "max_error": e})
    box_w = TreeBoxRepair(m, t, v.columns, types=("W",))
    ex_w = ExactRepair(m.risk, t, v.columns, types=("W",))
    rb, re = box_w.risk(X), ex_w.risk(X)
    rows.append({"id": "BOX.upper", "check": "box repair >= exact repair (W lattice)", "instances": n_rows,
                 "failures": int((rb < re - 1e-9).sum()), "max_error": float(np.abs(rb - re).max()),
                 "note": f"mean excess {float((rb - re).mean()):.4g}"})
    rows.append({"id": "BOX.exact_ge_model", "check": "exact repair >= model risk", "instances": n_rows,
                 "failures": int((re < m.risk(X) - 1e-9).sum())})
    # monotonicity: weakening a W coordinate (z -> z') never lowers the repaired risk
    W = t.cols("W")
    Xp = X.copy()
    for c in W:
        Xp[c] = np.floor(Xp[c] * rng.random(len(Xp)))
    viol_b = int((box_w.risk(Xp) < rb - 1e-9).sum())
    viol_e = int((ex_w.risk(Xp) < re - 1e-9).sum())
    rows.append({"id": "BOX.monotone", "check": "repaired risk does not fall when claims are withheld",
                 "instances": 2 * n_rows, "failures": viol_b + viol_e})
    return rows


def _random_space(rng, m):
    k = 2 ** m - 1
    sig = np.array([[(a >> j) & 1 for j in range(m)] for a in range(1, k + 1)], dtype=bool)
    keep = rng.random(k) < 0.7
    keep[rng.integers(k)] = True
    mass = np.where(keep, rng.random(k), 0.0)
    return SignatureSpace(sig, mass, rng.uniform(1, 5, m))


def check_theorems45(rng, instances: int) -> list[dict]:
    obst_bad = mi_bad = sticky_bad = cov_bad = zl_bad = 0
    for _ in range(instances):
        m = int(rng.integers(2, 5))
        ss = _random_space(rng, m)
        zl = ss.zero_leakage_system()
        if ss.two_branch_obstruction() is not None and zl["feasible"]:
            obst_bad += 1
        if zl["feasible"]:
            Q = ss.zero_leakage_kernel(zl["w"])
            zl_bad += ss.lam(Q) > 1e-9
        Q = ss.kernel_exp(float(rng.uniform(0.2, 2.0)))
        one = repeated.repeated_metrics(ss, Q, 1)
        for n in (2, 4):
            mn = repeated.repeated_metrics(ss, Q, n)
            mi_bad += mn["I_bits"] > n * one["I_bits"] + 1e-9
        st = repeated.sticky_metrics(ss, Q, 6)
        sticky_bad += abs(st["Lambda"] - one["Lambda"]) > 1e-12
        q = Q[0]
        qm = float(q[q > 0].min())
        delta = 0.05
        n_cov = repeated.coverage_payments(qm, int((q > 0).sum()), delta)
        cov_bad += repeated.prob_all_seen(q, n_cov) < 1 - delta - 1e-12
    return [{"id": "T4.obstruction", "check": "two-branch obstruction implies no zero-leakage kernel",
             "instances": instances, "failures": obst_bad},
            {"id": "T4.zero_kernel", "check": "LP solution gives Lambda = 0", "instances": instances, "failures": zl_bad},
            {"id": "T5.mi_bound", "check": "I(X;V^n) <= n I(X;V)", "instances": 2 * instances, "failures": mi_bad},
            {"id": "T5.sticky", "check": "sticky selection keeps Lambda_n = Lambda_1", "instances": instances,
             "failures": sticky_bad},
            {"id": "T5.coverage", "check": "ln(m/delta)/q_min payments reveal every branch w.p. >= 1 - delta",
             "instances": instances, "failures": cov_bad}]


def run(cfg, out):
    rng = np.random.default_rng(cfg.get("seed", 1))
    k = cfg.get("e0_instances", 2000)
    rows = check_theorem1(rng, k)
    rows.append(check_corollary1())
    rows += check_theorem2(rng, k)
    rows += check_theorem3(rng, k)
    rows += check_atomicity(rng, cfg.get("e0_atomic_trials", 50))
    rows.append(check_lattice_dp(rng, min(k, 300)))
    rows += check_box_repair(rng, cfg.get("e0_box_rows", 300))
    rows += check_theorems45(rng, min(k, 300))
    for r in rows:
        r.setdefault("max_error", "")
        r.setdefault("note", "")
        r["passed"] = r["failures"] == 0 if not r["id"].startswith("A3.racy") else "expected to fail"
    return [write_csv(os.path.join(out, "e0_verification.csv"), rows)]
