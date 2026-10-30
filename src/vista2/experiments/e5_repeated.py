"""E5: how fast does fresh selection leak across repeated agent payments? (Theorems 4 and 5)

Agents pay the same merchants repeatedly, so the receiver sees many disclosures from one holder. The
receiver's policy is a monotone rule over the delegated-authority chains; the holder discloses one
satisfied branch per payment. The prior over chain profiles is the empirical distribution of legitimate
agent payments in the agentic workload.

Part 1 (exact): for three receiver policies, the zero-leakage test of Theorem 4 and, for DET, the
exponential mechanism with fresh draws (lambda in {0.5, 1, 2}) and sticky selection, the selection leakage
Lambda_n, I(X; V^n) and the Bayes branch-inference accuracy for n = 1..10, with the coverage bound of
Theorem 5(1) against the exact probability.

Part 2 (simulation): 4,000 agent-merchant subscriptions of 12 payments each, evidence epochs that change
with probability 0.1 per payment, and an empirical attacker (trained on half the subscriptions, tested on
the other half) predicting whether the cheapest branch is unavailable from (payment count, branches seen,
last branch). Reported per payment count: distinct branches revealed, attacker accuracy and lift.
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

from .. import datasets
from ..claims import AGENT_SCHEMA
from ..disclosure import repeated, space
from ..util import sha256, write_csv

# level indices: mandate_kind, scope_coverage, agent_identity, tool_policy, user_reconfirmation
POLICIES = {
    "P1_three_branch": [(2, 0, 1, 0, 0), (1, 0, 0, 2, 1), (0, 3, 2, 0, 0)],
    "P2_two_branch": [(2, 0, 0, 0, 2), (0, 2, 0, 2, 0)],
    "P3_four_branch": [(2, 2, 0, 0, 0), (1, 0, 1, 2, 0), (0, 3, 0, 0, 2), (0, 0, 2, 1, 1)],
}
LAMBDAS = (0.5, 1.0, 2.0)


def prior_from_workload(cfg):
    t = datasets.load("agentic", cfg.get("agentic_size"))
    cols = AGENT_SCHEMA.names
    m = (t.y == 0) & (t.X["is_agent"] == 1) & (t.X[[f"cap_{c}" for c in cols]].min(1) == 1)
    return space.empirical_prior(t.X.loc[m, cols].to_numpy(), AGENT_SCHEMA.dims)


def part1(states, p, nmax: int) -> tuple[list, list]:
    rows, zl = [], []
    for name, br in POLICIES.items():
        front = space.frontier(br)
        ss, _ = space.build(states, p, front)
        z = ss.zero_leakage_system()
        zl.append({"policy": name, "branches": len(front), "signatures": ss.n_signatures,
                   "zero_leakage_feasible": z["feasible"], "residual": z["residual"],
                   "two_branch_obstruction": ss.two_branch_obstruction() is not None,
                   "accept_mass": float(p[(states[:, None, :] >= front[None]).all(2).any(1)].sum())})
        kernels = {"DET": ss.kernel_det(), "uniform_fresh": ss.kernel_uniform()}
        for lam in LAMBDAS:
            kernels[f"exp{lam}_fresh"] = ss.kernel_exp(lam)
        for kname, Q in kernels.items():
            qm = repeated.q_min(ss, Q)
            for n in range(1, nmax + 1):
                if kname == "DET":
                    met = repeated.sticky_metrics(ss, Q, n)
                else:
                    met = repeated.repeated_metrics(ss, Q, n)
                rows.append({"policy": name, "kernel": kname, "n": n, "Lambda": met["Lambda"], "I_bits": met["I_bits"],
                             "E_L_bits": met["E_L_bits"], "branch_attack_acc": met.get("branch_attack_acc"),
                             "branch_attack_base": met.get("branch_attack_base"), "q_min": qm})
            if kname != "DET":
                st = repeated.sticky_metrics(ss, Q, nmax)
                for n in range(1, nmax + 1):
                    rows.append({"policy": name, "kernel": kname.replace("fresh", "sticky"), "n": n,
                                 "Lambda": st["Lambda"], "I_bits": st["I_bits"], "E_L_bits": st["E_L_bits"],
                                 "branch_attack_acc": st.get("branch_attack_acc"),
                                 "branch_attack_base": st.get("branch_attack_base"), "q_min": qm})
                if qm > 0:
                    nb = repeated.coverage_payments(qm, len(front), 0.05)
                    worst = min(repeated.prob_all_seen(Q[a], nb) for a in range(ss.n_signatures))
                    zl.append({"policy": name, "kernel": kname, "coverage_bound_payments": nb,
                               "min_prob_all_seen_at_bound": worst, "bound_holds": worst >= 0.95 - 1e-12})
    return rows, zl


def part2(states, p, cfg, rng) -> list[dict]:
    rows = []
    n_subs, n_pay = cfg.get("e5_subscriptions", 4000), cfg.get("e5_payments", 12)
    for name, br in POLICIES.items():
        front = space.frontier(br)
        C = space.surprisal_costs(states, p, front)
        cheapest = int(np.argmin(C))
        avail_all = (states[:, None, :] >= front[None]).all(2)
        ok = avail_all.any(1)
        pp = np.where(ok, p, 0.0)
        pp = pp / pp.sum()
        expw = np.exp(-1.0 * (C - C.min()))
        store = repeated.StickyStore(sha256(f"e5|{name}".encode()))
        stats = defaultdict(lambda: np.zeros(2))
        obs = defaultdict(lambda: np.zeros(2))
        for sub in range(n_subs):
            half = sub % 2
            x = rng.choice(len(states), p=pp)
            epoch = 0
            seen = {k: 0 for k in ("DET", "RAND_fresh", "RAND_sticky")}
            for n in range(1, n_pay + 1):
                if n > 1 and rng.random() < 0.1:
                    x = rng.choice(len(states), p=pp)
                    epoch += 1
                av = avail_all[x]
                y = int(not av[cheapest])
                pr = np.where(av, expw, 0.0)
                pr = pr / pr.sum()
                ch = {"DET": int(np.flatnonzero(av)[np.argmin(C[av])]),
                      "RAND_fresh": int(rng.choice(len(front), p=pr)),
                      "RAND_sticky": store.choose((sub, name, epoch), av, pr)[0]}
                for k, b in ch.items():
                    seen[k] |= 1 << b
                    stats[(k, n)] += (1, bin(seen[k]).count("1"))
                    obs[(k, n, seen[k], b, half, y)] += (1, 0)
        for k in ("DET", "RAND_fresh", "RAND_sticky"):
            for n in range(1, n_pay + 1):
                table = defaultdict(lambda: np.zeros(2))
                base = np.zeros(2)
                for (kk, nn, sn, b, half, y), c in obs.items():
                    if kk == k and nn == n and half == 0:
                        table[(sn, b)][y] += c[0]
                        base[y] += c[0]
                hit = hb = tot = 0.0
                for (kk, nn, sn, b, half, y), c in obs.items():
                    if kk == k and nn == n and half == 1:
                        tb = table.get((sn, b))
                        pred = int(tb[1] > tb[0]) if tb is not None and tb.sum() else int(base[1] > base[0])
                        hit += c[0] * (pred == y)
                        hb += c[0] * (int(base[1] > base[0]) == y)
                        tot += c[0]
                s = stats[(k, n)]
                rows.append({"policy": name, "strategy": k, "n": n, "payments": int(s[0]),
                             "mean_distinct_branches": s[1] / s[0], "attacker_acc": hit / tot,
                             "attacker_base": hb / tot, "attacker_lift": (hit - hb) / tot})
    return rows


def run(cfg, out, log=print):
    rng = np.random.default_rng(cfg.get("seed", 1) + 5)
    states, p = prior_from_workload(cfg)
    log(f"  e5: empirical prior over {len(states)} chain profiles")
    rows, zl = part1(states, p, cfg.get("e5_nmax", 10))
    sim = part2(states, p, cfg, rng)
    return [write_csv(os.path.join(out, "e5_repeated_exact.csv"), rows),
            write_csv(os.path.join(out, "e5_zero_leakage_and_coverage.csv"), zl),
            write_csv(os.path.join(out, "e5_repeated_simulation.csv"), sim)]
