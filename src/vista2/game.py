"""Control-typed attested evidence games on finite presentation spaces (Section 2, Theorems 1-3).

A presentation space is a finite set of inputs z with a reachability relation z -> z' ("a holder whose
true input is z can present z'"). The relation is the reflexive-transitive closure of the allowed
manipulations, so it is a preorder. A decision rule d assigns each input an ordered decision
(0 = reject < 1 = step-up < 2 = accept, or any ordered integers).

  realized(d)   d*(z) = max_{z -> z'} d(z')       what a best-responding holder obtains
  repair(d)     d_(z) = min_{y -> z} d(y)          the largest strategy-proof rule below d
  is_sp(d)      no z -> z' with d(z') > d(z)       Definition 2

Everything here is exact on explicit finite spaces and is used by the verification experiment (E0) and
the tests. The learned-model versions of the same operators are in repair.py and attack.py.
"""
from __future__ import annotations

import itertools

import numpy as np


# ----------------------------------------------------------------------------- preorders
def transitive_closure(R: np.ndarray) -> np.ndarray:
    """Reflexive-transitive closure of a boolean relation (Warshall)."""
    C = np.asarray(R, dtype=bool).copy()
    np.fill_diagonal(C, True)
    for k in range(len(C)):
        C |= C[:, k:k + 1] & C[k:k + 1, :]
    return C


def is_preorder(R: np.ndarray) -> bool:
    R = np.asarray(R, dtype=bool)
    if not R.diagonal().all():
        return False
    return bool(((R.astype(np.int64) @ R.astype(np.int64) > 0) <= R).all())


def product_reach(parts: list[np.ndarray]) -> np.ndarray:
    """Reachability on a product space from per-coordinate relations (coordinates move independently)."""
    R = np.ones((1, 1), dtype=bool)
    for P in parts:
        R = np.kron(R, np.asarray(P, dtype=bool)).astype(bool)
    return R


def chain_down(levels: int) -> np.ndarray:
    """W coordinate: from level a the holder can present any b <= a."""
    a = np.arange(levels)
    return a[:, None] >= a[None, :]


def complete(values: int) -> np.ndarray:
    """F coordinate where every value is reachable from every other (e.g. the hour of payment)."""
    return np.ones((values, values), dtype=bool)


def fixed(values: int) -> np.ndarray:
    """T coordinate: nobody on the sender side can change it."""
    return np.eye(values, dtype=bool)


def split_down(values: int) -> np.ndarray:
    """Amount on an increasing grid under splitting: a payment can be presented as any smaller amount."""
    return chain_down(values)


# ----------------------------------------------------------------------------- decision operators
def realized(d: np.ndarray, R: np.ndarray) -> np.ndarray:
    """d*(z) = max over z -> z' of d(z')."""
    return np.where(R, np.asarray(d)[None, :], np.iinfo(np.int64).min).max(1)


def repair(d: np.ndarray, R: np.ndarray) -> np.ndarray:
    """d_(z) = min over y -> z of d(y): decide as if the truth were the least favourable input that could
    have produced what is shown."""
    return np.where(R, np.asarray(d)[:, None], np.iinfo(np.int64).max).min(0)


def sp_violations(d: np.ndarray, R: np.ndarray) -> int:
    """Number of pairs z -> z' with d(z') > d(z) (Definition 2: the adversary wins on any of them)."""
    d = np.asarray(d)
    return int((R & (d[None, :] > d[:, None])).sum())


def is_sp(d: np.ndarray, R: np.ndarray) -> bool:
    return sp_violations(d, R) == 0


def is_monotone(d: np.ndarray, R: np.ndarray) -> bool:
    """Monotone in the presentation order: z -> z' implies d(z) >= d(z')."""
    return is_sp(d, R)


def evaluation_gap(d: np.ndarray, R: np.ndarray, p: np.ndarray, accept: int) -> dict:
    """Theorem 2 on a distribution p over inputs: evaluated vs realized P(decision = accept)."""
    ds = realized(d, R)
    ev = float(p @ (np.asarray(d) >= accept))
    re = float(p @ (ds >= accept))
    return {"evaluated": ev, "realized": re, "gap": re - ev}


def all_sp_rules_below(d: np.ndarray, R: np.ndarray, levels: int):
    """Brute force: every strategy-proof rule g with g <= d pointwise (tiny spaces only)."""
    d = np.asarray(d)
    for g in itertools.product(*[range(int(v) + 1) for v in d]):
        g = np.array(g)
        if g.max(initial=0) < levels and is_sp(g, R):
            yield g


# ----------------------------------------------------------------------------- Theorem 3 (splitting)
def split_exposure_stateless(accepts_amount, amount: int, k_max: int) -> tuple[int, int]:
    """Largest total a holder moves through a stateless amount rule by splitting `amount` into k <= k_max
    equal parts (each part must be accepted). Returns (accepted_total, parts_used)."""
    best = (0, 0)
    for k in range(1, k_max + 1):
        part = amount // k
        if part > 0 and accepts_amount(part):
            tot = part * k
            if tot > best[0]:
                best = (tot, k)
    return best


def split_exposure_cumulative(bound: int, parts: list[int], prior_spend: int = 0) -> int:
    """Accepted total under a rule that accepts iff the issuer counter c (including the current part) is
    at most `bound`. Parts arrive in order; the issuer reserves before the receiver decides and releases
    the reservation of a rejected part, so the counter never falls below the accepted total."""
    c, acc = prior_spend, 0
    for a in parts:
        c += a
        if c <= bound:
            acc += a
        else:
            c -= a
    return acc
