"""Receiver policies over claim chains as frontiers, and the signature space they induce on a prior."""
from __future__ import annotations

import numpy as np

from .signature import SignatureSpace


def frontier(branches) -> np.ndarray:
    """Minimal elements of a set of profiles (the frontier of a monotone DNF policy)."""
    B = np.unique(np.asarray(branches, dtype=np.int64), axis=0)
    keep = [i for i in range(len(B)) if not any(j != i and (B[j] <= B[i]).all() for j in range(len(B)))]
    return B[keep]


def surprisal_costs(states: np.ndarray, p: np.ndarray, front: np.ndarray) -> np.ndarray:
    """Disclosure cost of each branch: -log2 P(X >= v) under the prior."""
    out = []
    for v in front:
        m = p[(states >= v).all(1)].sum()
        out.append(-np.log2(m) if m > 0 else np.inf)
    return np.array(out)


def build(states: np.ndarray, p: np.ndarray, front: np.ndarray, cost: np.ndarray | None = None) -> tuple:
    """Returns (SignatureSpace, avail matrix states x branches)."""
    avail = (states[:, None, :] >= front[None, :, :]).all(2)
    C = surprisal_costs(states, p, front) if cost is None else np.asarray(cost, dtype=float)
    cheapest = int(np.argmin(C))
    targets = {"branch": (~avail[:, cheapest]).astype(float)}
    return SignatureSpace.from_states(avail, p, C, targets), avail


def empirical_prior(levels: np.ndarray, dims) -> tuple[np.ndarray, np.ndarray]:
    """Distinct observed profiles and their frequencies."""
    st, cnt = np.unique(np.asarray(levels, dtype=np.int64), axis=0, return_counts=True)
    return st, cnt / cnt.sum()
