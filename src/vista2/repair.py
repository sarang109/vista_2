"""Safe repair of a trained risk model (Algorithm 2b and a tree-ensemble form of it).

For a decision rule d the repaired rule is d_(z) = min over y -> z of d(y): decide as if the truth were the
least favourable input that could have produced what is shown. On a risk score s (accept iff s < tau)
this is s_bar(z) = max over y -> z of s(y). It is monotone in the presentation order, never accepts more
than d, and is the largest strategy-proof rule below d.

The set {y : y -> z} is a product of per-coordinate up-sets:
  W chain   levels >= the shown level            W poset   everything if nothing is shown, else the value
  F split   amounts >= the shown amount          F complete / step   every value of the coordinate
  T         the shown value

`lattice_dp`        Algorithm 2b on a full chain lattice: s_bar[w] = max(s[w], max_i s_bar[w + e_i]).
`ExactRepair`       any black-box scorer: enumerate the up-set (with the F grids) and take the maximum.
                    Cost grows with the lattice; for nine four-level claims it is 262,144 points.
`TreeBoxRepair`     LightGBM: replace the up-set by its bounding box and take, tree by tree, the largest
                    leaf the box can reach. sum_trees max >= max sum_trees, so the score is an upper bound
                    on s_bar; it is still monotone (a larger box can only reach more leaves), so the rule is
                    strategy-proof, and it is exact when no two trees interact on the moved coordinates.
                    Cost is linear in trees x leaves per query, independent of the lattice size.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .control import TypedTable


# ----------------------------------------------------------------------------- Algorithm 2b
def lattice_dp(scores: np.ndarray) -> np.ndarray:
    """scores: array of shape dims (one axis per chain claim). Returns s_bar of the same shape."""
    s = np.array(scores, dtype=float, copy=True)
    for ax in range(s.ndim):
        # suffix maximum along each axis; composing them gives the max over the whole up-set
        s = np.flip(np.maximum.accumulate(np.flip(s, ax), axis=ax), ax)
    return s


def lattice_dp_reference(scores: np.ndarray) -> np.ndarray:
    """Literal recursion of Algorithm 2b in decreasing lattice order (for tests)."""
    dims = scores.shape
    out = np.full(dims, -np.inf)
    for w in sorted(itertools.product(*[range(d) for d in dims]), key=sum, reverse=True):
        v = scores[w]
        for i in range(len(dims)):
            if w[i] + 1 < dims[i]:
                up = list(w)
                up[i] += 1
                v = max(v, out[tuple(up)])
        out[w] = v
    return out


# ----------------------------------------------------------------------------- up-sets
def upset_values(t: TypedTable, col: str, shown: float, levels: dict) -> np.ndarray:
    s = t.specs[col]
    if s.ctype == "W":
        top = levels[col]
        if s.reach == "poset":
            return top if shown == 0 else np.array([shown])
        return np.arange(int(shown), int(top) + 1, dtype=float)
    if s.ctype == "F":
        g = np.asarray(s.values, dtype=float)
        if s.reach == "split":
            g = np.unique(np.concatenate([[shown], g[g >= shown]]))
            return g
        return np.unique(np.concatenate([[shown], g]))
    return np.array([shown])


def _levels(t: TypedTable, columns) -> dict:
    out = {}
    for c in columns:
        s = t.specs[c]
        if s.ctype == "W":
            vals = np.unique(t.X[c].to_numpy(float))
            out[c] = vals[vals > 0] if s.reach == "poset" else float(vals.max())
            if s.reach == "poset":
                out[c] = np.unique(np.concatenate([[0.0], out[c]]))
    return out


class ExactRepair:
    """s_bar(z) = max over the reachable-from set of z, by enumeration. Works for any scorer."""

    def __init__(self, risk_fn, t: TypedTable, columns, types=("W", "F"), chunk: int = 400_000):
        self.risk_fn, self.t, self.columns, self.chunk = risk_fn, t, list(columns), chunk
        self.levels = _levels(t, self.columns)
        self.moving = [c for c in self.columns if t.specs[c].ctype in types]
        self.calls = 0

    def upset_size(self, row) -> int:
        return int(np.prod([len(upset_values(self.t, c, float(row[c]), self.levels)) for c in self.moving]))

    def risk(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.columns].reset_index(drop=True)
        out = np.empty(len(X))
        # group rows by their up-set signature on moving coordinates to share enumeration work
        key = X[self.moving].astype(float).round(6).apply(tuple, axis=1)
        for k, idx in X.groupby(key).groups.items():
            idx = np.asarray(idx)
            rows = X.loc[idx]
            sets = [upset_values(self.t, c, v, self.levels) for c, v in zip(self.moving, k)]
            combos = np.array(list(itertools.product(*sets)), dtype=float)
            best = np.full(len(idx), -np.inf)
            per = max(1, self.chunk // max(len(idx), 1))
            for a in range(0, len(combos), per):
                cb = combos[a:a + per]
                big = rows.loc[rows.index.repeat(len(cb))].reset_index(drop=True)
                big[self.moving] = np.tile(cb, (len(idx), 1))
                r = self.risk_fn(big).reshape(len(idx), len(cb))
                self.calls += r.size
                best = np.maximum(best, r.max(1))
            out[idx] = best
        return out


# ----------------------------------------------------------------------------- tree box repair
class TreeBoxRepair:
    """Box-relaxed repair of a LightGBM model (see module docstring)."""

    def __init__(self, model, t: TypedTable, columns, types=("W", "F")):
        self.model, self.t, self.types = model, t, tuple(types)
        self.columns = list(model.columns)
        self.levels = _levels(t, self.columns)
        self.trees = self._parse(model.booster.dump_model())

    def _parse(self, dump):
        trees = []
        p = len(self.columns)
        for ti in dump["tree_info"]:
            leaves_lo, leaves_hi, vals = [], [], []

            def walk(node, lo, hi):
                if "leaf_value" in node or "split_feature" not in node:
                    leaves_lo.append(lo.copy()); leaves_hi.append(hi.copy())
                    vals.append(float(node.get("leaf_value", 0.0)))
                    return
                if node.get("decision_type", "<=") != "<=":
                    raise ValueError("categorical splits are not supported by the box repair")
                f, th = int(node["split_feature"]), float(node["threshold"])
                l_hi = hi.copy(); l_hi[f] = min(hi[f], th)
                walk(node["left_child"], lo, l_hi)
                r_lo = lo.copy(); r_lo[f] = max(lo[f], th)
                walk(node["right_child"], r_lo, hi)

            walk(ti["tree_structure"], np.full(p, -np.inf), np.full(p, np.inf))
            lo, hi = np.array(leaves_lo), np.array(leaves_hi)
            used = np.flatnonzero(np.isfinite(lo).any(0) | np.isfinite(hi).any(0))
            trees.append((used, lo[:, used], hi[:, used], np.array(vals)))
        return trees

    def boxes(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        A = X[self.columns].to_numpy(float)
        lo, hi = A.copy(), A.copy()
        for j, c in enumerate(self.columns):
            s = self.t.specs[c]
            if s.ctype not in self.types:
                continue
            if s.ctype == "W":
                if s.reach == "poset":
                    top = self.levels[c]
                    nothing = A[:, j] == 0
                    lo[nothing, j], hi[nothing, j] = top.min(), top.max()
                else:
                    hi[:, j] = self.levels[c]
            elif s.ctype == "F":
                g = np.asarray(s.values, dtype=float)
                hi[:, j] = np.maximum(A[:, j], g.max())
                if s.reach != "split":
                    lo[:, j] = np.minimum(A[:, j], g.min())
        return lo, hi

    def raw(self, X: pd.DataFrame, chunk: int = 4096) -> np.ndarray:
        qlo, qhi = self.boxes(X)
        out = np.zeros(len(X))
        for a in range(0, len(X), chunk):
            lo_q, hi_q = qlo[a:a + chunk], qhi[a:a + chunk]
            acc = np.zeros(len(lo_q))
            for used, llo, lhi, vals in self.trees:
                if len(used) == 0:
                    acc += vals.max()
                    continue
                ql, qh = lo_q[:, used], hi_q[:, used]
                ok = np.ones((len(ql), len(vals)), dtype=bool)
                for k in range(len(used)):
                    ok &= (qh[:, k, None] > llo[None, :, k]) & (ql[:, k, None] <= lhi[None, :, k])
                acc += np.where(ok, vals[None, :], -np.inf).max(1)
            out[a:a + chunk] = acc
        return out

    def risk(self, X: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.raw(X), -40, 40)))
