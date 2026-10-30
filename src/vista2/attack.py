"""Best-response presentation search: the realized decision d*(z) of Theorem 1 for a learned model.

For each payment the attacker looks for the presentation z' with z -> z' that minimises the model's risk.
Allowed moves follow the control types:

  W chain   any level <= the true level (withholding or weakening a signed claim)
  W poset   the true value or "nothing shown"
  F split   divide the amount into k <= split_k equal parts; every part must be accepted, and an
            issuer-signed cumulative claim, if the model reads one, grows with each part
  F complete any value on the feature's grid (hour, free text score, email domain, ...)
  F step    up to `relabel_steps` positions along an ordered category list
  T         never moves

The search is coordinate descent with `rounds` passes, optionally preceded by exhaustive enumeration of
the W lattice when it is small. The truth is always a candidate, so the attacked risk never exceeds the
truthful risk (Theorem 2 holds row by row). Results are lower bounds on attack success unless the search
was exhaustive over the whole reachable set.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .control import TypedTable


@dataclass
class AttackConfig:
    withhold: bool = True
    split_k: int = 4
    complete: bool = True
    relabel_steps: int = 1
    rounds: int = 2
    exhaustive_w_limit: int = 0  # enumerate the W lattice exactly when its size is at most this
    only: set | None = None  # restrict to these coordinates
    name: str = "full"
    # bounded attacker (both off by default): at most `max_f_fields` F fields change per payment, chosen
    # greedily as the fields that lower each payment's risk most; and a completely reachable field may only
    # move to values within a fraction `window` of its grid (by rank) around the true value
    max_f_fields: int | None = None
    window: float | None = None


WITHHOLD_ONLY = AttackConfig(withhold=True, split_k=1, complete=False, relabel_steps=0, name="withholding")
FULL = AttackConfig(name="full")
F_ONLY = AttackConfig(withhold=False, name="f_only")


@dataclass
class AttackResult:
    X: pd.DataFrame
    risk: np.ndarray
    truthful_risk: np.ndarray
    split_parts: np.ndarray
    moved: dict = field(default_factory=dict)  # coordinate -> number of rows whose best value differs


def _coords(t: TypedTable, columns, cfg: AttackConfig):
    out = []
    for c in columns:
        s = t.specs[c]
        if cfg.only is not None and c not in cfg.only:
            continue
        if s.ctype == "W" and cfg.withhold:
            out.append((c, "w_" + s.reach))
        elif s.ctype == "F":
            if s.reach == "split" and cfg.split_k > 1:
                out.append((c, "split"))
            elif s.reach == "complete" and cfg.complete and s.values is not None:
                out.append((c, "complete"))
            elif s.reach == "step" and cfg.relabel_steps > 0 and s.values is not None:
                out.append((c, "step"))
    return out


class _Evaluator:
    """Risk of presentations held as a float array; split parts are scored as the max over the first and
    last part (the cumulative claim grows with each part)."""

    def __init__(self, risk_fn, t: TypedTable, names):
        self.risk_fn, self.names = risk_fn, list(names)
        self.ia = self.names.index(t.amount) if t.amount in self.names else None
        self.ic = self.names.index(t.cumulative) if t.cumulative in self.names else None
        self.calls = 0

    def _score(self, A):
        self.calls += len(A)
        return np.asarray(self.risk_fn(pd.DataFrame(A, columns=self.names)), dtype=float)

    def __call__(self, A: np.ndarray, parts: np.ndarray, amount: np.ndarray, cum) -> np.ndarray:
        A = A.copy()
        if self.ia is not None:
            A[:, self.ia] = amount / parts
        last = self._score(A)
        if self.ic is None or not (parts > 1).any():
            return last
        A[:, self.ic] = cum - amount + amount / parts
        return np.maximum(last, self._score(A))


def best_response(risk_fn, X: pd.DataFrame, t: TypedTable, columns, cfg: AttackConfig = FULL) -> AttackResult:
    names = list(columns)
    for extra in (t.amount, t.cumulative):
        if extra and extra in X.columns and extra not in names:
            names.append(extra)
    A0 = X[names].to_numpy(np.float64)
    n = len(A0)
    col = {c: j for j, c in enumerate(names)}
    ev = _Evaluator(lambda D: risk_fn(D[list(columns)]), t, names)
    amount = A0[:, col[t.amount]].copy() if t.amount in col else np.zeros(n)
    cum = A0[:, col[t.cumulative]].copy() if t.cumulative in col else None
    cur = A0.copy()
    parts = np.ones(n)
    truth = ev(cur, parts, amount, cum)
    best = truth.copy()
    coords = _coords(t, columns, cfg)
    moved = {c: 0 for c, _ in coords}

    def result():
        return AttackResult(pd.DataFrame(cur, columns=names).astype(X[names].dtypes.to_dict(), errors="ignore"),
                            best, truth, parts, moved)

    if n == 0 or not coords:
        return result()

    def try_candidates(cands: list, cand_parts: list, valid: list) -> np.ndarray:
        nonlocal best
        if not cands:
            return np.zeros(n, dtype=bool)
        r = ev(np.concatenate(cands), np.concatenate(cand_parts), np.tile(amount, len(cands)),
               None if cum is None else np.tile(cum, len(cands))).reshape(len(cands), n)
        r = np.where(np.stack(valid), r, np.inf)
        k = r.argmin(0)
        rb = r[k, np.arange(n)]
        imp = rb < best - 1e-12
        if imp.any():
            rows = np.flatnonzero(imp)
            stacked = np.stack(cands)
            cur[rows] = stacked[k[rows], rows]
            pstack = np.stack(cand_parts)
            parts[rows] = pstack[k[rows], rows]
            best = np.where(imp, rb, best)
        return imp

    def with_value(j, v):
        C = cur.copy()
        C[:, j] = v
        return C

    ones = np.ones(n, dtype=bool)
    wcoords = [(c, kind) for c, kind in coords if kind.startswith("w_")]
    if cfg.exhaustive_w_limit and wcoords:
        tops = [int(A0[:, col[c]].max()) for c, _ in wcoords]
        if int(np.prod([v + 1 for v in tops])) <= cfg.exhaustive_w_limit:
            truthW = np.stack([A0[:, col[c]] for c, _ in wcoords], 1)
            cands, cps, val = [], [], []
            for combo in itertools.product(*[range(v + 1) for v in tops]):
                C = cur.copy()
                ok = ones.copy()
                for j, (c, kind) in enumerate(wcoords):
                    C[:, col[c]] = combo[j]
                    if kind == "w_chain":
                        ok &= combo[j] <= truthW[:, j]
                    else:
                        ok &= (combo[j] == 0) | (combo[j] == truthW[:, j])
                cands.append(C); cps.append(parts.copy()); val.append(ok)
                if len(cands) * n > 1_000_000:
                    try_candidates(cands, cps, val)
                    cands, cps, val = [], [], []
            try_candidates(cands, cps, val)

    def candidates(c, kind):
        """Candidate presentations for one coordinate."""
        j = col[c]
        truecol = A0[:, j]
        cands, cps, val = [], [], []
        if kind == "w_chain":
            for lv in range(int(truecol.max()) + 1):
                cands.append(with_value(j, lv)); cps.append(parts.copy()); val.append(lv <= truecol)
        elif kind == "w_poset":
            cands.append(with_value(j, 0.0)); cps.append(parts.copy()); val.append(ones)
            cands.append(with_value(j, truecol)); cps.append(parts.copy()); val.append(ones)
        elif kind == "split":
            for k in range(1, cfg.split_k + 1):
                cands.append(cur.copy()); cps.append(np.full(n, float(k))); val.append(ones)
        elif kind == "complete":
            grid = np.asarray(t.specs[c].values, dtype=float)
            if cfg.window is not None:
                pos = np.clip(np.searchsorted(grid, truecol), 0, len(grid) - 1)
                reach = cfg.window * max(len(grid) - 1, 1)
            for i, v in enumerate(grid):
                ok = ones if cfg.window is None else np.abs(i - pos) <= reach
                cands.append(with_value(j, float(v))); cps.append(parts.copy()); val.append(ok)
        elif kind == "step":
            grid = np.asarray(t.specs[c].values, dtype=float)
            pos = np.clip(np.searchsorted(grid, truecol), 0, len(grid) - 1)
            for dlt in range(-cfg.relabel_steps, cfg.relabel_steps + 1):
                if dlt == 0:
                    continue
                newpos = np.clip(pos + dlt, 0, len(grid) - 1)
                cands.append(with_value(j, grid[newpos])); cps.append(parts.copy()); val.append(newpos != pos)
        return cands, cps, val

    if cfg.max_f_fields is None:
        for _ in range(max(cfg.rounds, 0)):
            for c, kind in coords:
                imp = try_candidates(*candidates(c, kind))
                moved[c] += int(imp.sum())
        return result()

    # bounded attacker: withholding is the holder's free choice; F fields are spent from a budget
    for _ in range(max(cfg.rounds, 0)):
        for c, kind in wcoords:
            moved[c] += int(try_candidates(*candidates(c, kind)).sum())
    fcoords = [(c, kind) for c, kind in coords if not kind.startswith("w_")]
    used = np.zeros((n, len(fcoords)), dtype=bool)
    for _ in range(min(cfg.max_f_fields, len(fcoords))):
        # best achievable risk per (field, row) from the current presentation, without applying it
        best_by_field = np.full((len(fcoords), n), np.inf)
        choice = []
        for f, (c, kind) in enumerate(fcoords):
            cands, cps, val = candidates(c, kind)
            if not cands:
                choice.append(None)
                continue
            r = ev(np.concatenate(cands), np.concatenate(cps), np.tile(amount, len(cands)),
                   None if cum is None else np.tile(cum, len(cands))).reshape(len(cands), n)
            r = np.where(np.stack(val) & ~used[:, f][None, :], r, np.inf)
            k = r.argmin(0)
            best_by_field[f] = r[k, np.arange(n)]
            choice.append((cands, cps, k))
        f_star = best_by_field.argmin(0)
        rb = best_by_field[f_star, np.arange(n)]
        imp = rb < best - 1e-12
        for f in np.unique(f_star[imp]):
            rows = np.flatnonzero(imp & (f_star == f))
            cands, cps, k = choice[f]
            cur[rows] = np.stack(cands)[k[rows], rows]
            parts[rows] = np.stack(cps)[k[rows], rows]
            used[rows, f] = True
            moved[fcoords[f][0]] += len(rows)
        best = np.where(imp, rb, best)
        if not imp.any():
            break
    return result()


def best_response_restarts(risk_fn, X: pd.DataFrame, t: TypedTable, columns, cfg: AttackConfig = FULL,
                           restarts: int = 5, seed: int = 0, checkpoints=()) -> np.ndarray | dict:
    """Best risk over the plain search and `restarts` searches started from random reachable presentations:
    W claims weakened to a random level at or below the truth, completely reachable F fields set to a random
    grid value. The amount, the cumulative claim and step-relabelled fields start at the truth, so every
    start is reachable from the true input and so is every presentation found from it. With `checkpoints`,
    returns {k: best risk after k restarts} for each k listed (sharing one sequence of random starts)."""
    rng = np.random.default_rng(seed)
    best = best_response(risk_fn, X, t, columns, cfg).risk
    X0 = X.reset_index(drop=True)
    snaps = {0: best.copy()} if 0 in checkpoints else {}
    for k in range(1, restarts + 1):
        S = X0.copy()
        for c in columns:
            s = t.specs[c]
            if cfg.only is not None and c not in cfg.only:
                continue
            if s.ctype == "W" and cfg.withhold:
                truth = X0[c].to_numpy(float)
                if s.reach == "poset":
                    S[c] = np.where(rng.random(len(S)) < 0.5, 0.0, truth)
                else:
                    S[c] = np.floor(rng.random(len(S)) * (truth + 1))
            elif s.ctype == "F" and s.reach == "complete" and cfg.complete and s.values is not None:
                S[c] = rng.choice(np.asarray(s.values, dtype=float), size=len(S))
        best = np.minimum(best, best_response(risk_fn, S, t, columns, cfg).risk)
        if k in checkpoints:
            snaps[k] = best.copy()
    return snaps if checkpoints else best
