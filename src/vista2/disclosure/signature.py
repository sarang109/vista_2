"""Selection leakage on the signature space (Theorem 4, no free selection).

A receiver policy has frontier branches M = {v_1, ..., v_m}. The signature of an evidence vector x is
A(x) = {v in M : v <= x}, the branches x supports. A selection kernel q(v | A(x)) picks which branch the
holder discloses. With up(v) = pi_S(X >= v) and L_S(v) = -log2 up(v):

  Lambda(q) = I(X; V) - E[L_S(V)]      selection leakage: what the choice reveals beyond the claims shown

Theorem 4: Lambda = 0 is attainable iff sum_{v in a} w_v = 1 has a non-negative solution over the
signatures with positive mass; two incomparable branches whose singleton signatures and joint signature all
have mass make every branch-revealing kernel leak. Carried over from v1 (docs/theory.md has the proof).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog


class SignatureSpace:
    def __init__(self, sig: np.ndarray, mass: np.ndarray, C: np.ndarray, extra: dict | None = None):
        sig = np.asarray(sig, dtype=bool)
        mass = np.asarray(mass, dtype=float)
        keep = (mass > 0) & sig.any(1)
        self.sig, self.mass = sig[keep], mass[keep] / mass[keep].sum()
        self.C = np.asarray(C, dtype=float)
        self.m = self.sig.shape[1]
        self.up = self.mass @ self.sig  # pi_S(X >= v)
        self.live = self.up > 0
        with np.errstate(divide="ignore"):
            self.L = np.where(self.live, -np.log2(np.where(self.live, self.up, 1.0)), np.inf)
        # optional per-signature masses of a binary target Y (for Bayes attackers): extra[name] = P(A=a, Y=1)
        self.extra = {k: np.asarray(v, dtype=float)[keep] / mass[keep].sum() for k, v in (extra or {}).items()}
        self.codes = self.sig.astype(np.int64) @ (1 << np.arange(self.m, dtype=np.int64))

    # ------------------------------------------------------------------ construction (Lemma 2)
    @classmethod
    def from_states(cls, avail: np.ndarray, p: np.ndarray, C: np.ndarray, targets: dict | None = None):
        """Aggregate a state-level space (avail: N x m bool, p: prior over N states) by signature."""
        acc = avail.any(1) & (p > 0)
        codes = avail[acc].astype(np.int64) @ (1 << np.arange(avail.shape[1], dtype=np.int64))
        uniq, inv = np.unique(codes, return_inverse=True)
        mass = np.bincount(inv, weights=p[acc], minlength=len(uniq))
        sig = ((uniq[:, None] >> np.arange(avail.shape[1])) & 1).astype(bool)
        extra = {k: np.bincount(inv, weights=(p * t)[acc], minlength=len(uniq)) for k, t in (targets or {}).items()}
        return cls(sig, mass, C, extra)

    def index_of(self, sig_row) -> int:
        code = int(np.asarray(sig_row, dtype=np.int64) @ (1 << np.arange(self.m, dtype=np.int64)))
        hit = np.flatnonzero(self.codes == code)
        return int(hit[0]) if len(hit) else -1

    @property
    def n_signatures(self) -> int:
        return len(self.mass)

    # ------------------------------------------------------------------ kernels
    def _normalise(self, logits: np.ndarray) -> np.ndarray:
        logits = np.where(self.sig, logits, -np.inf)
        mx = logits.max(1, keepdims=True)
        z = np.exp(logits - mx)
        return z / z.sum(1, keepdims=True)

    def kernel_det(self) -> np.ndarray:
        Q = np.zeros(self.sig.shape)
        Q[np.arange(len(Q)), np.argmin(np.where(self.sig, self.C[None, :], np.inf), axis=1)] = 1.0
        return Q

    def kernel_exp(self, lam: float) -> np.ndarray:
        return self._normalise(np.broadcast_to(-lam * (self.C - self.C.min()), self.sig.shape))

    def kernel_uniform(self) -> np.ndarray:
        return self._normalise(np.zeros(self.sig.shape))

    # ------------------------------------------------------------------ metrics (Lemma 1)
    def metrics(self, Q: np.ndarray) -> dict:
        P = self.mass @ Q
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(Q > 0, Q / np.where(P > 0, P, 1.0)[None, :], 1.0)
            I = float((self.mass[:, None] * Q * np.log2(ratio)).sum())
        used = P > 0
        EL = float((P[used] * self.L[used]).sum())
        return {"I_bits": I, "E_L_bits": EL, "Lambda": I - EL, "cost": float(P @ self.C), "P": P}

    def lam(self, Q) -> float:
        return self.metrics(Q)["Lambda"]

    def cost(self, Q) -> float:
        return float(self.mass @ Q @ self.C)

    # ------------------------------------------------------------------ Theorem 4 (no free selection)
    def zero_leakage_system(self) -> dict:
        """Feasibility of sum_{v in a} w_v = 1 (all signatures a), w >= 0, solved as an L1 residual LP:
        min sum(s+ + s-) s.t. S w + s+ - s- = 1. Residual 0 <=> some kernel has Lambda = 0."""
        k, m = self.sig.shape
        A = np.hstack([self.sig.astype(float), np.eye(k), -np.eye(k)])
        c = np.concatenate([np.zeros(m), np.ones(2 * k)])
        res = linprog(c, A_eq=A, b_eq=np.ones(k), bounds=(0, None), method="highs")
        resid = float(res.fun) if res.status == 0 else math.inf
        w = res.x[:m] if res.status == 0 else None
        return {"feasible": resid < 1e-9, "residual": resid, "w": w}

    def zero_leakage_kernel(self, w: np.ndarray) -> np.ndarray:
        """The Lambda = 0 kernel q(v|a) = w_v from a solution of the Theorem 1 system."""
        Q = np.where(self.sig, np.maximum(w, 0.0)[None, :], 0.0)
        return Q / Q.sum(1, keepdims=True)

    def two_branch_obstruction(self) -> tuple | None:
        """Sufficient condition of Theorem 1: branches a != b with singleton signatures {a}, {b} of positive
        mass and some positive-mass signature containing both. Returns (a, b) or None."""
        single = [int(np.flatnonzero(s)[0]) for s in self.sig if s.sum() == 1]
        for i, a in enumerate(single):
            for b in single[i + 1:]:
                if (self.sig[:, a] & self.sig[:, b]).any():
                    return a, b
        return None
