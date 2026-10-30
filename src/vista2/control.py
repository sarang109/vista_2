"""Control typing of model inputs and Algorithm 1 (ControlTypeAudit).

Every feature the receiver's model reads is typed by who controls it:

  W  withholdable attested   signed claims the holder may omit or weaken, never strengthen
  F  freely chosen           fields the holder picks or shapes, within a reachability relation
  T  third-party fixed       facts signed or observed by someone other than the holder

Corollary 1 turns the types into a design rule: the decision must be non-decreasing in every W
coordinate, may not reward any reachable change of an F coordinate (and may not depend at all on a
completely reachable one), and may use T coordinates freely.

Reachability kinds used for F features:
  complete  every value reachable from every other (hour of payment, free text, email domain)
  split     a payment can be presented as any smaller amount (split-closed)
  step      ordered categories; one manipulation moves one step (closure: the whole component)
W features are integer coded with 0 = nothing shown; "chain" means levels 0 < 1 < ... and "poset" means a
bottom 0 below mutually incomparable values (e.g. a match flag whose values "match" and "mismatch" can
both be hidden but neither turned into the other).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeatureSpec:
    name: str
    ctype: str  # "W", "F" or "T"
    reach: str = "fixed"  # W: chain | poset ; F: complete | split | step ; T: fixed
    values: np.ndarray | None = None  # candidate grid for F manipulation and repair
    counterpart: str | None = None  # T-type replacement for a split-closed F feature
    uncertain: bool = False  # typing is a modelling assumption (sensitivity analysis moves it)
    capability: str | None = None  # T-type registry flag saying whether this W claim can be issued
    note: str = ""

    def __post_init__(self):
        if self.ctype not in ("W", "F", "T"):
            raise ValueError(f"bad control type {self.ctype!r} for {self.name}")


@dataclass
class TypedTable:
    """A dataset with control-typed columns, a time-ordered train/validation/test split and labels."""
    name: str
    X: pd.DataFrame
    y: np.ndarray
    split: np.ndarray  # 0 train, 1 validation, 2 test
    specs: dict  # column -> FeatureSpec
    amount: str | None = None  # per-payment amount column (F, split)
    cumulative: str | None = None  # issuer-signed cumulative spend including this payment (T)
    baseline_exclude: list = field(default_factory=list)  # T columns conventional pipelines do not have
    capability_cols: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def cols(self, ctype: str) -> list[str]:
        return [c for c, s in self.specs.items() if s.ctype == ctype]

    def part(self, k: int):
        m = self.split == k
        return self.X[m], self.y[m]

    def subset(self, mask) -> "TypedTable":
        """Rows where mask is true (meta arrays of full length are subset too)."""
        mask = np.asarray(mask, dtype=bool)
        meta = {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v) for k, v in self.meta.items()}
        return TypedTable(self.name, self.X[mask].reset_index(drop=True), self.y[mask], self.split[mask], self.specs,
                          self.amount, self.cumulative, list(self.baseline_exclude), list(self.capability_cols), meta)

    def retyped(self, moves: dict) -> "TypedTable":
        """Copy with some features moved to another type, for the typing-sensitivity analysis.
        moves: column -> (ctype, reach)."""
        specs = dict(self.specs)
        for c, (ct, reach) in moves.items():
            s = specs[c]
            vals = s.values
            if ct == "F" and vals is None:
                vals = grid_values(self.X[c])
            specs[c] = FeatureSpec(c, ct, reach, vals, s.counterpart, s.uncertain, s.capability, s.note)
        return TypedTable(self.name, self.X, self.y, self.split, specs, self.amount, self.cumulative,
                          list(self.baseline_exclude), list(self.capability_cols), dict(self.meta))


def grid_values(col: pd.Series, max_values: int = 32) -> np.ndarray:
    """Candidate values for manipulating a freely chosen feature: all distinct values if few, else quantiles."""
    v = np.asarray(col.dropna().unique(), dtype=float)
    if len(v) <= max_values:
        return np.sort(v)
    return np.unique(np.quantile(col.dropna().to_numpy(dtype=float), np.linspace(0, 1, max_values)))


# ----------------------------------------------------------------------------- Algorithm 1
@dataclass
class ProtocolEntry:
    """How a feature reaches the receiver, as written in the protocol specification."""
    name: str
    signed_by: str | None  # "issuer", "registry", "acquirer", "ledger", "receiver" or None (unsigned)
    holder_routed: bool  # True if the holder decides whether and how it travels with the payment
    reach: str = "complete"  # for unsigned fields: complete | split | step
    counterpart: str | None = None


@dataclass
class AuditResult:
    specs: dict
    drop: list
    monotone: dict  # column -> direction of the RISK score (-1: risk may not rise with more evidence)
    missing_counterparts: list


def control_type_audit(entries: list[ProtocolEntry], available: set | None = None) -> AuditResult:
    """Algorithm 1. Type each feature, drop what a strategy-proof rule cannot use, and list the monotone
    constraints for training. `available` lists the columns that exist (to check counterparts)."""
    specs, drop, mono, missing = {}, [], {}, []
    for e in entries:
        if e.signed_by is not None and e.holder_routed:
            specs[e.name] = FeatureSpec(e.name, "W", "chain")
            mono[e.name] = -1
        elif e.signed_by is not None:
            specs[e.name] = FeatureSpec(e.name, "T")
        else:
            specs[e.name] = FeatureSpec(e.name, "F", e.reach, counterpart=e.counterpart)
            if e.reach == "split":
                if e.counterpart and (available is None or e.counterpart in available):
                    mono[e.counterpart] = +1
                else:
                    missing.append(e.name)
            drop.append(e.name)
    return AuditResult(specs, drop, mono, missing)


# ----------------------------------------------------------------------------- model variants
@dataclass
class Variant:
    """A receiver model configuration: which columns it reads and which monotone constraints it obeys."""
    name: str
    columns: list
    monotone: dict  # column -> -1/+1 on the risk score
    repair: str = ""  # "" (none), "w" (Algorithm 2b over the claim lattice) or "full" (W and F)


DEFAULT_VARIANTS = ("naive", "unconstrained", "monotone_w", "repaired_w", "repaired_full", "control_typed")


def variants(t: TypedTable, which=DEFAULT_VARIANTS) -> list[Variant]:
    W, F, T = t.cols("W"), t.cols("F"), t.cols("T")
    base_T = [c for c in T if c not in t.baseline_exclude and c not in t.capability_cols]
    conventional = W + F + base_T
    out = []
    for v in which:
        if v == "naive":
            out.append(Variant("naive", conventional, {}))
        elif v == "unconstrained":
            out.append(Variant("unconstrained", conventional + t.capability_cols, {}))
        elif v == "monotone_w":
            out.append(Variant("monotone_w", conventional + t.capability_cols, {c: -1 for c in W}))
        elif v in ("repaired_w", "repaired_full"):
            out.append(Variant(v, conventional + t.capability_cols, {}, repair=v.split("_")[1]))
        elif v == "control_typed":
            cols = W + [c for c in T]
            mono = {c: -1 for c in W}
            if t.cumulative:
                mono[t.cumulative] = +1
            out.append(Variant("control_typed", cols, mono))
        else:
            raise ValueError(v)
    return out


def attack_surface(t: TypedTable, columns) -> dict:
    """The coordinates a holder can move, restricted to the columns a model reads."""
    return {c: t.specs[c] for c in columns if t.specs[c].ctype in ("W", "F")}
