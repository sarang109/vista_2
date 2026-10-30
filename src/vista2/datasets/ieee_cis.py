"""IEEE-CIS Fraud Detection (Vesta / IEEE Computational Intelligence Society, Kaggle competition 2019).

590,540 card-not-present transactions with an isFraud label; train_transaction.csv and train_identity.csv
joined by TransactionID. The data must be downloaded from the competition page after accepting its rules
and placed in data/ieee_cis/. Column meanings are anonymised, so the W/F/T typing below is a modelling
assumption; it is published with the results and stress-tested by E8.

  W  M1-M9 match flags (poset: nothing shown below each recorded value)              uncertain
     id_12, id_15, id_16, id_28, id_29 ("found" device/account continuity, chain)     uncertain
     id_35-id_38 binary flags (poset)                                                 uncertain
  F  TransactionAmt (split), hour and weekday from TransactionDT (complete), P/R email domain, addr1,
     addr2, DeviceType, DeviceInfo, id_30 (OS), id_31 (browser), id_33 (resolution), dist1/dist2*
  T  ProductCD*, card1-card6, C1-C14, D1-D15*, a correlation-pruned subset of V1-V339*,
     remaining id_ columns*, and cum_amt_24h: issuer-observable spend on the same card in the previous
     24 hours including this transaction (the cumulative claim of Algorithm 3; not used by baselines)
  (* = uncertain)
Missing values are encoded as -999 except for W claims, where 0 means "nothing shown".
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..control import FeatureSpec, TypedTable, grid_values

M_FLAGS = [f"M{i}" for i in range(1, 10)]
ID_CHAIN = {"id_12": {"NotFound": 1, "Found": 2}, "id_15": {"New": 1, "Unknown": 1, "Found": 2},
            "id_16": {"NotFound": 1, "Found": 2}, "id_28": {"New": 1, "Found": 2},
            "id_29": {"NotFound": 1, "Found": 2}}
ID_POSET = ["id_35", "id_36", "id_37", "id_38"]
F_CODES = ["P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33"]
F_NUM = ["addr1", "addr2"]
F_UNCERTAIN = ["dist1", "dist2"]
CARD = ["card1", "card2", "card3", "card4", "card5", "card6"]


def _codes(s: pd.Series, top: int = 60) -> pd.Series:
    vc = s.value_counts()
    keep = set(vc.index[:top])
    s = s.where(s.isin(keep), other="__other__").where(s.notna(), other=None)
    return s.astype("category").cat.codes.astype(np.float32).replace(-1, -999)


def _prune_v(df: pd.DataFrame, rng, thresh: float = 0.9) -> list[str]:
    """Keep a de-correlated subset of the V columns: group by missing count, then greedy |corr| < thresh."""
    vcols = [c for c in df.columns if c.startswith("V")]
    sample = df[vcols].sample(min(len(df), 100_000), random_state=int(rng.integers(1 << 30)))
    nan_count = sample.isna().sum()
    keep = []
    for _, cols in nan_count.groupby(nan_count).groups.items():
        cols = list(cols)
        sub = sample[cols].fillna(-999)
        corr = sub.corr().abs().fillna(1.0).to_numpy()
        chosen = []
        for j in range(len(cols)):
            if sub.iloc[:, j].nunique() <= 1:
                continue
            if all(corr[j, k] < thresh for k in chosen):
                chosen.append(j)
        keep += [cols[j] for j in chosen]
    return sorted(keep, key=lambda c: int(c[1:]))


def _cumulative(dt: np.ndarray, amt: np.ndarray, key: np.ndarray, window: int = 86_400) -> np.ndarray:
    """Spend on the same key within the previous `window` seconds, including the current transaction."""
    order = np.lexsort((dt, key))
    k, t, a = key[order], dt[order], amt[order]
    cs = np.cumsum(a)
    out = np.empty(len(a))
    start = 0
    for i in range(len(a)):
        if i > 0 and k[i] != k[i - 1]:
            start = i
        j = np.searchsorted(t[start:i + 1], t[i] - window, side="left") + start
        out[i] = cs[i] - (cs[j - 1] if j > 0 else 0.0)
    res = np.empty(len(a))
    res[order] = out
    return res


def load(data_dir: str, seed: int = 0, cache: bool = True) -> TypedTable:
    cache_path = os.path.join(data_dir, "typed.parquet")
    if cache and os.path.exists(cache_path):
        X = pd.read_parquet(cache_path)
    else:
        tr = pd.read_csv(os.path.join(data_dir, "train_transaction.csv"))
        idt = pd.read_csv(os.path.join(data_dir, "train_identity.csv"))
        df = tr.merge(idt, on="TransactionID", how="left").sort_values("TransactionDT").reset_index(drop=True)
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(index=df.index)
        X["isFraud"] = df["isFraud"].astype(np.int8)
        for c in M_FLAGS:
            if c == "M4":
                X[c] = df[c].map({"M0": 1, "M1": 2, "M2": 3}).fillna(0).astype(np.float32)
            else:
                X[c] = df[c].map({"F": 1, "T": 2}).fillna(0).astype(np.float32)
        for c, mp in ID_CHAIN.items():
            X[c] = df[c].map(mp).fillna(0).astype(np.float32)
        for c in ID_POSET:
            X[c] = df[c].map({"F": 1, "T": 2}).fillna(0).astype(np.float32)
        X["TransactionAmt"] = df["TransactionAmt"].astype(np.float32)
        X["hour"] = ((df["TransactionDT"] // 3600) % 24).astype(np.float32)
        X["weekday"] = ((df["TransactionDT"] // 86_400) % 7).astype(np.float32)
        for c in F_CODES:
            X[c] = _codes(df[c])
        for c in F_NUM + F_UNCERTAIN:
            X[c] = df[c].fillna(-999).astype(np.float32)
        X["ProductCD"] = _codes(df["ProductCD"])
        for c in CARD:
            X[c] = _codes(df[c], 500) if not pd.api.types.is_numeric_dtype(df[c]) else df[c].fillna(-999).astype(np.float32)
        for c in [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)]:
            X[c] = df[c].fillna(-999).astype(np.float32)
        for c in _prune_v(df, rng):
            X[c] = df[c].fillna(-999).astype(np.float32)
        other_id = [c for c in idt.columns if c.startswith("id_") and c not in ID_CHAIN and c not in ID_POSET
                    and c not in F_CODES]
        for c in other_id:
            X[c] = _codes(df[c]) if not pd.api.types.is_numeric_dtype(df[c]) else df[c].fillna(-999).astype(np.float32)
        key = df[CARD].astype(object).fillna("na").astype(str).agg("|".join, axis=1).astype("category").cat.codes.to_numpy()
        X = X.copy()  # columns were added one at a time; consolidate before the last two
        X["cum_amt_24h"] = _cumulative(df["TransactionDT"].to_numpy(np.int64), df["TransactionAmt"].to_numpy(float),
                                       key).astype(np.float32)
        X["card_key"] = key.astype(np.int64)
        if cache:
            X.to_parquet(cache_path)
    y = X.pop("isFraud").to_numpy(np.int8)
    X.pop("card_key")
    n = len(X)
    split = np.zeros(n, dtype=np.int8)
    split[int(0.6 * n):int(0.8 * n)] = 1
    split[int(0.8 * n):] = 2
    specs = {}
    for c in M_FLAGS + ID_POSET:
        specs[c] = FeatureSpec(c, "W", "poset", uncertain=True, note="match / verification flag")
    for c in ID_CHAIN:
        specs[c] = FeatureSpec(c, "W", "chain", uncertain=True, note="continuity: not found < found")
    amt_grid = np.unique(np.quantile(X["TransactionAmt"], np.linspace(0, 1, 24)).round(2))
    specs["TransactionAmt"] = FeatureSpec("TransactionAmt", "F", "split", amt_grid, counterpart="cum_amt_24h")
    specs["hour"] = FeatureSpec("hour", "F", "complete", np.arange(24, dtype=float))
    specs["weekday"] = FeatureSpec("weekday", "F", "complete", np.arange(7, dtype=float))
    for c in F_CODES + F_NUM:
        specs[c] = FeatureSpec(c, "F", "complete", grid_values(X[c]), note="holder-entered or spoofable")
    for c in F_UNCERTAIN:
        specs[c] = FeatureSpec(c, "F", "complete", grid_values(X[c]), uncertain=True)
    for c in X.columns:
        if c not in specs:
            unc = c == "ProductCD" or c.startswith(("D", "V", "id_"))
            specs[c] = FeatureSpec(c, "T", uncertain=unc, note="issuer, processor or receiver observed")
    specs["cum_amt_24h"] = FeatureSpec("cum_amt_24h", "T", note="issuer-signed cumulative spend (Algorithm 3)")
    return TypedTable("ieee_cis", X, y, split, specs, amount="TransactionAmt", cumulative="cum_amt_24h",
                      baseline_exclude=["cum_amt_24h"], meta={"rows": n, "fraud_rate": float(y.mean())})
