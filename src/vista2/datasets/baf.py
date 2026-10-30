"""Bank Account Fraud (BAF) suite, Base variant (Jesus et al., NeurIPS 2022 Datasets and Benchmarks; Feedzai).

1,000,000 bank-account applications over eight months with a fraud label. The suite is distributed through
Kaggle (sgpjesus/bank-account-fraud-dataset-neurips-2022) under CC BY-NC-SA 4.0; this module only reads a
local copy of Base.csv placed in data/baf/.

BAF is account opening, not payments, so there is no amount to split. It is used because it separates
verified from self-reported fields, which is the W/F distinction. The typing below is a modelling
assumption; uncertain entries are moved between types by the sensitivity experiment (E8).

  W  phone_home_valid, phone_mobile_valid        validated phone numbers; the applicant may give none
     bank_months_count                           tenure at a previous bank, as threshold levels
  F  income, name_email_similarity, customer_age*, current_address_months_count*,
     prev_address_months_count*, days_since_request*, intended_balcon_amount, payment_type,
     employment_status, housing_status, email_is_free, proposed_credit_limit, source,
     session_length_in_minutes, device_os, keep_alive_session        (all "complete")
  T  credit_risk_score, velocity_6h/24h/4w, zip_count_4w, bank_branch_count_8w,
     date_of_birth_distinct_emails_4w, device_distinct_emails_8w*, device_fraud_count*,
     foreign_request*, has_other_cards*
  (* = uncertain)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..control import FeatureSpec, TypedTable, grid_values

F_COLS = ["income", "name_email_similarity", "customer_age", "current_address_months_count",
          "prev_address_months_count", "days_since_request", "intended_balcon_amount", "payment_type",
          "employment_status", "housing_status", "email_is_free", "proposed_credit_limit", "source",
          "session_length_in_minutes", "device_os", "keep_alive_session"]
T_COLS = ["credit_risk_score", "velocity_6h", "velocity_24h", "velocity_4w", "zip_count_4w",
          "bank_branch_count_8w", "date_of_birth_distinct_emails_4w", "device_distinct_emails_8w",
          "device_fraud_count", "foreign_request", "has_other_cards"]
W_COLS = ["phone_home_valid", "phone_mobile_valid", "bank_months_count"]
UNCERTAIN = {"customer_age", "current_address_months_count", "prev_address_months_count", "days_since_request",
             "device_distinct_emails_8w", "device_fraud_count", "foreign_request", "has_other_cards"}
TENURE_CUTS = (1, 6, 12, 24)


def load(path: str, seed: int = 0) -> TypedTable:
    df = pd.read_csv(path)
    y = df["fraud_bool"].to_numpy(np.int8)
    X = pd.DataFrame(index=df.index)
    for c in W_COLS[:2]:
        X[c] = df[c].astype(np.float32)
    bm = df["bank_months_count"].to_numpy()
    X["bank_months_count"] = np.where(bm < 0, 0, 1 + np.searchsorted(TENURE_CUTS, bm, side="right")).astype(np.float32)
    for c in F_COLS + T_COLS:
        col = df[c]
        if not pd.api.types.is_numeric_dtype(col):
            col = col.astype("category").cat.codes
        X[c] = col.astype(np.float32)
    month = df["month"].to_numpy()
    split = np.where(month <= 4, 0, np.where(month == 5, 1, 2)).astype(np.int8)
    specs = {}
    for c in W_COLS:
        specs[c] = FeatureSpec(c, "W", "chain", note="verified; applicant may omit")
    for c in F_COLS:
        specs[c] = FeatureSpec(c, "F", "complete", grid_values(X[c]), uncertain=c in UNCERTAIN,
                               note="self-reported or chosen by the applicant")
    for c in T_COLS:
        specs[c] = FeatureSpec(c, "T", uncertain=c in UNCERTAIN, note="bureau or bank observed")
    return TypedTable("baf", X, y, split, specs, meta={"rows": len(X), "fraud_rate": float(y.mean()),
                                                         "source": os.path.basename(path)})
