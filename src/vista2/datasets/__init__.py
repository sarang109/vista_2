"""Dataset registry. Real datasets are read from data/ and are never committed (see data/README.md)."""
from __future__ import annotations

import os
from functools import lru_cache

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "data")


def available(name: str) -> bool:
    if name in ("synthetic", "agentic"):
        return True
    if name == "baf":
        return os.path.exists(os.path.join(DATA, "baf", "Base.csv"))
    if name == "ieee_cis":
        d = os.path.join(DATA, "ieee_cis")
        return os.path.exists(os.path.join(d, "typed.parquet")) or (
            os.path.exists(os.path.join(d, "train_transaction.csv")) and os.path.exists(os.path.join(d, "train_identity.csv")))
    if name == "paysim":
        return os.path.exists(os.path.join(DATA, "paysim_cache", "amount_minor.npy"))
    return False


@lru_cache(maxsize=8)
def load(name: str, size: int | None = None, seed: int = 1, nonparticipating: float = 0.3):
    if name == "synthetic":
        from . import synthetic
        return synthetic.generate(n=size or 240_000, seed=seed, nonparticipating=nonparticipating)
    if name == "agentic":
        from . import agentic
        return agentic.generate(n=size or 200_000, seed=seed)
    if name == "baf":
        from . import baf
        return baf.load(os.path.join(DATA, "baf", "Base.csv"))
    if name == "ieee_cis":
        from . import ieee_cis
        return ieee_cis.load(os.path.join(DATA, "ieee_cis"))
    raise KeyError(name)
