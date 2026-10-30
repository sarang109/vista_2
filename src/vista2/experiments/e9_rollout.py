"""E9: how does the withholding attack depend on the rollout stage?

Synthetic workload with the share of sending banks that issue no claims swept from 0% to 70%. During a
partial rollout "no claims" is an ordinary, mostly legitimate input, which is what makes withholding pay.
The registry capability flag (T-type) and monotone training remove the gain in different ways; this
experiment measures each against the share of non-participating banks.
"""
from __future__ import annotations

import os

from ..attack import FULL, WITHHOLD_ONLY
from ..control import variants
from ..datasets import synthetic
from ..util import write_csv
from .common import evaluate_trained, fit_variant
from .summarise import by_group


def run(cfg, out, log=print):
    rows = []
    for share in cfg.get("e9_shares", [0.0, 0.1, 0.3, 0.5, 0.7]):
        for seed in cfg.get("seeds_e9", cfg.get("seeds", [1])):
            t = synthetic.generate(n=cfg.get("synthetic_size") or 240_000, seed=seed, nonparticipating=share)
            for v in variants(t, ("naive", "unconstrained", "monotone_w", "control_typed")):
                tr = fit_variant(t, v, "lightgbm", seed=seed)
                for r in evaluate_trained(t, tr, [WITHHOLD_ONLY, FULL], seed=seed):
                    r.update(nonparticipating=share, variant=v.name, seed=seed)
                    rows.append(r)
        log(f"  e9 share {share}: done")
    return [write_csv(os.path.join(out, "e9_rollout_runs.csv"), rows),
            write_csv(os.path.join(out, "e9_rollout.csv"), by_group(rows, ("nonparticipating", "variant", "attack")))]
