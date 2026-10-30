"""E4: do the agentic attacks behave as Corollary 1 and Theorem 3 predict? (Gate B)

Agent workload (datasets/agentic.py). Four adversaries from blueprint Section 6 plus their combination:
  1 withhold_split   prompt-injected agent withholds weak claims and splits under a per-payment cap
  2 persuasive_text  agent writes the justification that the receiver's model likes best
  3 merchant_retry   agent retries across equivalent merchants and times until one is accepted
  4 chain_strip      colluding sub-agent strips links from the authority chain (chains drop to nothing)
  full               all of the above
Receiver models: naive, unconstrained (registry flags), monotone in W, control-typed, and control-typed
with a text hold (text may only lower acceptance). Results are reported on injected-agent frauds; human
account takeovers are excluded from the attacked set because adversaries 1-4 concern agents.

Gate B passes when, for the control-typed model, no adversary raises fraud acceptance above its truthful
level (Corollary 1) and exposure under splitting is bounded by the cumulative claim (Theorem 3), while
the conventional models do gain.

The text-hold variant is not strategy-proof (it reads a completely reachable F field). Its hold budget is
partly spent on text the agent can avoid, so at a fixed legitimate-hold rate it looks better than plain
control-typed training when evaluated and is worse once the agent writes benign text.
"""
from __future__ import annotations

import os

import numpy as np

from .. import datasets
from ..attack import FULL, AttackConfig
from ..control import Variant, variants
from ..datasets.agentic import CHAINS
from ..evaluate import hold_threshold
from ..util import write_csv
from .common import Trained, evaluate_trained, fit_variant
from .summarise import by_group


def adversaries(t) -> list[AttackConfig]:
    W = set(t.cols("W"))
    return [AttackConfig(split_k=4, complete=False, relabel_steps=0, only=W | {t.amount}, name="withhold_split"),
            AttackConfig(withhold=False, split_k=1, only={"text_score"}, name="persuasive_text"),
            AttackConfig(withhold=False, split_k=1, only={"merchant_id", "hour"}, name="merchant_retry"),
            AttackConfig(split_k=1, complete=False, only=set(CHAINS), name="chain_strip"),
            FULL]


class TextHold(Trained):
    """Control-typed decision plus a hold triggered by suspicious agent text: risk = ct risk, raised above
    every threshold when text_score exceeds text_tau. The text can only lower acceptance."""

    def __init__(self, base: Trained, text_tau: float):
        super().__init__(Variant("control_typed_text_hold", base.columns + ["text_score"], base.variant.monotone),
                         base.model, None, base.fit_s)
        self.base, self.text_tau = base, text_tau

    def risk(self, X):
        return self.base.risk(X) + 2.0 * (X["text_score"].to_numpy(float) > self.text_tau)


def run(cfg, out, log=print):
    t_all = datasets.load("agentic", cfg.get("agentic_size"))
    # attacked set: injected agents only (human ATO rows removed from validation and test frauds)
    keep = ~(np.asarray(t_all.meta["ato"]) & (t_all.split > 0))
    t = t_all.subset(keep)
    rows = []
    advs = adversaries(t)
    for seed in cfg.get("seeds_agentic", cfg.get("seeds", [1])):
        trained = []
        for v in variants(t, ("naive", "unconstrained", "monotone_w", "control_typed")):
            trained.append(fit_variant(t_all, v, "lightgbm", seed=seed))
        ct = trained[-1]
        va = (t.split == 1) & (t.y == 0)
        text_tau = hold_threshold(t.X.loc[va, "text_score"].to_numpy(float), 0.01)
        trained.append(TextHold(ct, text_tau))
        for tr in trained:
            res = evaluate_trained(t, tr, advs, seed=seed, max_fraud_attack=cfg.get("max_fraud_attack"))
            for r in res:
                r.update(variant=tr.variant.name, seed=seed)
                rows.append(r)
            log(f"  e4 seed {seed} {tr.variant.name}: " + ", ".join(
                f"{r['attack']} {r.get('fraud_accepted_attack', r['fraud_accepted_truthful']):.3f}" for r in res))
    summ = by_group(rows, ("variant", "attack"))
    gate = []
    for v in sorted({r["variant"] for r in summ}):
        sub = {r["attack"]: r for r in summ if r["variant"] == v}
        base = sub["truthful"]["fraud_accepted_truthful"]
        for a in [x.name for x in advs]:
            gain = sub[a]["fraud_accepted_attack"] - base
            # Corollary 1: a control-typed rule cannot gain from any adversary; a W-monotone rule cannot
            # gain from pure withholding (chain stripping) but can from F moves
            # (the text hold reads a completely reachable F field, so Corollary 1(b) says it is not SP)
            predicted = v == "control_typed" or (v == "monotone_w" and a == "chain_strip")
            gate.append({"variant": v, "attack": a, "truthful": base, "attacked": sub[a]["fraud_accepted_attack"],
                         "gain": gain, "predicted_no_gain": predicted,
                         "consistent": (gain <= 1e-9) if predicted else True,
                         "exposure_attack": sub[a].get("exposure_attack", ""),
                         "exposure_truthful": sub[a].get("exposure_truthful", "")})
    naive_gains = sum(1 for g in gate if g["variant"] == "naive" and g["attack"] != "full" and g["gain"] > 0)
    passed = all(g["consistent"] for g in gate) and naive_gains >= 3
    gate.append({"variant": "GATE_B", "attack": "all", "consistent": passed,
                 "gain": naive_gains, "predicted_no_gain": "naive model gains on this many of 4 adversaries"})
    return [write_csv(os.path.join(out, "e4_agentic_runs.csv"), rows),
            write_csv(os.path.join(out, "e4_agentic.csv"), summ),
            write_csv(os.path.join(out, "e4_gate_b.csv"), gate)]
