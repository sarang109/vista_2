"""Agent-initiated payment overlay (blueprint Section 6), synthetic with controlled ground truth.

A mix of human-initiated and agent-initiated payments reaches receiving banks. Agent payments carry
delegated-authority claim chains (claims.AGENT_SCHEMA): each link of user -> platform -> agent (->
sub-agent) signs a level per chain and the chain's level is the minimum over its links. Human payments
carry authentication claims instead. Agent platforms are in a partial rollout: some support no chains, and
the registry publishes which chains each platform can issue and whether a sender is an agent.

Scenarios: legit human, legit agent, human account takeover, prompt-injected agent. The injected agent
pays an attacker-controlled merchant under a weaker mandate (intent only, little scope slack, no
reconfirmation), at odd hours, for larger amounts, and writes a justification for the payment.

Justification text is generated from word lists (legitimate agents describe routine purchases; injected
agents more often use urgency and refund vocabulary) and scored by a hashed bag-of-words logistic model
trained on the training split. The resulting `text_score` is an F feature: the agent can write anything,
so `text_values` holds the scores of a pool of legitimate-style texts an attacker could write.

Columns
  W  mandate_kind, scope_coverage, agent_identity, tool_policy, user_reconfirmation (agent chains)
     auth_strength, device_continuity (human claims)
  F  hour (complete), amount (split), merchant_id (complete: choice among equivalent merchants and
     retries), text_score (complete: agent-written justification)
  T  is_agent (registry), cap_<chain> and cap_human (registry), cum_mandate_spend (ledger-signed,
     including this payment), cart_mcc (category fixed by a cart mandate, -1 otherwise),
     merchant_age_days (receiver-observed)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..claims import AGENT_SCHEMA
from ..control import FeatureSpec, TypedTable

CHAINS = AGENT_SCHEMA.names
HUMAN = ["auth_strength", "device_continuity"]
N_MERCHANTS = 40

_ROUTINE = ("renew", "monthly", "subscription", "storage", "groceries", "order", "invoice", "office", "supplies",
            "license", "weekly", "delivery", "plan", "standard", "approved", "budget", "per", "mandate",
            "household", "software", "replenish", "recurring", "scheduled", "cart", "confirmed")
_URGENT = ("urgent", "immediately", "refund", "verify", "gift", "card", "wallet", "transfer", "now", "account",
           "security", "unlock", "fee", "release", "pending", "claim", "prize", "crypto", "override", "asap")
_FILLER = ("the", "for", "to", "as", "of", "user", "payment", "this", "item", "requested", "on", "behalf")


def _text(rng, urgent_share: float, k: int) -> str:
    words = []
    for _ in range(k):
        r = rng.random()
        if r < 0.35:
            words.append(_FILLER[rng.integers(len(_FILLER))])
        elif r < 0.35 + 0.65 * urgent_share:
            words.append(_URGENT[rng.integers(len(_URGENT))])
        else:
            words.append(_ROUTINE[rng.integers(len(_ROUTINE))])
    return " ".join(words)


def text_model(texts, y, seed: int = 0):
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = HashingVectorizer(n_features=2 ** 12, alternate_sign=False, norm="l2")
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=seed)
    clf.fit(vec.transform(texts), y)
    return lambda tx: clf.predict_proba(vec.transform(tx))[:, 1]


def generate(n: int = 200_000, agent_share: float = 0.6, n_platforms: int = 20, n_banks: int = 40,
             platform_nonparticipating: float = 0.25, bank_nonparticipating: float = 0.3, p_injected: float = 0.03,
             p_ato: float = 0.015, seed: int = 3) -> TypedTable:
    rng = np.random.default_rng(seed)
    is_agent = rng.random(n) < agent_share
    u = rng.random(n)
    injected = is_agent & (u < p_injected / agent_share)
    ato = ~is_agent & (u < p_ato / (1 - agent_share))
    fraud = injected | ato
    platform = rng.integers(0, n_platforms, n)
    bank = rng.integers(0, n_banks, n)
    plat_on = np.ones(n_platforms, dtype=bool)
    plat_on[rng.permutation(n_platforms)[:int(round(platform_nonparticipating * n_platforms))]] = False
    bank_on = np.ones(n_banks, dtype=bool)
    bank_on[rng.permutation(n_banks)[:int(round(bank_nonparticipating * n_banks))]] = False
    plat_cap = (rng.random((n_platforms, len(CHAINS))) < 0.85) & plat_on[:, None]

    def draw(p_legit, p_bad, bad):
        p_legit, p_bad = np.asarray(p_legit), np.asarray(p_bad)
        out = rng.choice(len(p_legit), size=n, p=p_legit)
        out[bad] = rng.choice(len(p_bad), size=int(bad.sum()), p=p_bad)
        return out + 1

    # per-link levels, chain level = min over links (user, platform, agent, optional sub-agent)
    link_profiles = {
        "mandate_kind": ([0.45, 0.55], [0.85, 0.15]),
        "scope_coverage": ([0.05, 0.35, 0.60], [0.45, 0.40, 0.15]),
        "agent_identity": ([0.55, 0.45], [0.65, 0.35]),
        "tool_policy": ([0.35, 0.65], [0.65, 0.35]),
        "user_reconfirmation": ([0.55, 0.45], [0.88, 0.12]),
    }
    sub_agent = is_agent & (rng.random(n) < 0.3)
    chain = {}
    for j, c in enumerate(CHAINS):
        pl, pb = link_profiles[c]
        lv = draw(pl, pb, injected)
        # other links mostly at the top level; a sub-agent link occasionally lowers the chain
        top = len(pl)
        other = np.where(rng.random(n) < 0.9, top, rng.integers(1, top + 1, n))
        lv = np.minimum(lv, np.where(sub_agent, other, top))
        shown = is_agent & plat_cap[platform, j] & (rng.random(n) >= 0.02)
        chain[c] = np.where(shown, lv, 0)
    human_cap = bank_on[bank] & ~is_agent
    auth = draw([0.30, 0.45, 0.25], [0.50, 0.38, 0.12], ato)
    dev = draw([0.2, 0.8], [0.6, 0.4], ato)
    human = {"auth_strength": np.where(human_cap & (rng.random(n) >= 0.02), auth, 0),
             "device_continuity": np.where(human_cap & (rng.random(n) >= 0.02), dev, 0)}

    category = rng.integers(0, 8, n)
    base = np.array([20, 35, 60, 90, 150, 300, 600, 1200.0])[category]
    amount = np.round(base * np.where(fraud, 2.0, 1.0) * rng.lognormal(0, 0.8, n), 2).clip(1, 20_000)
    merchant_risk = rng.beta(1.0, 6.0, N_MERCHANTS)
    bad_merchants = np.argsort(-merchant_risk)[:12]
    merchant_id = rng.integers(0, N_MERCHANTS, n)
    to_bad = fraud & (rng.random(n) < 0.5)
    merchant_id[to_bad] = rng.choice(bad_merchants, size=int(to_bad.sum()))
    merchant_age = np.where(np.isin(merchant_id, bad_merchants) & fraud & (rng.random(n) < 0.5),
                            rng.integers(1, 60, n), rng.integers(30, 3000, n))
    curve = np.array([1, .6, .4, .3, .3, .5, 1.2, 2.5, 4, 4.5, 4.5, 4.6, 5, 4.8, 4.5, 4.4, 4.6, 5, 5.2, 5, 4.2,
                      3.2, 2.2, 1.5])
    agent_curve = curve * 0.5 + 1.0  # agents run around the clock
    hour = np.where(is_agent, rng.choice(24, n, p=agent_curve / agent_curve.sum()), rng.choice(24, n, p=curve / curve.sum()))
    night = rng.choice(24, n, p=(curve * np.where(np.arange(24) < 6, 4, 1)) / (curve * np.where(np.arange(24) < 6, 4, 1)).sum())
    hour = np.where(fraud & (rng.random(n) < 0.5), night, hour)
    prior = np.where(injected, rng.exponential(80.0, n), rng.lognormal(4.5, 1.2, n) * (rng.random(n) < 0.7))
    cum = np.round(prior + amount, 2)
    cart = chain["mandate_kind"] == 2
    cart_mcc = np.where(cart, category, -1)

    texts = np.array([_text(rng, 0.22 if injected[i] else 0.10, int(rng.integers(6, 14))) if is_agent[i] else ""
                      for i in range(n)], dtype=object)
    split = np.zeros(n, dtype=np.int8)
    split[int(0.6 * n):int(0.8 * n)] = 1
    split[int(0.8 * n):] = 2
    tr = (split == 0) & is_agent
    score = text_model(list(texts[tr]), fraud[tr].astype(int), seed)
    text_score = np.zeros(n)
    text_score[is_agent] = score(list(texts[is_agent]))
    pool = [_text(rng, 0.05, int(rng.integers(6, 14))) for _ in range(200)]
    pool_scores = np.unique(np.round(np.concatenate([score(pool), [0.0]]), 6))

    X = pd.DataFrame({c: chain[c] for c in CHAINS})
    for c in HUMAN:
        X[c] = human[c]
    X["hour"] = hour
    X["amount"] = amount
    X["merchant_id"] = merchant_id
    X["text_score"] = text_score
    X["is_agent"] = is_agent.astype(int)
    for j, c in enumerate(CHAINS):
        X[f"cap_{c}"] = (plat_cap[platform, j] & is_agent).astype(int)
    X["cap_human"] = human_cap.astype(int)
    X["cum_mandate_spend"] = cum
    X["cart_mcc"] = cart_mcc
    X["merchant_age_days"] = merchant_age
    X = X.astype(np.float32)
    y = fraud.astype(np.int8)

    specs = {c: FeatureSpec(c, "W", "chain", capability=f"cap_{c}", note="delegated-authority chain, min over links")
             for c in CHAINS}
    for c in HUMAN:
        specs[c] = FeatureSpec(c, "W", "chain", capability="cap_human", note="human authentication claim")
    specs["hour"] = FeatureSpec("hour", "F", "complete", np.arange(24, dtype=float))
    specs["amount"] = FeatureSpec("amount", "F", "split", np.unique(np.quantile(amount, np.linspace(0, 1, 24)).round(2)),
                                  counterpart="cum_mandate_spend")
    specs["merchant_id"] = FeatureSpec("merchant_id", "F", "complete", np.arange(N_MERCHANTS, dtype=float),
                                       note="choice among equivalent merchants; retries")
    specs["text_score"] = FeatureSpec("text_score", "F", "complete", pool_scores, note="agent-written justification")
    for c in ["is_agent", "cap_human", "cum_mandate_spend", "cart_mcc", "merchant_age_days"] + [f"cap_{c}" for c in CHAINS]:
        specs[c] = FeatureSpec(c, "T")
    meta = {"is_agent": is_agent, "injected": injected, "ato": ato, "texts": texts, "seed": seed,
            "text_pool_size": len(pool)}
    return TypedTable("agentic", X, y, split, specs, amount="amount", cumulative="cum_mandate_spend",
                      baseline_exclude=["cum_mandate_spend", "cart_mcc"],
                      capability_cols=["is_agent", "cap_human"] + [f"cap_{c}" for c in CHAINS], meta=meta)
