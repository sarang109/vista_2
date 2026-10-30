"""Ordered claim domains (W-type evidence).

Attribute i has levels 0 < 1 < ... < d_i, where 0 means "nothing shown". An issued claim (i, t) asserts
x_i >= t. A holder whose true level is x_i receives one signed threshold claim for every t <= x_i, so it can
present any weaker level and never a stronger one: exactly the order w' <= w of the evidence game.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Attribute:
    name: str
    levels: tuple  # labels; index 0 = nothing shown
    ttl_s: int | None = None

    @property
    def d(self) -> int:
        return len(self.levels) - 1

    @property
    def size(self) -> int:
        return len(self.levels)


class Schema:
    def __init__(self, version: str, attributes):
        self.version = version
        self.attributes = tuple(attributes)
        self._idx = {a.name: i for i, a in enumerate(self.attributes)}
        if len(self._idx) != len(self.attributes):
            raise ValueError("duplicate attribute names")

    @property
    def n(self) -> int:
        return len(self.attributes)

    @property
    def dims(self) -> tuple:
        return tuple(a.size for a in self.attributes)

    @property
    def names(self) -> list[str]:
        return [a.name for a in self.attributes]

    def index(self, name: str) -> int:
        if name not in self._idx:
            raise KeyError(f"unknown attribute {name!r} in schema {self.version}")
        return self._idx[name]

    def label(self, i: int, t: int) -> str:
        a = self.attributes[i]
        return f"{a.name}>={a.levels[t]}"

    def parse_label(self, s: str) -> tuple[int, int]:
        name, lab = s.split(">=", 1)
        i = self.index(name.strip())
        labels = [str(x) for x in self.attributes[i].levels]
        if lab.strip() not in labels:
            raise KeyError(f"unknown level {lab!r} for {name}")
        return i, labels.index(lab.strip())

    def issued_labels(self, x) -> list[str]:
        """Every threshold claim a holder with true profile x receives."""
        return [self.label(i, t) for i, xi in enumerate(x) for t in range(1, int(xi) + 1)]

    def profile_labels(self, w) -> list[str]:
        """The strongest claim per attribute for a presented profile w (weaker ones are implied)."""
        return [self.label(i, int(t)) for i, t in enumerate(w) if t > 0]

    def states(self) -> np.ndarray:
        return np.array(list(itertools.product(*[range(s) for s in self.dims])), dtype=np.int16)


# Payment-assurance claims used by the synthetic real-time payment workload. These are the v1 chains with the
# boolean claims reduced to "shown / not shown", since a holder never presents a negative boolean.
PAYMENT_SCHEMA = Schema("vista2-payment-2026.1", [
    Attribute("auth_strength", ("none", "A1", "A2", "A3"), ttl_s=600),
    Attribute("device_continuity", ("none", "new", "established")),
    Attribute("credential_age", ("none", "0d", "30d", "90d")),
    Attribute("phishing_resistant", ("none", "true"), ttl_s=600),
    Attribute("dynamic_binding", ("none", "true"), ttl_s=600),
    Attribute("account_age", ("none", "0d", "365d", "730d")),
])

# Delegated-authority claims for agent-initiated payments (blueprint Section 6). Each chain's level is the
# minimum over the delegation links user -> platform -> agent -> sub-agent.
AGENT_SCHEMA = Schema("vista2-agent-2026.1", [
    Attribute("mandate_kind", ("none", "intent_hnp", "cart_hp")),
    Attribute("scope_coverage", ("none", "outside", "within_lt25", "within_ge25")),
    Attribute("agent_identity", ("none", "registered", "attested_runtime")),
    Attribute("tool_policy", ("none", "logged", "enforced")),
    Attribute("user_reconfirmation", ("none", "within_24h", "within_10m")),
])
