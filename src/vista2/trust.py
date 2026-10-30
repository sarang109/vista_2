"""Trust registry: issuer keys, validity windows, rotation, revocation, status, and the T-type facts the
registry signs about institutions (which claims an institution is able to issue, whether a sender is a
registered agent).

The capability record is what turns "this claim is missing" from a holder-chosen message into a
third-party fact: if the registry says institution i can issue claim c, a payment from i without c was
withheld by the holder, not left out by the bank.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .crypto.signed import SigningKey


class TrustError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class KeyRecord:
    institution: str
    key_id: str
    public_bytes: bytes
    role: str  # "issuer", "ledger" or "registry"
    not_before: int
    not_after: int
    revoked_at: int | None = None


@dataclass
class InstitutionRecord:
    institution: str
    capabilities: frozenset = field(default_factory=frozenset)  # claim names it can issue
    agent_platform: bool = False


class TrustRegistry:
    def __init__(self, max_status_staleness_s: int = 300):
        self.keys: dict[str, KeyRecord] = {}
        self.institutions: dict[str, InstitutionRecord] = {}
        self.online = True
        self.snapshot_time = 0
        self.max_staleness = max_status_staleness_s

    def register(self, institution: str, key: SigningKey, role: str, not_before: int, not_after: int) -> KeyRecord:
        rec = KeyRecord(institution, key.key_id, key.public_bytes, role, not_before, not_after)
        self.keys[key.key_id] = rec
        return rec

    def set_capabilities(self, institution: str, claims, agent_platform: bool = False):
        self.institutions[institution] = InstitutionRecord(institution, frozenset(claims), agent_platform)

    def capability(self, institution: str, claim: str) -> bool:
        rec = self.institutions.get(institution)
        return rec is not None and claim in rec.capabilities

    def revoke(self, key_id: str, at: int):
        self.keys[key_id].revoked_at = at

    def rotate(self, institution: str, role: str, new_key: SigningKey, now: int, validity: int,
               grace: int = 60) -> KeyRecord:
        for r in self.keys.values():
            if r.institution == institution and r.role == role and r.not_after > now + grace:
                r.not_after = now + grace
        return self.register(institution, new_key, role, now, now + validity)

    def resolve(self, key_id: str, now: int, role: str | None = None, check_status: bool = True) -> KeyRecord:
        rec = self.keys.get(key_id)
        if rec is None:
            raise TrustError("unknown_key")
        if role and rec.role != role:
            raise TrustError("wrong_key_role")
        if not check_status:
            return rec
        if self.online:
            self.snapshot_time = now
        elif now - self.snapshot_time > self.max_staleness:
            raise TrustError("status_unavailable")
        if not (rec.not_before <= now <= rec.not_after):
            raise TrustError("expired_key")
        if rec.revoked_at is not None and rec.revoked_at <= self.snapshot_time:
            raise TrustError("revoked_key")
        return rec
