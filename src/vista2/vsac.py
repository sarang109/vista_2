"""Verifiable Security Assurance Capsule, holder-driven form.

The sending bank (issuer, honest) signs a selective-disclosure credential bound to one payment: a salted
digest for every threshold claim the payer actually has, padded with decoys to the full chain length. The
holder (the payer's wallet or AI agent) chooses which disclosures travel with the payment. A compromised
holder can therefore withhold or weaken claims but cannot strengthen, transplant or replay them; that
split between "may withhold" and "cannot forge" is the W type of the evidence game.

Wire format: canonical JSON {"cred": signed credential, "disclosed": [[salt, label], ...],
"cumulative": optional issuer-signed cumulative-spend claim (cumulative.py)}.

Invariants checked by `Verifier`, each switchable for ablations: credential signature, issuer status and
institution match, transaction binding (PID), audience binding, freshness, single-use receiver nonce,
replay cache, schema version, disclosure commitment, cumulative-claim signature and binding. Fails closed.
"""
from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field

from . import cumulative, payment as paymod
from .claims import Schema
from .crypto import signed
from .trust import TrustError, TrustRegistry
from .util import b64d, b64e, canonical_json, sha256

VERSION = "vsac-2"


def _digest(salt: bytes, label: str) -> bytes:
    return sha256(canonical_json([b64e(salt), label]))


# ============================================================================ issuer side
def issue(*, p: paymod.Payment, schema: Schema, issuer: signed.SigningKey, x, nonce: str, now: int,
          lifetime_s: int = 120, event_time: dict | None = None) -> tuple[dict, dict]:
    """Credential for payment p and true profile x. Returns (credential, disclosures {label: salt})."""
    pid = paymod.pid(p)
    labels = schema.issued_labels(x)
    disclosures, digests = {}, []
    for lab in labels:
        s = secrets.token_bytes(16)
        disclosures[lab] = b64e(s)
        digests.append(_digest(s, lab))
    while len(digests) < sum(a.d for a in schema.attributes):
        digests.append(secrets.token_bytes(32))  # decoys: the digest count does not reveal x
    digests.sort()
    event_time = event_time or {}
    ttl_exp = [int(event_time.get(a.name, now)) + a.ttl_s for a in schema.attributes if a.ttl_s]
    body = {"v": VERSION, "pid": pid, "sender": p.sender_fi, "aud": p.receiver_fi, "iat": now,
            "exp": min([now + lifetime_s] + ttl_exp), "nonce": nonce, "schema": schema.version,
            "kid": issuer.key_id, "digests": [b64e(d) for d in digests]}
    return {"body": body, "sig": b64e(issuer.sign(canonical_json(body)))}, disclosures


# ============================================================================ holder side
def present(cred: dict, disclosures: dict, schema: Schema, w, cum_claim: dict | None = None) -> bytes:
    """Holder presentation of profile w (must satisfy w <= x; otherwise the label is not available)."""
    reveal = schema.profile_labels(w)
    obj = {"cred": cred, "disclosed": [[disclosures[lab], lab] for lab in reveal]}
    if cum_claim is not None:
        obj["cumulative"] = cum_claim
    return canonical_json(obj)


# ============================================================================ receiver side
@dataclass
class VerifyConfig:
    check_binding: bool = True
    check_audience: bool = True
    check_freshness: bool = True
    check_nonce: bool = True
    check_replay: bool = True
    check_status: bool = True
    check_signature: bool = True
    check_commitment: bool = True
    require_cumulative: bool = False
    check_cumulative: bool = True
    clock_skew_s: int = 5
    max_lifetime_s: int = 900


@dataclass
class VerifyResult:
    ok: bool
    reason: str
    profile: tuple | None = None
    cumulative: int | None = None
    labels: list = field(default_factory=list)


class NonceStore:
    """Receiver-issued single-use challenges, pre-fetched in batches by each sender."""

    def __init__(self, ttl_s: int = 3600):
        self.ttl = ttl_s
        self.live: dict = {}

    def issue(self, sender: str, k: int, now: int) -> list[str]:
        out = []
        for _ in range(k):
            n = secrets.token_hex(16)
            self.live[(sender, n)] = now + self.ttl
            out.append(n)
        return out

    def consume(self, sender: str, nonce: str, now: int) -> bool:
        exp = self.live.pop((sender, nonce), None)
        return exp is not None and now <= exp


class Verifier:
    def __init__(self, receiver_id: str, schemas: dict, trust: TrustRegistry, cfg: VerifyConfig | None = None,
                 nonces: NonceStore | None = None, window_s: int = 86_400):
        self.rid = receiver_id
        self.schemas = schemas
        self.trust = trust
        self.cfg = cfg or VerifyConfig()
        self.nonces = nonces or NonceStore()
        self.seen: set = set()
        self.window_s = window_s

    def _fail(self, reason):
        return VerifyResult(False, reason)

    def verify(self, p: paymod.Payment, blob: bytes | None, now: int) -> VerifyResult:
        c = self.cfg
        if blob is None:
            return self._fail("missing_capsule")
        try:
            obj = json.loads(blob)
            cred, disclosed = obj["cred"], obj["disclosed"]
            h = cred["body"]
            _ = h["pid"], h["sender"], h["aud"], h["iat"], h["exp"], h["nonce"], h["kid"], h["digests"]
        except Exception:
            return self._fail("malformed")
        if h.get("v") != VERSION:
            return self._fail("unknown_version")
        schema = self.schemas.get(h.get("schema"))
        if schema is None:
            return self._fail("unknown_schema")
        try:
            rec = self.trust.resolve(h["kid"], now, role="issuer", check_status=c.check_status)
        except TrustError as e:
            return self._fail(e.reason)
        if rec.institution != h["sender"]:
            return self._fail("issuer_mismatch")
        if c.check_signature:
            try:
                ok = signed.verify(rec.public_bytes, b64d(cred["sig"]), canonical_json(h))
            except Exception:
                ok = False
            if not ok:
                return self._fail("bad_signature")
        pid = paymod.pid(p)
        if c.check_binding and (h["pid"] != pid or h["sender"] != p.sender_fi):
            return self._fail("pid_mismatch")
        if c.check_audience and (h["aud"] != self.rid or p.receiver_fi != self.rid):
            return self._fail("audience_mismatch")
        if c.check_freshness:
            if now + c.clock_skew_s < h["iat"]:
                return self._fail("not_yet_valid")
            if now > h["exp"] or h["exp"] - h["iat"] > c.max_lifetime_s:
                return self._fail("expired")
        prof = [0] * schema.n
        labels = []
        try:
            digs = set(h["digests"])
            for salt, lab in disclosed:
                if c.check_commitment and b64e(_digest(b64d(salt), lab)) not in digs:
                    return self._fail("disclosure_not_committed")
                i, t = schema.parse_label(lab)
                prof[i] = max(prof[i], t)
                labels.append(lab)
        except Exception:
            return self._fail("unknown_claim")
        cum = None
        if "cumulative" in obj:
            if c.check_cumulative:
                try:
                    cum = cumulative.verify_claim(obj["cumulative"], self.trust, now, h["pid"], h["nonce"],
                                                  h["sender"], self.window_s, check_status=c.check_status)
                except cumulative.ClaimError as e:
                    return self._fail(e.reason)
            else:
                cum = int(obj["cumulative"]["body"]["c"])
        elif c.require_cumulative:
            return self._fail("missing_cumulative")
        if c.check_replay and (h["pid"], h["nonce"]) in self.seen:
            return self._fail("replay")
        if c.check_nonce and not self.nonces.consume(h["sender"], h["nonce"], now):
            return self._fail("nonce_unknown_or_reused")
        if c.check_replay:
            self.seen.add((h["pid"], h["nonce"]))
        return VerifyResult(True, "ok", tuple(prof), cum, labels)
