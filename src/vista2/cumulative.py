"""Algorithm 3: issuer-signed cumulative-spend claim (enables Theorem 3.2).

On each payment request the issuer atomically adds the amount to a counter keyed by (mandate or
beneficiary key K, window) and signs (K, window, c, PID, nonce). The receiver verifies the signature and
the PID binding, rejects when the claim is missing, and keeps acceptance non-increasing in c. Because the
counter only grows within a window, the last accepted part of a split payment carries c equal to the
accepted total, so the total accepted per window is at most the largest c the rule accepts.

The atomic counter itself is the budget ledger of CAPE and MERIDIAN; what this module adds is carrying it
to another institution as a PID-bound third-party fact. `RacyCounter` is the non-atomic variant used to
show why atomicity is required (two parallel requests both read the old value).
"""
from __future__ import annotations

import threading
import time

from .crypto import signed
from .trust import TrustError, TrustRegistry
from .util import b64d, b64e, canonical_json

CLAIM_TYPE = "vista2-cumulative-v1"


class ClaimError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def window_of(ts: int, window_s: int) -> int:
    return int(ts) // int(window_s)


class SpendCounter:
    """Atomic per-(key, window) counter. reserve() returns the new cumulative value including `amount`."""

    def __init__(self):
        self._lock = threading.Lock()
        self.counter: dict = {}

    def reserve(self, key: str, window: int, amount: int) -> int:
        with self._lock:
            c = self.counter.get((key, window), 0) + int(amount)
            self.counter[(key, window)] = c
            return c

    def release(self, key: str, window: int, amount: int):
        """Undo a reservation for a payment the issuer did not send (e.g. the receiver rejected it)."""
        with self._lock:
            self.counter[(key, window)] = self.counter.get((key, window), 0) - int(amount)


class RacyCounter(SpendCounter):
    """Read-modify-write without a lock, with a delay between read and write to widen the race."""

    def __init__(self, delay_s: float = 0.0005):
        super().__init__()
        self.delay = delay_s

    def reserve(self, key: str, window: int, amount: int) -> int:
        old = self.counter.get((key, window), 0)
        time.sleep(self.delay)
        c = old + int(amount)
        self.counter[(key, window)] = c
        return c


def issue_claim(issuer: signed.SigningKey, institution: str, key: str, window: int, c: int, pid: str,
                nonce: str) -> dict:
    body = {"typ": CLAIM_TYPE, "iss": institution, "kid": issuer.key_id, "key": key, "window": int(window),
            "c": int(c), "pid": pid, "nonce": nonce}
    return {"body": body, "sig": b64e(issuer.sign(canonical_json(body)))}


def verify_claim(claim: dict, trust: TrustRegistry, now: int, pid: str, nonce: str, sender: str,
                 window_s: int, check_status: bool = True) -> int:
    try:
        body, sig = claim["body"], claim["sig"]
        if body.get("typ") != CLAIM_TYPE:
            raise ClaimError("cumulative_bad_type")
        rec = trust.resolve(body["kid"], now, role="issuer", check_status=check_status)
    except TrustError as e:
        raise ClaimError("cumulative_" + e.reason)
    except (KeyError, TypeError):
        raise ClaimError("cumulative_malformed")
    if rec.institution != sender or body.get("iss") != sender:
        raise ClaimError("cumulative_issuer_mismatch")
    if not signed.verify(rec.public_bytes, b64d(sig), canonical_json(body)):
        raise ClaimError("cumulative_bad_signature")
    if body["pid"] != pid or body["nonce"] != nonce:
        raise ClaimError("cumulative_binding_mismatch")
    if int(body["window"]) != window_of(now, window_s):
        raise ClaimError("cumulative_wrong_window")
    return int(body["c"])
