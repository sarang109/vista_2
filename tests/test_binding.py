import json
import threading

from vista2 import cumulative, payment as paymod, vsac
from vista2.claims import PAYMENT_SCHEMA as S
from vista2.crypto.signed import SigningKey
from vista2.trust import TrustRegistry


def _setup():
    trust = TrustRegistry()
    k = SigningKey()
    trust.register("FI1", k, "issuer", 0, 10 ** 10)
    v = vsac.Verifier("FI2", {S.version: S}, trust, vsac.VerifyConfig(require_cumulative=True))
    p = paymod.Payment("FI1", "FI2", "P1", "B1", 12_345, "USD", 1_700_000_000_000, "T1")
    return trust, k, v, p


def _capsule(k, v, p, w, now=1_700_000_000, c=None):
    nonce = v.nonces.issue("FI1", 1, now)[0]
    x = tuple(a.d for a in S.attributes)
    cred, disc = vsac.issue(p=p, schema=S, issuer=k, x=x, nonce=nonce, now=now)
    cc = cumulative.issue_claim(k, "FI1", "P1", cumulative.window_of(now, 86_400), c or p.amount_minor,
                                paymod.pid(p), nonce)
    return vsac.present(cred, disc, S, w, cc), now


def test_withholding_is_accepted_and_reported():
    _, k, v, p = _setup()
    w = (1, 0, 2, 0, 1, 0)
    blob, now = _capsule(k, v, p, w)
    res = v.verify(p, blob, now)
    assert res.ok and res.profile == w and res.cumulative == p.amount_minor


def test_upgrade_and_rewrites_are_rejected():
    _, k, v, p = _setup()
    blob, now = _capsule(k, v, p, (1, 0, 0, 0, 0, 0))
    o = json.loads(blob)
    o["disclosed"][0][1] = S.label(0, 3)
    assert v.verify(p, vsac.canonical_json(o), now).reason == "disclosure_not_committed"
    blob, now = _capsule(k, v, p, (1, 0, 0, 0, 0, 0))
    o = json.loads(blob)
    o["cumulative"]["body"]["c"] = 1
    assert v.verify(p, vsac.canonical_json(o), now).reason == "cumulative_bad_signature"
    blob, now = _capsule(k, v, p, (1, 0, 0, 0, 0, 0))
    assert v.verify(p.with_(amount_minor=1), blob, now).reason == "pid_mismatch"


def test_atomic_counter_bounds_parallel_parts():
    ctr = cumulative.SpendCounter()
    acc = []
    lock = threading.Lock()

    def go():
        c = ctr.reserve("K", 0, 100)
        if c <= 1_000:
            with lock:
                acc.append(100)
        else:
            ctr.release("K", 0, 100)

    ts = [threading.Thread(target=go) for _ in range(40)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sum(acc) == 1_000
