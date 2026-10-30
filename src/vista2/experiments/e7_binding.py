"""E7: the binding layer under attack (v1's adversarial suite on the v2 holder-driven capsule).

Every mutation below must be rejected by a fully configured verifier, except `withhold_claims`, which is
the holder's legitimate choice and is accepted by design: it is the strategic presentation that Theorems
1-3 handle at the decision layer. Ablations switch single checks off to show which invariant stops which
attack. New in v2: attacks on the cumulative-spend claim (rewrite, replay, strip, wrong window, foreign
issuer, bit flip).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from .. import cumulative, payment as paymod, vsac
from ..claims import PAYMENT_SCHEMA as S
from ..crypto.signed import SigningKey
from ..trust import TrustRegistry
from ..util import b64d, b64e, canonical_json, write_csv

ABLATIONS = {
    "full": {},
    "no_transaction_binding": {"check_binding": False},
    "no_audience_binding": {"check_audience": False},
    "no_freshness": {"check_freshness": False},
    "no_nonce_no_replay_cache": {"check_nonce": False, "check_replay": False},
    "no_status_check": {"check_status": False},
    "no_signature_check": {"check_signature": False},
    "no_commitment_check": {"check_commitment": False},
    "no_cumulative_check": {"check_cumulative": False},
}
EXPECTED = {
    "replay_same_payment": "nonce + replay cache", "amount_substitution": "PID binding",
    "beneficiary_substitution": "PID binding", "currency_substitution": "PID binding",
    "cross_tx_transplant": "PID binding", "audience_redirect": "audience binding", "expired_capsule": "freshness",
    "claim_injection": "disclosure commitment", "claim_upgrade": "disclosure commitment",
    "digest_injection": "credential signature", "pid_rewrite": "credential signature",
    "signature_bitflip": "credential signature", "issuer_substitution": "issuer-institution match",
    "kid_swap": "key resolution + signature", "unknown_schema": "schema versioning", "version_downgrade": "version check",
    "malformed_json": "parser fails closed", "revoked_issuer_key": "status / revocation",
    "nonce_forgery": "receiver challenge", "missing_capsule": "fails closed",
    "cum_rewrite": "cumulative signature", "cum_replay": "cumulative PID binding", "cum_strip": "cumulative required",
    "cum_wrong_window": "cumulative window", "cum_foreign_issuer": "cumulative issuer match",
    "cum_bitflip": "cumulative signature", "withhold_claims": "accepted by design (decision layer)",
}
WINDOW = 86_400


def _mut(blob: bytes, fn) -> bytes:
    obj = json.loads(blob)
    fn(obj)
    return canonical_json(obj)


def _flip(sig: str) -> str:
    s = bytearray(b64d(sig))
    s[5] ^= 0x01
    return b64e(bytes(s))


def run(cfg, out, log=print):
    rng = np.random.default_rng(cfg.get("seed", 1) + 7)
    n_base = cfg.get("e7_base_payments", 200)
    trust = TrustRegistry()
    banks = [f"FI{i:03d}" for i in range(6)]
    keys = {b: SigningKey() for b in banks}
    for b in banks:
        trust.register(b, keys[b], "issuer", 0, 10 ** 10)
    counters = {b: cumulative.SpendCounter() for b in banks}
    agg = defaultdict(lambda: [0, 0])
    reasons = defaultdict(lambda: defaultdict(int))
    for j in range(n_base):
        s, r = rng.choice(len(banks), 2, replace=False)
        snd, rcv, r2 = banks[s], banks[r], banks[(r + 1) % len(banks) if (r + 1) % len(banks) != s else (r + 2) % len(banks)]
        now = 1_700_000_000 + j * 37
        p = paymod.Payment(snd, rcv, f"P{j:06d}", f"B{j:06d}", int(rng.integers(100, 10 ** 6)), "USD", now * 1000,
                           f"TX{j:08d}")
        p2 = p.with_(tx_id=p.tx_id + "X", amount_minor=p.amount_minor + 1)
        x = tuple(int(rng.integers(1, a.size)) for a in S.attributes)
        nonces = {k: rng.bytes(16).hex() for k in ("main", "p2", "stale", "rev", "redir")}
        win = cumulative.window_of(now, WINDOW)

        def cum_claim(payment, nonce, c=None, key=None, window=win, inst=snd):
            c = counters[snd].reserve(payment.payer_ref, window, payment.amount_minor) if c is None else c
            return cumulative.issue_claim(key or keys[snd], inst, payment.payer_ref, window, c, paymod.pid(payment), nonce)

        def capsule(payment, nonce, w=x, key=None, lifetime=120, when=now, cum=True):
            cred, disc = vsac.issue(p=payment, schema=S, issuer=key or keys[snd], x=x, nonce=nonce, now=when,
                                    lifetime_s=lifetime)
            return vsac.present(cred, disc, S, w, cum_claim(payment, nonce) if cum else None), cred, disc

        blob, cred, disc = capsule(p, nonces["main"])
        blob2, _, _ = capsule(p2, nonces["p2"])
        good_cum = json.loads(blob)["cumulative"]
        old_cum = json.loads(blob2)["cumulative"]
        rk = SigningKey()
        trust.register(snd, rk, "issuer", now - 100, now + 10_000)
        trust.revoke(rk.key_id, now - 50)
        p_rev = p.with_(tx_id=p.tx_id + "R")
        rev_blob, _, _ = capsule(p_rev, nonces["rev"], key=rk)
        p_nf = p.with_(tx_id=p.tx_id + "N")
        nf_blob, _, _ = capsule(p_nf, "f" * 32)
        foreign = keys[r2]
        p_redir = p.with_(receiver_fi=r2)
        weaker = tuple(max(v - 1, 0) for v in x)
        top_label = S.label(0, S.attributes[0].d)

        def inject(o):
            o["disclosed"].append([b64e(b"\x00" * 16), top_label])

        def upgrade(o):
            for item in o["disclosed"]:
                i, t = S.parse_label(item[1])
                if t < S.attributes[i].d:
                    item[1] = S.label(i, S.attributes[i].d)
                    return

        cases = [
            ("replay_same_payment", p, blob, now),
            ("amount_substitution", p.with_(amount_minor=p.amount_minor * 10), blob, now),
            ("beneficiary_substitution", p.with_(beneficiary_ref="BATTACKER"), blob, now),
            ("currency_substitution", p.with_(currency="EUR"), blob, now),
            ("cross_tx_transplant", p2, blob, now),
            ("audience_redirect", p_redir, blob, now),
            ("expired_capsule", p, blob, now + 3600),
            ("claim_injection", p, _mut(blob, inject), now),
            ("claim_upgrade", p, _mut(blob, upgrade), now),
            ("digest_injection", p, _mut(blob, lambda o: o["cred"]["body"]["digests"].append(b64e(b"\x01" * 32))), now),
            ("pid_rewrite", p2, _mut(blob, lambda o: o["cred"]["body"].__setitem__("pid", paymod.pid(p2))), now),
            ("signature_bitflip", p, _mut(blob, lambda o: o["cred"].__setitem__("sig", _flip(o["cred"]["sig"]))), now),
            ("issuer_substitution", p, capsule(p, nonces["main"], key=foreign)[0], now),
            ("kid_swap", p, _mut(blob, lambda o: o["cred"]["body"].__setitem__("kid", foreign.key_id)), now),
            ("unknown_schema", p, _mut(blob, lambda o: o["cred"]["body"].__setitem__("schema", "vista2-payment-1999.0")), now),
            ("version_downgrade", p, _mut(blob, lambda o: o["cred"]["body"].__setitem__("v", "vsac-1")), now),
            ("malformed_json", p, blob[: len(blob) // 2], now),
            ("revoked_issuer_key", p_rev, rev_blob, now),
            ("nonce_forgery", p_nf, nf_blob, now),
            ("missing_capsule", p, None, now),
            ("cum_rewrite", p, _mut(blob, lambda o: o["cumulative"]["body"].__setitem__("c", 1)), now),
            ("cum_replay", p, _mut(blob, lambda o: o.__setitem__("cumulative", old_cum)), now),
            ("cum_strip", p, _mut(blob, lambda o: o.pop("cumulative")), now),
            ("cum_wrong_window", p, _mut(blob, lambda o: o.__setitem__(
                "cumulative", cum_claim(p, nonces["main"], c=good_cum["body"]["c"], window=win - 1))), now),
            ("cum_foreign_issuer", p, _mut(blob, lambda o: o.__setitem__(
                "cumulative", cum_claim(p, nonces["main"], c=1, key=foreign, inst=snd))), now),
            ("cum_bitflip", p, _mut(blob, lambda o: o["cumulative"].__setitem__("sig", _flip(o["cumulative"]["sig"]))), now),
            ("withhold_claims", p, vsac.present(cred, disc, S, weaker, good_cum), now),
        ]
        for name, pay, b, when in cases:
            if name not in ("replay_same_payment", "withhold_claims") and b == blob and pay == p and when == now:
                continue  # the mutation was a no-op (e.g. every disclosed claim already at its top level)
            for abl, flags in ABLATIONS.items():
                cfgv = vsac.VerifyConfig(require_cumulative=True, **flags)
                rid = pay.receiver_fi if name == "audience_redirect" else rcv
                v = vsac.Verifier(rid, {S.version: S}, trust, cfgv, vsac.NonceStore(), WINDOW)
                for nn in list(nonces.values()) + ["f" * 32]:
                    if nn != "f" * 32:
                        v.nonces.live[(snd, nn)] = now + 3600
                if name == "replay_same_payment":
                    v.verify(p, blob, now)
                res = v.verify(pay, b, when)
                a = agg[(name, abl)]
                a[0] += res.ok
                a[1] += 1
                reasons[(name, abl)][res.reason] += 1
        trust.keys.pop(rk.key_id, None)
    rows = []
    for (name, abl), (acc, n) in sorted(agg.items()):
        rs = reasons[(name, abl)]
        rows.append({"attack": name, "ablation": abl, "trials": n, "accepted": acc, "acceptance_rate": acc / n,
                     "expected_defence": EXPECTED[name], "top_reason": max(rs, key=rs.get)})
    full = [r for r in rows if r["ablation"] == "full" and r["attack"] != "withhold_claims"]
    summary = [{"attacks": len(full), "trials": sum(r["trials"] for r in full),
                "accepted": sum(r["accepted"] for r in full),
                "withhold_accepted_by_design": next(r["accepted"] for r in rows if r["attack"] == "withhold_claims"
                                                    and r["ablation"] == "full")}]
    log(f"  e7: {summary[0]['accepted']} of {summary[0]['trials']} targeted mutations accepted (full verifier)")
    return [write_csv(os.path.join(out, "e7_binding_attacks.csv"), rows),
            write_csv(os.path.join(out, "e7_summary.csv"), summary)]
