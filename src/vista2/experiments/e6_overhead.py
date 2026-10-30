"""E6: what does v2 cost in the payment path?

CPU-path timings on this machine (network latency is not modelled): issuing the payment-bound credential
and the cumulative claim (sending bank), presenting (holder), verifying both (receiver), and scoring with a
control-typed model, a W-repaired model (exact up-set enumeration) and a box-repaired model, single payment
at a time and in batches. Capsule sizes in bytes.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .. import cumulative, datasets, payment as paymod, vsac
from ..claims import PAYMENT_SCHEMA as S
from ..control import variants
from ..crypto.signed import SigningKey
from ..repair import ExactRepair, TreeBoxRepair
from ..trust import TrustRegistry
from ..util import percentiles, write_csv
from .common import fit_variant


def _row(stage, ms, extra=None):
    p = percentiles(ms)
    r = {"stage": stage, "n": p["n"], "mean_ms": p["mean"], "p50_ms": p["p50"], "p95_ms": p["p95"], "p99_ms": p["p99"]}
    r.update(extra or {})
    return r


def run(cfg, out, log=print):
    rng = np.random.default_rng(cfg.get("seed", 1))
    n = cfg.get("e6_payments", 2000)
    trust = TrustRegistry()
    k_send = SigningKey()
    trust.register("FI001", k_send, "issuer", 0, 10 ** 10)
    counter = cumulative.SpendCounter()
    ver = vsac.Verifier("FI002", {S.version: S}, trust, vsac.VerifyConfig(require_cumulative=True))
    t = datasets.load("synthetic", cfg.get("synthetic_size"))
    vs = {v.name: v for v in variants(t, ("unconstrained", "control_typed"))}
    ct = fit_variant(t, vs["control_typed"], "lightgbm", seed=1)
    un = fit_variant(t, vs["unconstrained"], "lightgbm", seed=1)
    exact_w = ExactRepair(un.model.risk, t, un.columns, ("W",))
    box_w = TreeBoxRepair(un.model, t, un.columns, ("W",))
    box_full = TreeBoxRepair(un.model, t, un.columns, ("W", "F"))
    Xte = t.X[t.split == 2].reset_index(drop=True)
    times = {k: [] for k in ("issue_credential", "issue_cumulative", "present", "verify", "ct_score",
                             "receiver_total")}
    sizes, sizes_nocum = [], []
    for j in range(n):
        now = 1_700_000_000 + j
        p = paymod.Payment("FI001", "FI002", f"P{j}", f"B{j}", int(rng.integers(100, 10 ** 6)), "USD", now * 1000, f"T{j}")
        x = tuple(int(rng.integers(1, a.size)) for a in S.attributes)
        nonce = ver.nonces.issue("FI001", 1, now)[0]
        t0 = time.perf_counter()
        cred, disc = vsac.issue(p=p, schema=S, issuer=k_send, x=x, nonce=nonce, now=now)
        t1 = time.perf_counter()
        c = counter.reserve(p.payer_ref, cumulative.window_of(now, 86_400), p.amount_minor)
        cc = cumulative.issue_claim(k_send, "FI001", p.payer_ref, cumulative.window_of(now, 86_400), c, paymod.pid(p), nonce)
        t2 = time.perf_counter()
        blob = vsac.present(cred, disc, S, x, cc)
        t3 = time.perf_counter()
        res = ver.verify(p, blob, now)
        t4 = time.perf_counter()
        row = Xte.iloc[[j % len(Xte)]]
        ct.risk(row)
        t5 = time.perf_counter()
        assert res.ok, res.reason
        times["issue_credential"].append((t1 - t0) * 1e3)
        times["issue_cumulative"].append((t2 - t1) * 1e3)
        times["present"].append((t3 - t2) * 1e3)
        times["verify"].append((t4 - t3) * 1e3)
        times["ct_score"].append((t5 - t4) * 1e3)
        times["receiver_total"].append((t5 - t3) * 1e3)
        sizes.append(len(blob))
        sizes_nocum.append(len(vsac.present(cred, disc, S, x, None)))
    rows = [_row(k, v) for k, v in times.items()]
    rows.append({"stage": "capsule_bytes", "n": n, "mean_ms": "", "bytes_mean": float(np.mean(sizes)),
                 "bytes_p99": float(np.percentile(sizes, 99)), "bytes_without_cumulative": float(np.mean(sizes_nocum))})
    q = Xte.sample(cfg.get("e6_repair_queries", 300), random_state=1)
    for name, fn in (("score_unconstrained", un.risk), ("score_exact_w_repair", exact_w.risk),
                     ("score_box_w_repair", box_w.risk), ("score_box_full_repair", box_full.risk)):
        lat = []
        for i in range(len(q)):
            t0 = time.perf_counter()
            fn(q.iloc[[i]])
            lat.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        fn(q)
        batch = (time.perf_counter() - t0) * 1e3 / len(q)
        rows.append(_row(name, lat, {"batch_ms_per_payment": batch}))
    calls = [exact_w.upset_size(r) for _, r in q.iterrows()]
    rows.append({"stage": "exact_w_repair_upset_size", "n": len(calls), "mean_ms": "",
                 "upset_mean": float(np.mean(calls)), "upset_max": int(np.max(calls))})
    log(f"  e6: verify p99 {percentiles(times['verify'])['p99']:.3f} ms, receiver p99 "
        f"{percentiles(times['receiver_total'])['p99']:.3f} ms, capsule {np.mean(sizes):.0f} bytes")
    return [write_csv(os.path.join(out, "e6_overhead.csv"), rows)]
