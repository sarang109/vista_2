"""SUMMARY.md for a run: the headline tables and both evidence gates, built only from the run's CSVs."""
from __future__ import annotations

import os

from .util import read_csv


def _load(out, name):
    p = os.path.join(out, name)
    return read_csv(p) if os.path.exists(p) else []


def _num(v, pct=False, digits=3):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v) if v not in (None, "") else ""
    if pct:
        return f"{100 * x:.1f}%"
    return f"{x:.{digits}f}"


def _pm(r, k, pct=True):
    sd = r.get(k + "_sd")
    base = _num(r.get(k), pct)
    if sd not in (None, "") and float(sd) > 0:
        base += f" ± {_num(sd, pct)}" if pct else f" ± {_num(sd)}"
    return base


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def gate_a(e2_rows, e1_rows) -> list[str]:
    """Gate A: on IEEE-CIS or BAF, a standard model shows a realized gap and control-typed training costs at
    most 0.02 AUC."""
    lines = []
    for d in ("ieee_cis", "baf"):
        base = {r["variant"]: r for r in e2_rows if r["dataset"] == d and r["attack"] == "truthful"}
        full = {r["variant"]: r for r in e2_rows if r["dataset"] == d and r["attack"] == "full"}
        if "control_typed" not in base:
            lines.append(f"- {d}: not run")
            continue
        ref = "unconstrained" if "unconstrained" in base else "naive"
        cost = float(base[ref]["auc"]) - float(base["control_typed"]["auc"])
        gap = float(full[ref]["fraud_accepted_attack"]) - float(base[ref]["fraud_accepted_truthful"])
        ok = gap > 0 and cost <= 0.02
        lines.append(f"- {d}: realized gap of the {ref} model {100 * gap:.1f} points; control-typed AUC cost "
                     f"{cost:.3f} -> Gate A **{'passes' if ok else 'fails'}** on this dataset")
    return lines


def write(out, manifest):
    L = [f"# VISTA v2 results: run `{manifest['name']}`", ""]
    v = manifest["versions"]
    L += [f"Python {v['python']}, LightGBM {v['lightgbm']}, XGBoost {v['xgboost']}, scikit-learn {v['scikit-learn']}, "
          f"{v['cpus']} CPUs ({v['machine']}). Seeds: {manifest['config'].get('seeds')}. "
          "Fraud acceptance is measured at the threshold that holds 5% of legitimate validation payments; ± is the "
          "standard deviation over seeds.", ""]
    L += ["| step | status | seconds |", "|---|---|---|"]
    L += [f"| {s['step']} {s['description']} | {s['status']} | {s['seconds']} |" for s in manifest["steps"]]
    L.append("")

    e0 = _load(out, "e0_verification.csv")
    if e0:
        bad = [r for r in e0 if r["passed"] not in ("True", "expected to fail")]
        L += ["## E0 theory checks", "",
              f"{len(e0)} checks, {len(bad)} failing. The non-atomic counter is expected to break the bound; it "
              "is the control for Algorithm 3's atomicity requirement.", ""]
        L.append(_table(["id", "check", "instances", "failures"],
                        [[r["id"], r["check"], r["instances"], r["failures"]] for r in e0]))
        L.append("")

    e1 = _load(out, "e1_gap.csv")
    if e1:
        L += ["## E1 evaluation gap (Theorem 2)", ""]
        rows = []
        for r in e1:
            if r["attack"] == "truthful":
                continue
            rows.append([r["dataset"], r["learner"], r["attack"], _pm(r, "auc", False), _pm(r, "fraud_accepted_truthful"),
                         _pm(r, "fraud_accepted_attack"), _pm(r, "gap")])
        L.append(_table(["dataset", "learner", "attacker", "AUC", "evaluated", "realized", "gap"], rows))
        L.append("")

    e2 = _load(out, "e2_defences.csv")
    if e2:
        L += ["## E2 cost of each defence", ""]
        for d in dict.fromkeys(r["dataset"] for r in e2):
            sub = [r for r in e2 if r["dataset"] == d]
            vs = list(dict.fromkeys(r["variant"] for r in sub))
            rows = []
            for vname in vs:
                t = next(r for r in sub if r["variant"] == vname and r["attack"] == "truthful")
                w = next((r for r in sub if r["variant"] == vname and r["attack"] == "withholding"), {})
                f = next((r for r in sub if r["variant"] == vname and r["attack"] == "full"), {})
                rows.append([vname, _pm(t, "auc", False), _pm(t, "legit_held"), _pm(t, "fraud_accepted_truthful"),
                             _pm(w, "fraud_accepted_attack"), _pm(f, "fraud_accepted_attack"),
                             _pm(f, "fraud_value_accepted_attack")])
            L += [f"**{d}**", "", _table(["model", "AUC", "legit held", "fraud acc. truthful", "withholding",
                                          "full attack", "fraud value acc. (full)"], rows), ""]
        bve = _load(out, "e2_box_vs_exact.csv")
        if bve:
            L += ["Full repair of the unconstrained synthetic model, exact enumeration against the box relaxation "
                  "(subsample):", "",
                  _table(["repair", "AUC", "legit held", "fraud accepted"],
                         [[r["repair"], _num(r["auc"]), _num(r["legit_held"], True), _num(r["fraud_accepted"], True)]
                          for r in bve]), ""]
        L += ["### Gate A", ""] + gate_a(e2, e1) + [""]

    e3 = _load(out, "e3_splitting.csv")
    if e3:
        L += ["## E3 splitting (Theorem 3), PaySim", ""]
        L.append(_table(["rule", "K", "legit held", "fraud value accepted", "max exposure", "bound holds"],
                        [[r["rule"], r["K"], _num(r["legit_held"], True), _num(r["fraud_value_accepted"], True),
                          _num(r["max_exposure"], digits=0), r.get("bound_holds", "")] for r in e3
                         if r["K"] in ("1", "4", "16", "64")]))
        at = _load(out, "e3_atomicity.csv")
        if at:
            L += ["", _table(["counter", "trials", "trials over bound", "worst accepted total", "claim p99 ms"],
                             [[r["counter"], r["trials"], r["trials_over_bound"], r["worst_accepted_total"],
                               _num(r["claim_ms_p99"])] for r in at])]
        L.append("")

    e4 = _load(out, "e4_agentic.csv")
    if e4:
        L += ["## E4 agentic adversaries (Gate B)", ""]
        atts = ["truthful", "withhold_split", "persuasive_text", "merchant_retry", "chain_strip", "full"]
        rows = []
        for vname in dict.fromkeys(r["variant"] for r in e4):
            row = [vname]
            for a in atts:
                r = next((x for x in e4 if x["variant"] == vname and x["attack"] == a), None)
                row.append(_pm(r, "fraud_accepted_truthful" if a == "truthful" else "fraud_accepted_attack") if r else "")
            rows.append(row)
        L.append(_table(["model"] + atts, rows))
        g = [r for r in _load(out, "e4_gate_b.csv") if r["variant"] == "GATE_B"]
        if g:
            L += ["", f"Gate B **{'passes' if g[0]['consistent'] == 'True' else 'fails'}** (control-typed gains under "
                  f"no adversary; the naive model gains under {g[0]['gain']} of 4)."]
        L.append("")

    e5 = _load(out, "e5_repeated_exact.csv")
    if e5:
        L += ["## E5 repeated agent payments (Theorems 4 and 5)", ""]
        rows = []
        for pol in dict.fromkeys(r["policy"] for r in e5):
            for k in ("DET", "exp1.0_fresh", "exp1.0_sticky"):
                s = {r["n"]: r for r in e5 if r["policy"] == pol and r["kernel"] == k}
                if s:
                    rows.append([pol, k, _num(s["1"]["Lambda"]), _num(s.get("5", s["1"])["Lambda"]),
                                 _num(s.get("10", s["1"])["Lambda"])])
        L.append(_table(["policy", "selection", "Λ1 (bits)", "Λ5", "Λ10"], rows))
        zl = [r for r in _load(out, "e5_zero_leakage_and_coverage.csv") if r.get("zero_leakage_feasible")]
        if zl:
            L += ["", "Zero-leakage feasible (Theorem 4): " + ", ".join(f"{r['policy']} {r['zero_leakage_feasible']}" for r in zl)]
        sim = _load(out, "e5_repeated_simulation.csv")
        if sim:
            rows = [[r["policy"], r["strategy"], r["n"], _num(r["mean_distinct_branches"], digits=2),
                     _num(r["attacker_lift"], True)] for r in sim if r["n"] in ("1", "4", "12")]
            L += ["", _table(["policy", "strategy", "payment n", "branches revealed", "attacker lift"], rows)]
        L.append("")

    e6 = _load(out, "e6_overhead.csv")
    if e6:
        L += ["## E6 payment-path overhead", ""]
        L.append(_table(["stage", "p50 ms", "p99 ms", "batch ms / payment", "bytes"],
                        [[r["stage"], _num(r.get("p50_ms")), _num(r.get("p99_ms")), _num(r.get("batch_ms_per_payment")),
                          _num(r.get("bytes_mean"), digits=0)] for r in e6]))
        L.append("")

    e7 = _load(out, "e7_summary.csv")
    if e7:
        r = e7[0]
        L += ["## E7 binding layer", "",
              f"{r['accepted']} of {r['trials']} targeted mutations ({r['attacks']} attack types) accepted by the full "
              f"verifier; withholding is accepted by design ({r['withhold_accepted_by_design']} trials) and handled by "
              "the decision rule.", ""]

    e8 = _load(out, "e8_typing_sensitivity.csv")
    if e8:
        L += ["## E8 typing sensitivity", ""]
        L.append(_table(["dataset", "change", "kind", "control-typed AUC", "AUC cost", "naive realized"],
                        [[r["dataset"], r["change"], r["kind"], _num(r["ct_auc"]), _num(r["auc_cost"]),
                          _num(r["naive_realized"], True)] for r in e8]))
        L.append("")

    e9 = _load(out, "e9_rollout.csv")
    if e9:
        L += ["## E9 rollout stage", ""]
        rows = [[r["nonparticipating"], r["variant"], _pm(r, "fraud_accepted_truthful"), _pm(r, "fraud_accepted_attack"),
                 _pm(r, "gap")] for r in e9 if r["attack"] == "withholding"]
        L.append(_table(["no-claim banks", "model", "truthful", "withholding", "gain"], rows))
        L.append("")

    e10 = _load(out, "e10_comparison.csv")
    if e10:
        L += ["## E10 measured comparison of designs (agentic workload)", ""]
        rows = []
        for d in dict.fromkeys(r["design"] for r in e10):
            g = {r["attack"]: r for r in e10 if r["design"] == d}
            rows.append([d, _pm(g["full"], "legit_held"), _pm(g["full"], "fraud_accepted_truthful"),
                         _pm(g["withholding"], "fraud_accepted_attack"), _pm(g["f_only"], "fraud_accepted_attack"),
                         _pm(g["full"], "fraud_accepted_attack"), _num(g["full"].get("disclosed_bits"), digits=2)])
        L.append(_table(["design", "legit held", "truthful", "withholding", "F only", "full", "claim bits seen"], rows))
        L.append("")

    e11 = _load(out, "e11_attacker_strength.csv")
    if e11:
        L += ["## E11 attacker strength and knowledge", ""]
        rows = [[r["variant"], r["attack"], r.get("knowledge", ""),
                 _num(r.get("fraud_accepted_attack") or r["fraud_accepted_truthful"], True)] for r in e11]
        L.append(_table(["model", "attacker", "knowledge", "fraud accepted"], rows))
        L.append("")

    e13 = _load(out, "e13_repair_scaling.csv")
    if e13:
        L += ["## E13 repair cost against lattice size", ""]
        L.append(_table(["claims", "lattice", "payments", "exact ms", "exact calls", "box ms", "box excess (mean)"],
                        [[r["claims"], r["lattice_points"], r["queries"], _num(r["exact_ms_per_payment"], digits=2),
                          _num(r["exact_model_calls_per_payment"], digits=0), _num(r["box_ms_per_payment"], digits=2),
                          _num(r["box_minus_exact_mean"])] for r in e13]))
        L.append("")
    e14 = _load(out, "e14_bounded.csv")
    if e14:
        L += ["## E14 bounded attacker", "",
              "Fraud accepted under attackers limited in how many freely chosen fields they may change and how far "
              "each may move; withholding stays free.", ""]
        order = ["truthful", "fields_0", "fields_1", "fields_2", "fields_3", "fields_5", "window_0.02", "window_0.05",
                 "window_0.1", "window_0.25", "window_0.5", "fields_2_window_0.1", "unbounded"]
        rows = []
        for d in dict.fromkeys(r["dataset"] for r in e14):
            for vname in dict.fromkeys(r["variant"] for r in e14 if r["dataset"] == d):
                g = {r["attack"]: r for r in e14 if r["dataset"] == d and r["variant"] == vname}
                rows.append([d, vname] + [_pm(g[a], "fraud_accepted_truthful" if a == "truthful" else
                                              "fraud_accepted_attack") if a in g else "" for a in order])
        L.append(_table(["dataset", "model"] + [a.replace("_", " ") for a in order], rows))
        L.append("")
    figs = sorted(os.listdir(os.path.join(out, "figures"))) if os.path.isdir(os.path.join(out, "figures")) else []
    if figs:
        L += ["## Figures", ""] + [f"- `figures/{f}`" for f in figs] + [""]
    with open(os.path.join(out, "SUMMARY.md"), "w") as fh:
        fh.write("\n".join(L))
