"""Figures for the paper, regenerated from the CSVs of a run (results/<run>/figures/*.png)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .util import read_csv  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 200,
})


def _f(r, k, default=np.nan):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return default


def _load(out, name):
    p = os.path.join(out, name)
    return read_csv(p) if os.path.exists(p) else None


def _save(fig, out, name):
    d = os.path.join(out, "figures")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def _grouped_bars(ax, groups, series, values, errors=None):
    x = np.arange(len(groups))
    w = 0.8 / len(series)
    for j, s in enumerate(series):
        v = [values.get((g, s), np.nan) for g in groups]
        e = [errors.get((g, s), 0) for g in groups] if errors else None
        ax.bar(x + (j - (len(series) - 1) / 2) * w, v, w * 0.92, color=SERIES[j], label=s, yerr=e,
               error_kw={"elinewidth": 0.8, "ecolor": INK2, "capsize": 0})
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", " ") for g in groups], rotation=20, ha="right")
    ax.set_axisbelow(True)


def fig_defences(out):
    rows = _load(out, "e2_defences.csv")
    if not rows:
        return []
    dsets = list(dict.fromkeys(r["dataset"] for r in rows))
    fig, axes = plt.subplots(1, len(dsets), figsize=(4.2 * len(dsets), 3.4), squeeze=False)
    for ax, d in zip(axes[0], dsets):
        sub = [r for r in rows if r["dataset"] == d]
        groups = list(dict.fromkeys(r["variant"] for r in sub))
        vals, errs = {}, {}
        for r in sub:
            key = {"truthful": "truthful", "withholding": "withholding only", "full": "full attack"}[r["attack"]]
            m = "fraud_accepted_truthful" if r["attack"] == "truthful" else "fraud_accepted_attack"
            vals[(r["variant"], key)] = 100 * _f(r, m)
            errs[(r["variant"], key)] = 100 * _f(r, m + "_sd", 0)
        _grouped_bars(ax, groups, ["truthful", "withholding only", "full attack"], vals, errs)
        ax.set_title(d.replace("_", "-"))
        ax.set_ylim(0, 100)
        ax.set_ylabel("fraud accepted at 5% legitimate hold (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    return [_save(fig, out, "fig_e2_defences.png")]


def fig_gap(out):
    rows = _load(out, "e1_gap.csv")
    if not rows:
        return []
    full = [r for r in rows if r["attack"] == "full"]
    labels = [f"{r['dataset']} / {r['learner']}" for r in full]
    ev = [100 * _f(r, "fraud_accepted_truthful") for r in full]
    re = [100 * _f(r, "fraud_accepted_attack") for r in full]
    wh = {f"{r['dataset']} / {r['learner']}": 100 * _f(r, "fraud_accepted_attack") for r in rows if r["attack"] == "withholding"}
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(full) + 1.2))
    y = np.arange(len(full))
    ax.hlines(y, ev, re, color=GRID, linewidth=3)
    ax.scatter(ev, y, color=SERIES[0], s=36, zorder=3, label="evaluated (truthful inputs)")
    ax.scatter([wh.get(l, np.nan) for l in labels], y, color=SERIES[2], s=36, zorder=3, label="realized, withholding only")
    ax.scatter(re, y, color=SERIES[1], s=36, zorder=3, label="realized, full attack")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("fraud accepted at 5% legitimate hold (%)")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    return [_save(fig, out, "fig_e1_evaluation_gap.png")]


def fig_splitting(out):
    rows = _load(out, "e3_splitting.csv")
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for j, rule in enumerate(dict.fromkeys(r["rule"] for r in rows)):
        sub = [r for r in rows if r["rule"] == rule]
        ax.plot([_f(r, "K") for r in sub], [_f(r, "max_exposure") for r in sub], marker="o", markersize=5,
                color=SERIES[j], label=rule.replace("_", " "))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("maximum parts per payment K")
    ax.set_ylabel("worst accepted total per attacker window")
    ax.legend(fontsize=8)
    return [_save(fig, out, "fig_e3_splitting_exposure.png")]


def fig_repeated(out):
    rows = _load(out, "e5_repeated_exact.csv")
    if not rows:
        return []
    pol = "P1_three_branch"
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for j, k in enumerate(["DET", "exp1.0_fresh", "exp1.0_sticky", "uniform_fresh"]):
        sub = [r for r in rows if r["policy"] == pol and r["kernel"] == k]
        if sub:
            ax.plot([_f(r, "n") for r in sub], [_f(r, "Lambda") for r in sub], marker="o", markersize=4,
                    color=SERIES[j], label=k.replace("_", " "))
    ax.set_xlabel("payments to the same receiver")
    ax.set_ylabel("selection leakage Λn (bits)")
    ax.set_title("three-branch receiver policy, empirical agent prior")
    ax.legend(fontsize=8)
    return [_save(fig, out, "fig_e5_repeated_leakage.png")]


def fig_agentic(out):
    rows = _load(out, "e4_agentic.csv")
    if not rows:
        return []
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    attacks = ["truthful", "withhold_split", "persuasive_text", "merchant_retry", "chain_strip", "full"]
    vals, errs = {}, {}
    for r in rows:
        m = "fraud_accepted_truthful" if r["attack"] == "truthful" else "fraud_accepted_attack"
        vals[(r["variant"], r["attack"])] = 100 * _f(r, m)
        errs[(r["variant"], r["attack"])] = 100 * _f(r, m + "_sd", 0)
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    _grouped_bars(ax, variants, attacks, vals, errs)
    ax.set_ylim(0, 100)
    ax.set_ylabel("injected-agent fraud accepted (%)")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    return [_save(fig, out, "fig_e4_agentic.png")]


def fig_rollout(out):
    rows = _load(out, "e9_rollout.csv")
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for j, v in enumerate(dict.fromkeys(r["variant"] for r in rows)):
        sub = sorted([r for r in rows if r["variant"] == v and r["attack"] == "withholding"],
                     key=lambda r: _f(r, "nonparticipating"))
        ax.plot([100 * _f(r, "nonparticipating") for r in sub], [100 * _f(r, "gap") for r in sub], marker="o",
                markersize=5, color=SERIES[j], label=v.replace("_", " "))
    ax.set_xlabel("sending banks issuing no claims (%)")
    ax.set_ylabel("gain from withholding (points)")
    ax.legend(fontsize=8)
    return [_save(fig, out, "fig_e9_rollout.png")]


def fig_comparison(out):
    rows = _load(out, "e10_comparison.csv")
    if not rows:
        return []
    full = [r for r in rows if r["attack"] == "full"]
    fig, ax = plt.subplots(figsize=(6.0, 0.42 * len(full) + 1.2))
    y = np.arange(len(full))
    ev = [100 * _f(r, "fraud_accepted_truthful") for r in full]
    re = [100 * _f(r, "fraud_accepted_attack") for r in full]
    ax.hlines(y, ev, re, color=GRID, linewidth=3)
    ax.scatter(ev, y, color=SERIES[0], s=36, zorder=3, label="truthful")
    ax.scatter(re, y, color=SERIES[1], s=36, zorder=3, label="full attack")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['design'].replace('_', ' ')} (held {100 * _f(r, 'legit_held'):.1f}%)" for r in full])
    ax.set_xlim(0, 100)
    ax.set_xlabel("fraud accepted (%)")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    return [_save(fig, out, "fig_e10_comparison.png")]


def fig_curves(out):
    rows = _load(out, "e12_operating_curves.csv")
    if not rows:
        return []
    paths = []
    for d in dict.fromkeys(r["dataset"] for r in rows):
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        sub = [r for r in rows if r["dataset"] == d and r["attack"] == "full"]
        for j, v in enumerate(dict.fromkeys(r["variant"] for r in sub)):
            s = sorted([r for r in sub if r["variant"] == v], key=lambda r: _f(r, "hold"))
            x = [100 * _f(r, "hold") for r in s]
            ax.plot(x, [100 * _f(r, "fraud_accepted_attack") for r in s], color=SERIES[j], label=v.replace("_", " "))
            ax.plot(x, [100 * _f(r, "fraud_accepted_truthful") for r in s], color=SERIES[j], linestyle=(0, (2, 2)),
                    linewidth=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("legitimate payments held (%)")
        ax.set_ylabel("fraud accepted (%), solid = full attack, dashed = truthful")
        ax.set_title(d.replace("_", "-"))
        ax.legend(fontsize=7)
        paths.append(_save(fig, out, f"fig_e12_curves_{d}.png"))
    return paths


def fig_scaling(out):
    rows = _load(out, "e13_repair_scaling.csv")
    if not rows:
        return []
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for j, (q, m, lab) in enumerate([("nothing_shown", "exact_ms_per_payment", "exact, nothing shown"),
                                     ("random", "exact_ms_per_payment", "exact, random payment"),
                                     ("nothing_shown", "box_ms_per_payment", "box, nothing shown")]):
        s = [r for r in rows if r["queries"] == q]
        ax.plot([_f(r, "lattice_points") for r in s], [_f(r, m) for r in s], marker="o", markersize=5,
                color=SERIES[j], label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("claim lattice points (4^n)")
    ax.set_ylabel("repair time per payment (ms)")
    ax.legend(fontsize=8)
    return [_save(fig, out, "fig_e13_repair_scaling.png")]


def make_all(out) -> list[str]:
    paths = []
    for fn in (fig_defences, fig_gap, fig_splitting, fig_repeated, fig_agentic, fig_rollout, fig_comparison,
               fig_curves, fig_scaling):
        try:
            paths += fn(out)
        except Exception as e:  # a missing column should not stop the remaining figures
            print(f"figure {fn.__name__} skipped: {e}")
    return paths
