"""One-command reproduction: python -m vista2.run_all --config configs/full.json

Runs the experiments listed in the config, writes CSVs under results/<run-name>/, regenerates figures
(results/<run-name>/figures/) and a summary (results/<run-name>/SUMMARY.md), and records a manifest with
the config, library versions, hardware, per-step timings and SHA-256 hashes of every output.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback

from .util import sha256_file

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

STEPS = {
    "e0": ("e0_theory", "theory checks (Theorems 1-5, Corollary 1, Algorithms 2b and 3)"),
    "e1": ("e1_gap", "evaluation gap on real data"),
    "e2": ("e2_defences", "cost of each defence, operating curves"),
    "e3": ("e3_splitting", "splitting and the cumulative claim (PaySim)"),
    "e4": ("e4_agentic", "agentic adversaries, Gate B"),
    "e5": ("e5_repeated", "repeated agent payments"),
    "e6": ("e6_overhead", "payment-path overhead"),
    "e7": ("e7_binding", "binding-layer attack suite"),
    "e8": ("e8_sensitivity", "typing sensitivity"),
    "e9": ("e9_rollout", "rollout-stage sweep"),
    "e10": ("e10_comparison", "measured comparison of designs"),
    "e11": ("e11_attacker", "attacker strength and transfer"),
    "e13": ("e13_repair_scaling", "repair cost against lattice size"),
    "e14": ("e14_bounded", "bounded attacker"),
}


def versions() -> dict:
    import cryptography
    import lightgbm
    import numpy
    import pandas
    import scipy
    import sklearn
    import xgboost
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__,
            "pandas": pandas.__version__, "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
            "xgboost": xgboost.__version__, "cryptography": cryptography.__version__,
            "platform": platform.platform(), "machine": platform.machine(), "cpus": os.cpu_count()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "smoke.json"))
    ap.add_argument("--only", default="", help="comma-separated step names, e.g. e0,e3")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)
    with open(args.config) as fh:
        cfg = json.load(fh)
    name = cfg.get("name") or os.path.splitext(os.path.basename(args.config))[0]
    out = os.path.join(ROOT, "results", name)
    os.makedirs(out, exist_ok=True)
    steps = [s.strip() for s in args.only.split(",") if s.strip()] or cfg.get("steps", list(STEPS))
    log_path = os.path.join(out, "run.log")
    logf = open(log_path, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    man_path = os.path.join(out, "manifest.json")
    manifest = {"name": name, "config": cfg, "versions": versions(), "steps": []}
    if os.path.exists(man_path) and args.only:
        with open(man_path) as fh:
            old = json.load(fh)
        manifest["steps"] = [s for s in old.get("steps", []) if s["step"] not in steps]
    log(f"run {name}: steps {', '.join(steps)}")
    import importlib
    for st in steps:
        mod_name, desc = STEPS[st]
        log(f"{st}: {desc}")
        t0 = time.time()
        rec = {"step": st, "description": desc}
        try:
            mod = importlib.import_module(f"vista2.experiments.{mod_name}")
            files = mod.run(cfg, out, log) if "log" in mod.run.__code__.co_varnames else mod.run(cfg, out)
            rec.update(status="ok", files={os.path.basename(f): sha256_file(f) for f in files})
        except Exception as e:  # keep going so one failing step does not lose the rest of the run
            rec.update(status="failed", error=f"{type(e).__name__}: {e}")
            log(traceback.format_exc())
        rec["seconds"] = round(time.time() - t0, 1)
        log(f"{st}: {rec['status']} in {rec['seconds']} s")
        manifest["steps"].append(rec)
        with open(man_path, "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)
    if not args.no_report:
        from . import figures, report
        figs = figures.make_all(out)
        log(f"figures: {len(figs)} written")
        report.write(out, manifest)
        log("summary: SUMMARY.md written")
    logf.close()


if __name__ == "__main__":
    main()
