import numpy as np
import pandas as pd

from vista2.attack import FULL, WITHHOLD_ONLY, best_response
from vista2.control import FeatureSpec, TypedTable, variants
from vista2.datasets import synthetic
from vista2.experiments.common import fit_variant
from vista2.repair import ExactRepair, TreeBoxRepair, lattice_dp, lattice_dp_reference


def test_lattice_dp_matches_recursion(rng):
    for _ in range(20):
        s = rng.random(tuple(rng.integers(2, 4, 3)))
        assert np.allclose(lattice_dp(s), lattice_dp_reference(s))


def _small():
    t = synthetic.generate(n=20_000, seed=3)
    v = {x.name: x for x in variants(t)}
    return t, v


def test_box_repair_bounds_exact_and_matches_model_at_a_point():
    t, v = _small()
    tr = fit_variant(t, v["unconstrained"], "lightgbm", seed=0, rounds=40)
    X = t.X.iloc[:80]
    assert np.allclose(TreeBoxRepair(tr.model, t, tr.columns, ()).raw(X), tr.model.raw(X), atol=1e-6)
    ex = ExactRepair(tr.model.risk, t, tr.columns, ("W",)).risk(X)
    bx = TreeBoxRepair(tr.model, t, tr.columns, ("W",)).risk(X)
    assert (bx >= ex - 1e-9).all()
    assert (ex >= tr.model.risk(X) - 1e-9).all()


def test_attack_never_worse_than_truth_and_respects_withholding_order():
    t, v = _small()
    tr = fit_variant(t, v["naive"], "lightgbm", seed=0, rounds=40)
    fr = t.X[t.y == 1].iloc[:150]
    res = best_response(tr.risk, fr, t, tr.columns, WITHHOLD_ONLY)
    assert (res.risk <= res.truthful_risk + 1e-12).all()
    for c in t.cols("W"):
        assert (res.X[c].to_numpy() <= fr[c].to_numpy()).all()


def test_control_typed_model_gains_nothing():
    t, v = _small()
    tr = fit_variant(t, v["control_typed"], "lightgbm", seed=0, rounds=40)
    fr = t.X[t.y == 1].iloc[:150]
    res = best_response(tr.risk, fr, t, tr.columns, FULL)
    assert np.allclose(res.risk, res.truthful_risk)


def test_poset_withholding():
    X = pd.DataFrame({"m": [0.0, 1.0, 2.0], "t": [0.0, 0.0, 0.0]})
    specs = {"m": FeatureSpec("m", "W", "poset"), "t": FeatureSpec("t", "T")}
    tt = TypedTable("toy", X, np.array([1, 1, 1]), np.zeros(3, dtype=np.int8), specs)
    risk = lambda D: np.where(D["m"].to_numpy() == 1.0, 0.9, np.where(D["m"].to_numpy() == 2.0, 0.1, 0.5))  # noqa: E731
    res = best_response(risk, X, tt, ["m", "t"], WITHHOLD_ONLY)
    # mismatch (1) can only be hidden (0), never turned into a match (2)
    assert list(res.X["m"]) == [0.0, 0.0, 2.0]
