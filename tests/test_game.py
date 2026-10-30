import numpy as np

from vista2 import game


def _random_space(rng, n):
    return game.transitive_closure(rng.random((n, n)) < 0.25)


def test_closure_is_preorder(rng):
    for _ in range(50):
        assert game.is_preorder(_random_space(rng, int(rng.integers(2, 9))))


def test_theorem1_items(rng):
    for _ in range(200):
        n = int(rng.integers(2, 8))
        R = _random_space(rng, n)
        d = rng.integers(0, 3, n)
        ds = game.realized(d, R)
        assert (ds >= d).all()
        assert game.is_monotone(ds, R)
        assert game.is_sp(d, R) == bool((ds == d).all())


def test_repair_is_largest_sp_rule_below(rng):
    for _ in range(60):
        n = int(rng.integers(2, 6))
        R = _random_space(rng, n)
        d = rng.integers(0, 3, n)
        rep = game.repair(d, R)
        assert game.is_sp(rep, R) and (rep <= d).all()
        for g in game.all_sp_rules_below(d, R, 3):
            assert (g <= rep).all()


def test_corollary1_hour_must_be_ignored():
    # W chain(2) x F complete(2): a rule that prefers hour 1 is manipulable
    R = game.product_reach([game.chain_down(2), game.complete(2)])
    d = np.array([0, 1, 1, 1])  # states (w, f) in order (0,0),(0,1),(1,0),(1,1)
    assert not game.is_sp(d, R)
    assert game.is_sp(np.array([0, 0, 1, 1]), R)


def test_splitting_impossibility_and_bound():
    tot, k = game.split_exposure_stateless(lambda a: a <= 100, 1000, 10)
    assert tot == 1000 and k == 10
    assert game.split_exposure_cumulative(500, [100] * 20) == 500
    assert game.split_exposure_cumulative(500, [300, 300, 100], prior_spend=50) == 400
