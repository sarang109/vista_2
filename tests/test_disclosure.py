import numpy as np

from vista2.disclosure import repeated, space


def _toy():
    states = np.array([(a, b) for a in range(3) for b in range(3)])
    p = np.full(len(states), 1 / len(states))
    front = space.frontier([(2, 0), (0, 2), (2, 2)])
    return space.build(states, p, front)[0]


def test_frontier_drops_dominated_branches():
    assert len(space.frontier([(2, 0), (0, 2), (2, 2)])) == 2


def test_two_branch_obstruction():
    ss = _toy()
    assert ss.two_branch_obstruction() is not None
    assert not ss.zero_leakage_system()["feasible"]


def test_sticky_keeps_single_payment_leakage():
    ss = _toy()
    Q = ss.kernel_exp(1.0)
    one = repeated.repeated_metrics(ss, Q, 1)
    assert abs(repeated.sticky_metrics(ss, Q, 8)["Lambda"] - one["Lambda"]) < 1e-12
    assert repeated.repeated_metrics(ss, Q, 4)["Lambda"] > one["Lambda"]
