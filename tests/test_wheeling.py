"""A wheel's guarantee is checked by brute force, not trusted."""

from itertools import combinations

import pytest

from lotterylab import games
from lotterylab.wheeling import covering_design, spread_numbers, wheel_report


@pytest.mark.parametrize("game,n", [("eurodreams", 8), ("eurodreams", 11), ("loto", 9)])
def test_every_triple_is_covered(game, n):
    spec = games.get(game)
    chosen = spread_numbers(n, 1, spec.main_max)
    report = wheel_report(spec, chosen)
    assert len(chosen) == n == len(set(chosen))
    for triple in combinations(chosen, 3):
        assert any(set(triple) <= set(ticket) for ticket in report.tickets)
    assert report.cost == report.n_tickets * spec.price
    assert 0 < report.p_condition < 1


def test_wheel_rejects_bad_input():
    spec = games.get("eurodreams")
    with pytest.raises(ValueError):
        wheel_report(spec, [1, 2, 3, 4, 5, 41])
    with pytest.raises(ValueError):
        covering_design([1, 2, 3], 6)
