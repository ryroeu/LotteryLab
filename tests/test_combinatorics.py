"""The odds are exact — assert they reproduce the published figures."""

from lotterylab import games
from lotterylab.combinatorics import (
    jackpot_odds,
    match_main_exactly,
    order_statistic_means,
    tier_probability,
)


def one_in(p):
    """Convert a probability into rounded one-in-N odds."""
    return round(1 / p)


def test_match3_odds():
    """Match-3 odds should match published values."""
    assert one_in(match_main_exactly(games.get("powerball"), 3)) == 557
    assert one_in(match_main_exactly(games.get("megamillions"), 3)) == 582
    assert one_in(match_main_exactly(games.get("euromillions"), 3)) == 214
    assert one_in(match_main_exactly(games.get("eurodreams"), 3)) == 32


def test_jackpot_odds():
    """Jackpot odds should match published values."""
    assert jackpot_odds(games.get("powerball")) == 292_201_338
    assert jackpot_odds(games.get("megamillions")) == 290_472_336
    assert jackpot_odds(games.get("euromillions")) == 139_838_160
    assert jackpot_odds(games.get("eurodreams")) == 19_191_900


def test_tier_probabilities_sum_to_one():
    """Every game's full tier distribution should sum to one."""
    for spec in games.all_games():
        total = sum(
            tier_probability(spec, m, s)
            for m in range(spec.main_count + 1)
            for s in range(spec.special_count + 1)
        )
        assert abs(total - 1.0) < 1e-9


def test_order_statistic_means_powerball():
    """Powerball order-statistic means should match the closed form."""
    means = order_statistic_means(69, 5)
    assert [round(x, 1) for x in means] == [11.7, 23.3, 35.0, 46.7, 58.3]
