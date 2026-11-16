"""Exact odds must reproduce the operators' published figures."""

import pytest

from lotterylab import games
from lotterylab.baseline import all_tier_probabilities
from lotterylab.combinatorics import (
    jackpot_odds,
    joint_match_main_at_least,
    match_main_at_least,
    match_main_exactly,
    median_wait_draws,
)

PUBLISHED_JACKPOT = {
    "powerball": 292_201_338,
    "megamillions": 290_472_336,
    "euromillions": 139_838_160,
    "eurodreams": 19_191_900,
    "loto": 19_068_840,
}
PUBLISHED_MATCH3 = {
    "powerball": 557,
    "euromillions": 214,
    "eurodreams": 32,
    "loto": 202,
}


@pytest.mark.parametrize("game,odds", PUBLISHED_JACKPOT.items())
def test_jackpot_odds_match_published(game, odds):
    assert jackpot_odds(games.get(game)) == odds


@pytest.mark.parametrize("game,one_in", PUBLISHED_MATCH3.items())
def test_match3_odds(game, one_in):
    assert round(1 / match_main_exactly(games.get(game), 3)) == one_in


@pytest.mark.parametrize("game", games.GAMES)
def test_tier_probabilities_sum_to_one(game):
    assert sum(all_tier_probabilities(games.get(game)).values()) == pytest.approx(1.0)


@pytest.mark.parametrize("game", games.GAMES)
def test_joint_probability_limits(game):
    spec = games.get(game)
    p = match_main_at_least(spec, 3)
    # Identical tickets win together; the joint of a ticket with itself is p.
    assert joint_match_main_at_least(spec, spec.main_count, 3) == pytest.approx(p)
    # Disjoint tickets are negatively correlated: both winning is rarer than p^2.
    assert joint_match_main_at_least(spec, 0, 3) < p * p


def test_median_wait_is_about_three_months_for_eurodreams():
    spec = games.get("eurodreams")
    draws = median_wait_draws(match_main_exactly(spec, 3))
    assert 20 <= draws <= 24  # two draws a week -> ~11 weeks
