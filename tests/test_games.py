"""The registry is internally consistent and matches the published rules."""

import pytest

from lotterylab import games

FDJ = ["euromillions", "eurodreams", "loto"]


@pytest.mark.parametrize("game", games.GAMES)
def test_prize_tiers_are_reachable(game):
    spec = games.get(game)
    for main, special in [*spec.prize_table, spec.jackpot_tier]:
        assert 0 <= main <= spec.main_count
        assert 0 <= special <= spec.special_count
    assert spec.jackpot_tier == (spec.main_count, spec.special_count)
    assert spec.jackpot_tier not in spec.prize_table


@pytest.mark.parametrize("game", FDJ)
def test_ranks_partition_the_prize_table(game):
    spec = games.get(game)
    tiers = [tier for rank in spec.rank_tiers for tier in rank]
    assert len(tiers) == len(set(tiers)), "a tier belongs to exactly one rank"
    assert spec.rank_tiers[0] == (spec.jackpot_tier,)
    assert set(tiers) == set(spec.prize_table) | {spec.jackpot_tier}
    assert spec.rank_of(spec.jackpot_tier) == 1
    assert spec.rank_of((0, 0)) is None
    assert spec.payout_since >= spec.ranks_since >= spec.matrix_since


def test_eurodreams_follows_the_published_structure():
    spec = games.get("eurodreams")
    # Six mains without the Dream Number: EUR 2,000 a month for five years.
    assert spec.prize_table[(6, 0)] == 120_000
    # The Dream Number changes nothing below six mains...
    for main in (2, 3, 4, 5):
        assert spec.prize_table[(main, 0)] == spec.prize_table[(main, 1)]
    # ...and two mains refund the ticket.
    assert spec.prize_table[(2, 0)] == spec.price
    assert (1, 1) not in spec.prize_table
    assert not spec.jackpot_rolls


def test_loto_chance_number_refunds_the_ticket():
    spec = games.get("loto")
    assert spec.prize_table[(0, 1)] == spec.prize_table[(1, 1)] == spec.price
    assert (0, 1) not in spec.pari_tiers and (2, 0) in spec.pari_tiers
