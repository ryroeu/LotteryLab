"""Expected value: exact where it can be, and honest about what moves it."""

from dataclasses import replace
from math import exp

import numpy as np
import pytest

from lotterylab import ev, games, popularity
from lotterylab.combinatorics import jackpot_probability, tier_probability

from .conftest import make_payout_frame


def test_jackpot_share_matches_direct_summation():
    for lam in (0.01, 0.5, 3.0):
        term, total = exp(-lam), 0.0
        for b in range(200):
            total += term / (1 + b)
            term *= lam / (b + 1)
        assert ev.jackpot_share(lam) == pytest.approx(total)
    assert ev.jackpot_share(0) == 1.0
    # With a cap of three full winners, sharing barely bites at small lambda.
    assert ev.jackpot_share(0.5, cap=3) > ev.jackpot_share(0.5, cap=1)
    assert ev.jackpot_share(0.5, cap=3) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("game", games.GAMES)
def test_uniform_crowd_gives_the_textbook_ev(game):
    spec = games.get(game)
    main = tuple(range(1, spec.main_count + 1))
    report = ev.ticket_ev(main, spec, n_players=0, model=popularity.uniform_model(spec))
    textbook = (
        sum(
            tier_probability(spec, m, s) * amount
            for (m, s), amount in spec.prize_table.items()
        )
        + jackpot_probability(spec) * spec.jackpot_estimate
    )
    assert report.ev_per_ticket == pytest.approx(textbook)
    assert report.shared_multiplier == pytest.approx(1.0)
    assert 0 < report.return_rate < 1


def measured_market(game="loto"):
    spec = games.get(game)
    numbers = np.arange(1, spec.main_max + 1)
    truth = np.exp(np.where(numbers <= 31, 0.2, -0.2))
    frame = make_payout_frame(spec, truth / truth.mean(), 400, seed=6)
    return ev.Market(
        spec=spec, model=popularity.fit(spec, frame), frame=frame, raffle=0.05
    )


def test_neglected_numbers_have_higher_ev_but_identical_odds():
    mkt = measured_market()
    least, most = ev.best_and_worst(mkt)
    assert min(least) > 31 > max(most) - 1
    good, bad = ev.evaluate(least, mkt), ev.evaluate(most, mkt)
    assert good.ev_per_ticket > bad.ev_per_ticket
    assert good.ev_fixed == bad.ev_fixed and good.ev_raffle == bad.ev_raffle == 0.05
    assert good.expected_jackpot_share > bad.expected_jackpot_share
    assert good.shared_multiplier > 1 > bad.shared_multiplier


def test_bigger_jackpots_help_less_than_linearly():
    mkt = measured_market()
    least, _ = ev.best_and_worst(mkt)
    small = ev.evaluate(least, mkt, jackpot=2e6)
    big = ev.evaluate(least, mkt, jackpot=20e6)
    assert big.ev_jackpot > small.ev_jackpot
    assert big.ev_jackpot < 10 * small.ev_jackpot * 1.0001  # never more than linear
    even = ev.breakeven_jackpot(least, mkt)
    assert even is None or ev.evaluate(
        least, mkt, jackpot=even
    ).return_rate == pytest.approx(1, abs=1e-3)


def test_fixed_top_prize_has_no_break_even():
    spec = games.get("eurodreams")
    mkt = ev.Market(
        spec=spec, model=popularity.uniform_model(spec), frame=None, raffle=0.0
    )
    assert ev.breakeven_jackpot((1, 2, 3, 4, 5, 6), mkt) is None
    assert mkt.tickets() == ev.DEFAULT_PLAYERS and not mkt.measured


def test_rejects_nonsense():
    spec = games.get("loto")
    with pytest.raises(ValueError):
        ev.ticket_ev((1, 2, 3, 4, 5), spec, jackpot=-1)
    with pytest.raises(ValueError):
        ev.ticket_ev((1, 2, 3, 4, 5), replace(spec), n_players=float("nan"))
    with pytest.raises(ValueError):
        ev.ticket_ev((1, 2, 3, 4, 99), spec)
