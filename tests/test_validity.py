"""No strategy and no parsed draw may ever emit an impossible ticket."""

import numpy as np

from lotterylab import games
from lotterylab.strategy import OrderStatMean
from lotterylab.strategy import BUILTIN_STRATEGIES, get_strategy
from lotterylab.synth import synth_history
from lotterylab.validate import is_valid_ticket, validate_ticket


def test_valid_and_invalid_tickets():
    """Accept valid tickets and reject malformed tickets."""
    spec = games.get("powerball")
    validate_ticket((1, 2, 3, 4, 5), (10,), spec)  # ok
    assert not is_valid_ticket((1, 2, 3, 4, 4), (10,), spec)  # dup main
    assert not is_valid_ticket((1, 2, 3, 4, 70), (10,), spec)  # out of range
    assert not is_valid_ticket((1, 2, 3, 4, 5), (27,), spec)  # special out of range
    assert not is_valid_ticket((1, 2, 3, 4), (10,), spec)  # wrong count


def test_every_strategy_emits_valid_tickets():
    """Every built-in strategy should emit playable tickets for every game."""
    rng = np.random.default_rng(0)
    for spec in games.all_games():
        hist = synth_history(spec, 120, seed=3)
        for name in BUILTIN_STRATEGIES:
            tickets = get_strategy(name).generate(hist, spec, 5, rng)
            for main, special in tickets:
                # raises if invalid — including the order-stat-mean dedupe path
                validate_ticket(main, special, spec)


def test_order_stat_mean_is_valid_despite_collisions():
    """Rounded order-statistic means should repair duplicate picks."""
    # EuroMillions order-stat means can round to collisions; dedupe must repair them.
    rng = np.random.default_rng(0)
    spec = games.get("euromillions")
    hist = synth_history(spec, 60, seed=1)
    (main, special) = OrderStatMean().generate(hist, spec, 1, rng)[0]
    validate_ticket(main, special, spec)
