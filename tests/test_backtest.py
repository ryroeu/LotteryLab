"""The harness: no lookahead, valid tickets, and fair data lands on chance."""

import numpy as np
import pytest

from lotterylab import games
from lotterylab.backtest import backtest
from lotterylab.strategy import (
    BUILTIN_STRATEGIES,
    Past,
    Strategy,
    as_past,
    get_strategy,
)
from lotterylab.synth import synth_history
from lotterylab.validate import validate_ticket


@pytest.mark.parametrize("name", BUILTIN_STRATEGIES)
def test_no_strategy_beats_chance_on_fair_draws(name):
    spec = games.get("eurodreams")
    history = synth_history(spec, 6000, seed=11)
    result = backtest(get_strategy(name), history, spec, n_tickets=2, seed=5)
    assert abs(result.z_vs_baseline) < 3.5
    assert result.n_draws == 6000 - 50
    assert sum(result.tier_hits.values()) == result.n_draws * 2


class Spy(Strategy):
    """Records what it was shown, and plays random tickets."""

    name = "spy"

    def __init__(self):
        self.seen = []

    def generate(self, past, spec, n_tickets, rng):
        self.seen.append(
            (past.n_draws, int(past.main_counts.sum()), past.last_main, len(past.frame))
        )
        return get_strategy("random").generate(past, spec, n_tickets, rng)


def test_strategies_only_ever_see_the_past():
    spec = games.get("loto")
    history = synth_history(spec, 120, seed=2)
    spy = Spy()
    backtest(spy, history, spec, warmup=20)
    mains = history[[f"m{i}" for i in range(1, 6)]].to_numpy()
    for step, (n_draws, balls, last_main, frame_len) in enumerate(spy.seen):
        t = 20 + step
        assert n_draws == t == frame_len
        assert balls == t * spec.main_count
        assert last_main == tuple(sorted(mains[t - 1].tolist()))


@pytest.mark.parametrize("game", games.GAMES)
@pytest.mark.parametrize("name", BUILTIN_STRATEGIES)
def test_builtin_strategies_emit_valid_tickets(game, name):
    spec = games.get(game)
    rng = np.random.default_rng(0)
    for history in (synth_history(spec, 0), synth_history(spec, 40, seed=4)):
        tickets = get_strategy(name).generate(history, spec, 4, rng)
        assert len(tickets) == 4
        for main, special in tickets:
            validate_ticket(main, special, spec)


def test_hot_and_cold_deal_different_tickets_down_the_ranking():
    spec = games.get("eurodreams")
    past = as_past(synth_history(spec, 300, seed=9), spec)
    assert isinstance(past, Past)
    rng = np.random.default_rng(0)
    hot = [main for main, _ in get_strategy("hot").generate(past, spec, 3, rng)]
    assert len(set(hot)) == 3 and not set(hot[0]) & set(hot[1])
    counts = past.main_counts
    assert min(counts[list(hot[0])]) >= max(counts[list(hot[2])])
