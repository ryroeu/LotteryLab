"""On provably-fair synthetic data, every strategy must hug the chance line."""

from lotterylab import games
from lotterylab.backtest import backtest
from lotterylab.store import load_canonical
from lotterylab.strategy import BUILTIN_STRATEGIES, get_strategy
from lotterylab.synth import synth_history


def test_random_strategy_matches_baseline_on_fair_data():
    """Random play should match the exact baseline on synthetic fair data."""
    spec = games.get("eurodreams")  # highest match-3 rate -> tightest test
    hist = synth_history(spec, 4000, seed=7)
    res = backtest(get_strategy("random"), hist, spec, n_tickets=1, seed=2)
    # z within a few sigma; rate close to exact baseline
    assert abs(res.z_vs_baseline) < 4
    assert abs(res.three_plus_rate - res.baseline_three_plus_rate) < 0.01


def test_no_strategy_beats_chance_on_fair_data():
    """No built-in strategy should meaningfully beat fair synthetic draws."""
    spec = games.get("eurodreams")
    hist = synth_history(spec, 4000, seed=11)
    for name in BUILTIN_STRATEGIES:
        res = backtest(get_strategy(name), hist, spec, n_tickets=1, seed=3)
        # The whole thesis: nothing meaningfully exceeds the baseline.
        assert res.z_vs_baseline < 5, f"{name} suspiciously high: z={res.z_vs_baseline}"


def test_backtest_runs_on_real_history():
    """Real Powerball history should run through the backtest harness."""
    spec = games.get("powerball")
    hist = load_canonical("powerball")
    res = backtest(get_strategy("random"), hist, spec, seed=1)
    assert res.n_draws > 800  # ~999 draws since 2018, minus warmup
    assert res.spent > 0
