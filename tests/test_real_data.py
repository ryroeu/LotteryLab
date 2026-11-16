"""Checks against the committed snapshots (skipped where a game has none)."""

import numpy as np
import pytest

from lotterylab import ev, games, payouts, popularity, store
from lotterylab.sharing import sharing_backtest

FDJ = list(store.FDJ_SOURCES)


def _has(game):
    return bool(store.all_snapshots(game))


@pytest.mark.parametrize("game", games.GAMES)
def test_history_loads_and_is_ordered(game):
    if not _has(game):
        pytest.skip("no snapshot on disk")
    spec = games.get(game)
    frame = store.load_canonical(game)
    assert len(frame) > 50
    assert frame["date"].is_monotonic_increasing and frame["date"].is_unique
    assert frame["date"].iloc[0] >= spec.matrix_since
    assert (
        frame[[f"m{i}" for i in range(1, spec.main_count + 1)]].to_numpy().max()
        <= spec.main_max
    )


@pytest.mark.parametrize("game", FDJ)
def test_operator_records_agree_with_the_rules(game):
    frame = payouts.try_load(game)
    if frame is None:
        pytest.skip("no payout snapshot on disk")
    spec = games.get(game)
    summary = payouts.rank_summary(spec, frame).set_index("rank")
    for rank, row in summary.iterrows():
        if rank in spec.static_ranks:
            continue
        shared = rank in spec.pari_ranks
        assert (row["kind"] == "shared") == shared, f"rank {rank} is {row['kind']}"
    lowest = summary.iloc[-1]
    if lowest["kind"] == "fixed":
        assert lowest["mean_payout"] == pytest.approx(spec.price)
    # The static fallback table should stay close to what is really paid.
    measured = payouts.empirical_prize_table(spec, frame)
    for tier, amount in spec.prize_table.items():
        assert measured[tier] == pytest.approx(amount, rel=0.15), tier
    assert 1e5 < payouts.typical_tickets(spec, frame) < 1e8


@pytest.mark.parametrize("game", FDJ)
def test_players_really_do_favour_birthday_numbers(game):
    if payouts.try_load(game) is None:
        pytest.skip("no payout snapshot on disk")
    model = popularity.load_model(game)
    weights = np.array(model.weights)
    assert model.source == "fitted" and model.skill > 0.3
    assert weights[:31].mean() > weights[31:].mean()
    mkt = ev.market(game)
    least, most = ev.best_and_worst(mkt)
    assert ev.evaluate(least, mkt).ev_per_ticket > ev.evaluate(most, mkt).ev_per_ticket


@pytest.mark.parametrize("game", ["loto", "euromillions"])
def test_sharing_effect_holds_out_of_sample(game):
    if payouts.try_load(game) is None:
        pytest.skip("no payout snapshot on disk")
    result = sharing_backtest(game, n_tickets=2000)
    least, everyone, most = (group.realised for group in result.groups)
    assert least > everyone > most
    assert everyone == pytest.approx(1.0, abs=0.02)
    assert least > 1.04 and most < 0.97
