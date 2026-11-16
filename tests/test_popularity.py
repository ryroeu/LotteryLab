"""The popularity fit must recover a crowd we invented, and conserve money."""

import numpy as np
import pytest

from lotterylab import games, payouts, popularity

from .conftest import make_payout_frame


def crowd(spec, strength=0.35, seed=0):
    """Weights favouring low 'birthday' numbers plus some idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    numbers = np.arange(1, spec.main_max + 1)
    logw = np.where(numbers <= 31, strength / 2, -strength / 2)
    logw = logw + rng.normal(0, 0.08, spec.main_max)
    return np.exp(logw - logw.mean())


@pytest.mark.parametrize("game", ["loto", "eurodreams", "euromillions"])
def test_fit_recovers_known_weights(game):
    spec = games.get(game)
    truth = crowd(spec)
    frame = make_payout_frame(spec, truth, 500, seed=1)
    model = popularity.fit(spec, frame)
    fitted = np.array(model.weights)
    assert model.source == "fitted" and fitted.mean() == pytest.approx(1.0)
    assert np.corrcoef(np.log(fitted), np.log(truth))[0, 1] > 0.9
    assert model.skill is not None and model.skill > 0.3
    # Shared ranks must show a positive payout response to popularity.
    assert all(value > 0 for _, value in model.elasticity)


def test_fit_finds_nothing_in_a_uniform_crowd():
    spec = games.get("loto")
    frame = make_payout_frame(spec, np.ones(spec.main_max), 400, seed=2)
    weights = np.array(popularity.fit(spec, frame).weights)
    assert weights.min() > 0.93 and weights.max() < 1.07


@pytest.mark.parametrize("game", ["loto", "eurodreams"])
def test_sharing_only_moves_money_between_tickets(game):
    spec = games.get(game)
    frame = make_payout_frame(spec, crowd(spec), 300, seed=3)
    model = popularity.fit(spec, frame)
    rng = np.random.default_rng(0)
    for hits, rank in [
        (2, None),
        (3, None),
        (3, dict(model.elasticity) and model.elasticity[-2][0]),
    ]:
        values = [
            popularity.share_multiplier(
                model,
                tuple(
                    rng.choice(
                        np.arange(1, spec.main_max + 1), spec.main_count, replace=False
                    )
                ),
                spec,
                hits,
                rank,
            )
            for _ in range(3000)
        ]
        assert np.mean(values) == pytest.approx(1.0, abs=0.01)
    combos = [
        popularity.combo_popularity(
            model,
            tuple(
                rng.choice(
                    np.arange(1, spec.main_max + 1), spec.main_count, replace=False
                )
            ),
            spec,
        )
        for _ in range(4000)
    ]
    assert np.mean(combos) == pytest.approx(1.0, abs=0.03)


def test_uniform_model_changes_nothing():
    spec = games.get("euromillions")
    model = popularity.uniform_model(spec)
    assert popularity.combo_popularity(model, (1, 2, 3, 4, 5), spec) == pytest.approx(1)
    assert popularity.share_multiplier(
        model, (1, 2, 3, 4, 5), spec, 3
    ) == pytest.approx(1)


def test_unpopular_numbers_are_worth_more():
    spec = games.get("loto")
    model = popularity.fit(spec, make_payout_frame(spec, crowd(spec), 400, seed=4))
    high, low = (40, 43, 45, 47, 49), (3, 7, 11, 19, 27)
    assert (
        popularity.combo_popularity(model, high, spec)
        < 1
        < popularity.combo_popularity(model, low, spec)
    )
    for hits in (2, 3, 4):
        assert popularity.share_multiplier(model, high, spec, hits) > 1
        assert popularity.share_multiplier(model, low, spec, hits) < 1


def test_comparable_ranks_share_one_special_condition():
    assert popularity.comparable_ranks(games.get("loto")) == [
        (2, 5),
        (4, 4),
        (6, 3),
        (8, 2),
    ]
    assert popularity.comparable_ranks(games.get("eurodreams")) == [
        (3, 5),
        (4, 4),
        (5, 3),
        (6, 2),
    ]
    assert popularity.comparable_ranks(games.get("powerball")) == []


def test_payout_summary_and_ticket_estimate():
    spec = games.get("loto")
    frame = make_payout_frame(spec, crowd(spec), 200, seed=5, tickets=4e6)
    summary = payouts.rank_summary(spec, frame).set_index("rank")
    assert summary.loc[9, "kind"] == "fixed" and summary.loc[1, "kind"] == "top"
    assert summary.loc[8, "kind"] == "shared"
    # Which numbers came up nudges each draw's estimate; the median stays close.
    assert payouts.typical_tickets(spec, frame) == pytest.approx(4e6, rel=0.05)
    table = payouts.empirical_prize_table(spec, frame)
    assert table[(2, 0)] == pytest.approx(summary.loc[8, "mean_payout"], abs=0.01)
    assert table[(0, 1)] == spec.price
