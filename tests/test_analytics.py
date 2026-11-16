"""The uniformity test must be calibrated on data known to be fair."""

import numpy as np
import pytest

from lotterylab import analytics, games
from lotterylab.synth import synth_history


@pytest.mark.parametrize("game", ["eurodreams", "loto"])
def test_chi_square_is_calibrated_on_fair_draws(game):
    spec = games.get(game)
    results = [
        analytics.uniformity_test(synth_history(spec, 300, seed=seed), spec)
        for seed in range(400)
    ]
    corrected = np.array([r["chi2"] for r in results])
    raw = np.array([r["chi2_uncorrected"] for r in results])
    pool, k = spec.main_max, spec.main_count
    # Without replacement the raw statistic averages N-k; corrected, N-1.
    assert raw.mean() == pytest.approx(pool - k, rel=0.04)
    assert corrected.mean() == pytest.approx(pool - 1, rel=0.04)
    false_alarms = np.mean([r["p_value"] < 0.05 for r in results])
    assert 0.02 <= false_alarms <= 0.09  # nominal 5%


def test_chi_square_detects_a_biased_machine():
    spec = games.get("eurodreams")
    history = synth_history(spec, 2000, seed=1).copy()
    history["m1"] = 1  # number 1 comes up every draw
    # (rows may now repeat 1; the count vector is what the test sees)
    assert analytics.uniformity_test(history, spec)["p_value"] < 1e-6


def test_frequency_counts_every_ball():
    spec = games.get("loto")
    history = synth_history(spec, 250, seed=3)
    counts = analytics.frequency(history, spec)
    assert counts[0] == 0 and counts.sum() == 250 * spec.main_count
