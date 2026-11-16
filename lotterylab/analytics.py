"""Frequency analytics — fun to look at, ZERO predictive power.

Every output here carries that disclaimer, because that is the honest truth: a fair
draw has no memory, so 'hot' and 'cold' numbers are sampling noise. The built-in
chi-square uniformity test makes the point quantitatively — real histories do not
deviate from uniform beyond chance.

The test is corrected for how lotteries draw. A textbook chi-square assumes every
ball is an independent pick, but the k balls of one draw are taken WITHOUT
replacement, so a number can appear at most once per draw and the counts are less
spread out than the textbook expects: the raw statistic averages N-k, not N-1.
Scaling it by (N-1)/(N-k) restores the chi-square(N-1) reference (Joe, 1993).
Uncorrected, the test is too lenient — on fair 6/40 data it raises a false alarm
about 1% of the time instead of the nominal 5%, and would miss real bias as easily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .games import GameSpec
from .schema import main_columns

DISCLAIMER = "NOTE: historical only — zero predictive power. A fair draw has no memory."


def frequency(history: pd.DataFrame, spec: GameSpec) -> np.ndarray:
    """Count how often each main number appears in the draw history."""
    vals = history[main_columns(spec)].to_numpy(dtype=int).ravel()
    return np.bincount(vals, minlength=spec.main_max + 1)


def uniformity_test(history: pd.DataFrame, spec: GameSpec) -> dict:
    """Chi-square goodness-of-fit vs uniform over the main pool, corrected for
    drawing without replacement within each draw (see the module docstring)."""
    counts = frequency(history, spec)[1:]  # drop index 0
    n = counts.sum()
    if n == 0:
        raise ValueError("uniformity_test requires at least one observed ball")
    pool, per_draw = spec.main_max, spec.main_count
    expected = n / pool
    raw = float(((counts - expected) ** 2 / expected).sum())
    chi2 = raw * (pool - 1) / (pool - per_draw)
    p = float(stats.chi2.sf(chi2, pool - 1))
    return {
        "n_balls_observed": int(n),
        "chi2": chi2,
        "chi2_uncorrected": raw,
        "dof": pool - 1,
        "p_value": p,
        "verdict": (
            "consistent with uniform (as expected)"
            if p > 0.05
            else "deviates — check for bias or small sample"
        ),
    }


def ascii_bars(history: pd.DataFrame, spec: GameSpec, width: int = 40) -> str:
    """Render a terminal-friendly frequency chart plus uniformity test."""
    counts = frequency(history, spec)[1:]
    if counts.max() == 0:
        return "(no data)"
    lines = [DISCLAIMER, ""]
    expected = counts.sum() / spec.main_max
    for i, c in enumerate(counts, start=1):
        bar_text = "#" * int(round(width * c / counts.max()))
        lines.append(f"  {i:2d} | {bar_text:<{width}} {c}")
    lines.append("")
    lines.append(f"  expected per number if uniform: {expected:.1f}")
    test = uniformity_test(history, spec)
    lines.append(
        f"  chi-square uniformity: chi2={test['chi2']:.1f} "
        f"dof={test['dof']} p={test['p_value']:.3f} -> {test['verdict']}"
    )
    return "\n".join(lines)
