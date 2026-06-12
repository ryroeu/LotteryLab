"""Frequency analytics — fun to look at, ZERO predictive power.

Every output here carries that disclaimer, because that is the honest truth: a fair
draw has no memory, so 'hot' and 'cold' numbers are sampling noise. The built-in
chi-square uniformity test makes the point quantitatively — real histories do not
deviate from uniform beyond chance.
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
    counts = np.zeros(spec.main_max + 1, dtype=int)
    vals = history[main_columns(spec)].to_numpy().ravel()
    for v in vals:
        counts[int(v)] += 1
    return counts


def uniformity_test(history: pd.DataFrame, spec: GameSpec) -> dict:
    """Chi-square goodness-of-fit vs uniform over the main pool."""
    counts = frequency(history, spec)[1:]  # drop index 0
    n = counts.sum()
    expected = np.full(spec.main_max, n / spec.main_max)
    chi2, p = stats.chisquare(counts, expected)
    return {
        "n_balls_observed": int(n),
        "chi2": float(chi2),
        "dof": spec.main_max - 1,
        "p_value": float(p),
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
