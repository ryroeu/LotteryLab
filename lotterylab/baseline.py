"""The chance line — what any fair strategy must converge to.

Exact per-tier probabilities from combinatorics, plus expected tier counts for a
given number of draws/tickets. The backtest compares every strategy against this.
"""

from __future__ import annotations

from .combinatorics import match_main_at_least, tier_probability
from .games import GameSpec


def three_plus_main_probability(spec: GameSpec) -> float:
    """P(a single ticket matches >= 3 main balls) — the user's headline metric."""
    return match_main_at_least(spec, 3)


def expected_three_plus(spec: GameSpec, n_draws: int, n_tickets: int = 1) -> float:
    return three_plus_main_probability(spec) * n_draws * n_tickets


def all_tier_probabilities(spec: GameSpec) -> dict[tuple[int, int], float]:
    out = {}
    for m in range(spec.main_count + 1):
        for s in range(spec.special_count + 1):
            p = tier_probability(spec, m, s)
            if p > 0:
                out[(m, s)] = p
    return out
