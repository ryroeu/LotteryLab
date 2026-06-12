"""Exact lottery probabilities — hypergeometric, no simulation.

Every figure here is closed-form and exact. The tests assert these reproduce the
published odds (Powerball match-3 = 1 in 557, jackpot 1 in 292,201,338, etc.).
"""

from __future__ import annotations

from math import comb, log

from .games import GameSpec, all_games


def hypergeom_pmf(pool: int, drawn: int, picked: int, hits: int) -> float:
    """P(exactly ``hits`` of your ``picked`` numbers are among the ``drawn``).

    Symmetric in (drawn, picked): choosing ``drawn`` balls from ``pool`` and then
    asking how many land in your fixed ``picked``-set.
    """
    if hits < 0 or hits > picked or hits > drawn:
        return 0.0
    if picked - hits > pool - drawn:
        return 0.0
    return comb(drawn, hits) * comb(pool - drawn, picked - hits) / comb(pool, picked)


def main_hits_pmf(spec: GameSpec, hits: int) -> float:
    """Probability of exactly ``hits`` main-number matches."""
    return hypergeom_pmf(spec.main_max, spec.main_count, spec.main_count, hits)


def special_hits_pmf(spec: GameSpec, hits: int) -> float:
    """Probability of exactly ``hits`` special-number matches."""
    if spec.special_count == 0:
        return 1.0 if hits == 0 else 0.0
    return hypergeom_pmf(
        spec.special_max, spec.special_count, spec.special_count, hits
    )


def tier_probability(spec: GameSpec, main_hits: int, special_hits: int) -> float:
    """Exact probability of a single ticket landing in tier (main, special)."""
    return main_hits_pmf(spec, main_hits) * special_hits_pmf(spec, special_hits)


def match_main_exactly(spec: GameSpec, m: int) -> float:
    """Probability of exactly ``m`` main-number matches."""
    return main_hits_pmf(spec, m)


def match_main_at_least(spec: GameSpec, m: int) -> float:
    """Probability of at least ``m`` main-number matches."""
    return sum(main_hits_pmf(spec, h) for h in range(m, spec.main_count + 1))


def jackpot_probability(spec: GameSpec) -> float:
    """Probability of the jackpot tier for one ticket."""
    return tier_probability(spec, spec.main_count, spec.special_count)


def jackpot_odds(spec: GameSpec) -> int:
    """Total number of equally-likely combinations (1-in-N)."""
    n = comb(spec.main_max, spec.main_count)
    if spec.special_count:
        n *= comb(spec.special_max, spec.special_count)
    return n


def order_statistic_means(pool: int, count: int) -> list[float]:
    """E[i-th smallest] for ``count`` distinct draws from 1..pool.

    This is exactly the fixed vector an MSE regressor on *sorted* draws converges
    to — the centroid the deleted LSTM was quietly learning.
    """
    factor = (pool + 1) / (count + 1)
    return [round(i * factor, 4) for i in range(1, count + 1)]


def expected_wait_draws(p: float) -> float:
    """Mean number of draws until a p-probability event first occurs (geometric)."""
    return float("inf") if p <= 0 else 1.0 / p


def median_wait_draws(p: float) -> float:
    """Median draws until first success: ceil(ln 0.5 / ln(1-p))."""
    if p <= 0:
        return float("inf")
    if p >= 1:
        return 1.0
    return log(0.5) / log(1.0 - p)


def odds_table() -> list[dict]:
    """A compact summary row per game, for the README / CLI ``odds`` command."""
    rows = []
    for spec in all_games():
        p3 = match_main_exactly(spec, 3)
        rows.append(
            {
                "game": spec.name,
                "matrix": f"{spec.main_count}/{spec.main_max}"
                + (f" + {spec.special_count}/{spec.special_max}" if spec.special_count else ""),
                "p_match3": p3,
                "match3_one_in": round(1 / p3) if p3 else None,
                "jackpot_one_in": jackpot_odds(spec),
            }
        )
    return rows
