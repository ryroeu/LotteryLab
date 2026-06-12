"""Variance simulator + 'how long until a 3-match?' calculator.

These make the abstract odds *felt*: play a strategy for many seasons and watch the
net-loss distribution and its long unlucky tail; or read, per game, the expected and
median wait for a single ticket to finally match three.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baseline import all_tier_probabilities
from .combinatorics import expected_wait_draws, match_main_exactly, median_wait_draws
from .games import GameSpec


@dataclass
class WaitReport:
    """Calendar-time summary for waiting until one match-3 hit."""

    game: str
    p_match3: float
    one_in: int
    draws_per_week: float
    expected_draws: float
    median_draws: float
    expected_years: float
    median_years: float

    def __str__(self) -> str:
        return (
            f"{self.game}: match-3 = 1 in {self.one_in}\n"
            f"  one ticket/draw -> expected wait {self.expected_draws:,.0f} draws "
            f"(~{self.expected_years:.1f} yrs), median ~{self.median_years:.1f} yrs "
            f"(@ {self.draws_per_week:g} draws/week)"
        )


# Approximate draws per week per game (for translating draws -> calendar time).
DRAWS_PER_WEEK = {
    "powerball": 3,
    "megamillions": 2,
    "euromillions": 2,
    "eurodreams": 2,  # Monday & Thursday
}


def time_to_match3(spec: GameSpec) -> WaitReport:
    """Calculate expected and median waits for one ticket to match 3 mains."""
    p = match_main_exactly(spec, 3)
    dpw = DRAWS_PER_WEEK.get(spec.key, 2)
    exp_d = expected_wait_draws(p)
    med_d = median_wait_draws(p)
    return WaitReport(
        game=spec.name,
        p_match3=p,
        one_in=round(1 / p),
        draws_per_week=dpw,
        expected_draws=exp_d,
        median_draws=med_d,
        expected_years=exp_d / (dpw * 52),
        median_years=med_d / (dpw * 52),
    )


@dataclass
class VarianceReport:
    """Summary statistics for repeated one-ticket simulated seasons."""

    game: str
    strategy: str
    draws: int
    n_seasons: int
    mean_net: float
    p05_net: float
    p50_net: float
    p95_net: float
    best_net: float
    currency: str
    any_3plus_fraction: float  # fraction of seasons with >=1 three-match

    def __str__(self) -> str:
        return (
            f"Variance: {self.strategy} on {self.game}, "
            f"{self.draws} draws x {self.n_seasons} seasons (1 ticket/draw)\n"
            f"  net result {self.currency}: mean {self.mean_net:,.0f}  "
            f"[p05 {self.p05_net:,.0f} | median {self.p50_net:,.0f} | "
            f"p95 {self.p95_net:,.0f} | best {self.best_net:,.0f}]\n"
            f"  seasons with >=1 three-match: {self.any_3plus_fraction:.1%}"
        )


def variance_samples(
    spec: GameSpec, *, draws: int = 104, n_seasons: int = 2000, seed: int = 0
) -> tuple[np.ndarray, float]:
    """Monte-Carlo a uniform player over ``n_seasons`` independent seasons.

    Net per season = winnings - spend. Uses exact tier probabilities to draw
    outcomes (no need to simulate ball-by-ball). Returns the per-season net
    array plus the fraction of seasons containing at least one 3-match.
    """
    rng = np.random.default_rng(seed)

    # all_tier_probabilities is the COMPLETE distribution over (main, special)
    # outcomes and already sums to 1 — every losing tier (incl. (0,0)) is in it.
    tiers = all_tier_probabilities(spec)
    tier_keys = list(tiers.keys())
    probs = np.array([tiers[k] for k in tier_keys])
    probs = probs / probs.sum()  # guard against float drift
    # payout per tier (jackpot uses nominal estimate; non-winning tiers -> 0)
    payouts = np.array(
        [
            spec.jackpot_estimate if k == spec.jackpot_tier else spec.prize_table.get(k, 0.0)
            for k in tier_keys
        ]
    )
    three_plus_idx = np.array([1 if k[0] >= 3 else 0 for k in tier_keys], dtype=bool)

    nets = np.empty(n_seasons)
    any3 = 0
    for s in range(n_seasons):
        picks = rng.choice(len(probs), size=draws, p=probs)
        winnings = payouts[picks].sum()
        nets[s] = winnings - draws * spec.price
        if three_plus_idx[picks].any():
            any3 += 1

    return nets, any3 / n_seasons


def simulate_variance(
    spec: GameSpec, *, draws: int = 104, n_seasons: int = 2000, seed: int = 0
) -> VarianceReport:
    """Summary statistics over ``variance_samples`` — see there for the model."""
    nets, any3_fraction = variance_samples(
        spec, draws=draws, n_seasons=n_seasons, seed=seed
    )

    return VarianceReport(
        game=spec.name,
        strategy="random",
        draws=draws,
        n_seasons=n_seasons,
        mean_net=float(nets.mean()),
        p05_net=float(np.percentile(nets, 5)),
        p50_net=float(np.percentile(nets, 50)),
        p95_net=float(np.percentile(nets, 95)),
        best_net=float(nets.max()),
        currency=spec.currency,
        any_3plus_fraction=any3_fraction,
    )
