"""The backtest harness — the heart of the project.

Walk forward over real draws. At each draw, ask the strategy for ticket(s) using
ONLY the history available before that draw (no leakage — the exact bug the LSTM
committed by fitting its scaler on the whole dataset). Score the matches, tally
the tiers, and compare the realized match distribution to the exact baseline.

The headline finding, reproduced for every strategy: the number of >=3-main hits
sits within a couple of standard errors of the hypergeometric expectation. No edge.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from .baseline import three_plus_main_probability
from .combinatorics import joint_match_main_at_least
from .games import GameSpec
from .schema import frame_to_draws
from .strategy import Strategy
from .validate import validate_ticket


def score_ticket(
    ticket, draw_main: set[int], draw_special: set[int]
) -> tuple[int, int]:
    """Count main and special matches for one ticket against one draw."""
    main, special = ticket
    return len(set(main) & draw_main), len(set(special) & draw_special)


@dataclass
class BacktestResult:
    """Aggregate walk-forward backtest metrics for one strategy."""

    game: str
    strategy: str
    n_draws: int
    n_tickets: int
    tier_hits: dict[tuple[int, int], int]
    three_plus_hits: int
    three_plus_rate: float
    baseline_three_plus_rate: float
    baseline_three_plus_se: float
    z_vs_baseline: float
    spent: float
    won: float
    currency: str

    @property
    def roi(self) -> float:
        """Return winnings divided by spend."""
        return self.won / self.spent if self.spent else 0.0

    def __str__(self) -> str:
        opportunities = self.n_draws * self.n_tickets
        lines = [
            f"Backtest: {self.strategy} on {self.game}",
            f"  draws evaluated : {self.n_draws}  (tickets played: {opportunities})",
            f"  >=3 main hits   : {self.three_plus_hits}  "
            f"(rate {self.three_plus_rate:.5f}, 1 in "
            f"{round(1 / self.three_plus_rate) if self.three_plus_rate else float('inf')})",
            f"  baseline rate   : {self.baseline_three_plus_rate:.5f}  "
            f"(1 in {round(1 / self.baseline_three_plus_rate)})",
            f"  z vs baseline   : {self.z_vs_baseline:+.2f}  "
            f"({'no edge' if abs(self.z_vs_baseline) < 2 else 'INVESTIGATE'})",
            f"  spend / win     : {self.spent:,.2f} / {self.won:,.2f} {self.currency}  "
            f"(ROI {self.roi:.3f})",
        ]
        if self.tier_hits:
            lines.append("  tier breakdown  :")
            for (m, s), c in sorted(self.tier_hits.items(), reverse=True):
                if m + s == 0:
                    continue
                lines.append(f"      {m} main + {s} special : {c}")
        return "\n".join(lines)


def _generated_tickets(strategy, past, spec, n_tickets, rng):
    """Generate exactly the requested number of tickets or fail loudly."""
    tickets = list(strategy.generate(past, spec, n_tickets, rng))
    if len(tickets) != n_tickets:
        raise ValueError(
            f"Strategy {strategy.name!r} returned {len(tickets)} ticket(s); "
            f"expected {n_tickets}."
        )
    return tickets


def _null_hit_count_variance(tickets, spec: GameSpec, probability: float) -> float:
    """Exact conditional variance for >=3-main hits across one draw's tickets."""
    variance = len(tickets) * probability * (1 - probability)
    main_sets = [set(ticket[0]) for ticket in tickets]
    for first, main_a in enumerate(main_sets):
        for main_b in main_sets[first + 1 :]:
            overlap = len(main_a & main_b)
            joint = joint_match_main_at_least(spec, overlap, 3)
            variance += 2 * (joint - probability**2)
    return variance


def _score_tickets(tickets, actual, spec: GameSpec):
    """Validate and score one draw's ticket block."""
    tier = Counter()
    three_plus = 0
    won = 0.0
    draw_main, draw_special = set(actual.main), set(actual.special)
    for ticket in tickets:
        validate_ticket(ticket[0], ticket[1], spec)
        main_hits, special_hits = score_ticket(ticket, draw_main, draw_special)
        tier[(main_hits, special_hits)] += 1
        if main_hits >= 3:
            three_plus += 1
        if (main_hits, special_hits) == spec.jackpot_tier:
            won += spec.jackpot_estimate
        else:
            won += spec.prize_table.get((main_hits, special_hits), 0.0)
    return tier, three_plus, won


def backtest(
    strategy: Strategy,
    history: pd.DataFrame,
    spec: GameSpec,
    *,
    n_tickets: int = 1,
    warmup: int = 50,
    seed: int = 0,
) -> BacktestResult:
    """Walk a strategy forward through history and score every generated ticket."""
    if isinstance(n_tickets, bool) or not isinstance(n_tickets, Integral):
        raise ValueError("n_tickets must be a positive integer")
    if n_tickets <= 0:
        raise ValueError("n_tickets must be a positive integer")
    if isinstance(warmup, bool) or not isinstance(warmup, Integral):
        raise ValueError("warmup must be a non-negative integer")
    if warmup < 0:
        raise ValueError("warmup must be a non-negative integer")

    # The no-lookahead guarantee depends on chronological order. Canonical data is
    # already sorted, but normalizing here keeps the public API safe for callers.
    history = history.sort_values("date", kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    draws = frame_to_draws(history, spec)
    n = len(draws)
    warmup = min(warmup, n)

    tier = Counter()
    three_plus = 0
    won = 0.0
    p3 = three_plus_main_probability(spec)
    null_variance = 0.0

    for t in range(warmup, n):
        past = history.iloc[:t]
        tickets = _generated_tickets(strategy, past, spec, n_tickets, rng)
        draw_tier, draw_three_plus, draw_won = _score_tickets(tickets, draws[t], spec)
        tier.update(draw_tier)
        three_plus += draw_three_plus
        won += draw_won
        null_variance += _null_hit_count_variance(tickets, spec, p3)

    opportunities = (n - warmup) * n_tickets
    spent = opportunities * spec.price
    rate = three_plus / opportunities if opportunities else 0.0
    # Ticket outcomes from one draw are correlated when their main picks overlap.
    # The conditional null variance above is exact for those fixed ticket sets.
    se = max(0.0, null_variance) ** 0.5
    z = (three_plus - opportunities * p3) / se if se > 0 else 0.0

    return BacktestResult(
        game=spec.key,
        strategy=strategy.name,
        n_draws=n - warmup,
        n_tickets=n_tickets,
        tier_hits=dict(tier),
        three_plus_hits=three_plus,
        three_plus_rate=rate,
        baseline_three_plus_rate=p3,
        baseline_three_plus_se=(se / opportunities if opportunities else 0.0),
        z_vs_baseline=z,
        spent=spent,
        won=won,
        currency=spec.currency,
    )
