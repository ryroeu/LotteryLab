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

import numpy as np
import pandas as pd

from .baseline import three_plus_main_probability
from .games import GameSpec
from .schema import frame_to_draws
from .strategy import Strategy
from .validate import validate_ticket


def score_ticket(ticket, draw_main: set[int], draw_special: set[int]) -> tuple[int, int]:
    main, special = ticket
    return len(set(main) & draw_main), len(set(special) & draw_special)


@dataclass
class BacktestResult:
    game: str
    strategy: str
    n_draws: int
    n_tickets: int
    tier_hits: dict[tuple[int, int], int]
    three_plus_hits: int
    three_plus_rate: float
    baseline_three_plus_rate: float
    z_vs_baseline: float
    spent: float
    won: float
    currency: str

    @property
    def roi(self) -> float:
        return self.won / self.spent if self.spent else 0.0

    def __str__(self) -> str:
        opportunities = self.n_draws * self.n_tickets
        lines = [
            f"Backtest: {self.strategy} on {self.game}",
            f"  draws evaluated : {self.n_draws}  (tickets played: {opportunities})",
            f"  >=3 main hits   : {self.three_plus_hits}  "
            f"(rate {self.three_plus_rate:.5f}, 1 in "
            f"{round(1/self.three_plus_rate) if self.three_plus_rate else float('inf')})",
            f"  baseline rate   : {self.baseline_three_plus_rate:.5f}  "
            f"(1 in {round(1/self.baseline_three_plus_rate)})",
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


def backtest(
    strategy: Strategy,
    history: pd.DataFrame,
    spec: GameSpec,
    *,
    n_tickets: int = 1,
    warmup: int = 50,
    seed: int = 0,
) -> BacktestResult:
    rng = np.random.default_rng(seed)
    draws = frame_to_draws(history, spec)
    n = len(draws)
    warmup = min(warmup, max(1, n - 1))

    tier = Counter()
    three_plus = 0
    spent = won = 0.0

    for t in range(warmup, n):
        past = history.iloc[:t]
        actual = draws[t]
        dmain, dspecial = set(actual.main), set(actual.special)
        for ticket in strategy.generate(past, spec, n_tickets, rng):
            validate_ticket(ticket[0], ticket[1], spec)  # impossible tickets can't pass
            m, s = score_ticket(ticket, dmain, dspecial)
            tier[(m, s)] += 1
            if m >= 3:
                three_plus += 1
            spent += spec.price
            if (m, s) == spec.jackpot_tier:
                won += spec.jackpot_estimate
            else:
                won += spec.prize_table.get((m, s), 0.0)

    opportunities = (n - warmup) * n_tickets
    p3 = three_plus_main_probability(spec)
    rate = three_plus / opportunities if opportunities else 0.0
    # z-score: count of >=3 hits ~ Binomial(opportunities, p3)
    se = (opportunities * p3 * (1 - p3)) ** 0.5
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
        z_vs_baseline=z,
        spent=spent,
        won=won,
        currency=spec.currency,
    )
