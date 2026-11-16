"""Does avoiding popular numbers really pay? — a walk-forward test on real money.

The popularity model makes a claim: tickets built from numbers other players
neglect collect bigger slices of the shared prize pools. This module checks that
claim the way the backtest checks strategies — out of sample, against what was
actually paid.

1. Fit number popularity using only draws BEFORE a cut-off date.
2. From that fit, form three groups of hypothetical tickets: numbers from the
   least-picked third of the pool, from the whole pool (random), and from the
   most-picked third.
3. For every draw AFTER the cut-off, look up what each ticket would really have
   been paid by rank, from the operator's published payouts for that draw.

The result is each group's *realised sharing multiplier*: money actually collected
from shared ranks, divided by what the same wins would have paid at that period's
average payout. Dividing out the wins removes hit-frequency luck (no group hits
more often — that is the whole premise) and isolates the sharing effect. 1.00 means
"paid like the average winner".

Only the high-volume shared ranks are scored: a rank won once in a hundred thousand
tickets contributes a handful of huge, lucky payouts that would drown the signal.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import numpy as np

from .combinatorics import special_hits_pmf
from .games import GameSpec, get
from .payouts import rank_probability
from .popularity import fit, ranked_numbers, share_multiplier
from .schema import main_columns
from .store import load_payouts

BULK_MIN_PROBABILITY = 1 / 5000  # ranks at least this likely are scored
HOLDOUT = 0.4


def bulk_ranks(spec: GameSpec) -> list[int]:
    """Shared-pool ranks common enough to score without lottery-sized noise."""
    return [
        rank
        for rank in sorted(spec.pari_ranks - spec.static_ranks)
        if rank_probability(spec, rank) >= BULK_MIN_PROBABILITY
    ]


@dataclass
class SharingGroup:
    """One group of hypothetical tickets and how it fared."""

    label: str
    realised: float  # actual shared-rank money / same wins at average payouts
    predicted: float  # what the model, fitted before the cut-off, expected
    wins: float  # shared-rank wins scored (expected over special-ball picks)


@dataclass
class SharingTest:
    """Result of one walk-forward sharing test."""

    game: str
    cutoff: _dt.date
    n_fit_draws: int
    n_test_draws: int
    groups: list[SharingGroup]

    def __str__(self) -> str:
        lines = [
            f"Sharing test for {self.game}: popularity fitted on {self.n_fit_draws} "
            f"draws before {self.cutoff}, scored on the {self.n_test_draws} after it.",
            "  Shared-rank money actually paid, relative to the average winner (1.00):",
            f"    {'tickets drawn from':30s} {'realised':>9s} {'predicted':>10s}",
        ]
        for group in self.groups:
            lines.append(
                f"    {group.label:30s} {group.realised:>9.3f} {group.predicted:>10.3f}"
            )
        return "\n".join(lines)


def sharing_backtest(
    game: str, *, holdout: float = HOLDOUT, n_tickets: int = 4000, seed: int = 0
) -> SharingTest:
    """Run the walk-forward sharing test for one FDJ game (see module docstring)."""
    spec = get(game)
    frame = load_payouts(game, current_rules_only=False)
    frame = frame.sort_values("date").reset_index(drop=True)
    current = frame[frame["date"] >= spec.payout_since] if spec.payout_since else frame
    if len(current) < 60:
        raise ValueError(f"too few {spec.name} draws under the current prize rules")
    cutoff = current["date"].iloc[int(len(current) * (1 - holdout))]
    before = frame[frame["date"] < cutoff]
    after = current[current["date"] >= cutoff].reset_index(drop=True)
    model = fit(spec, before)

    pool, k = spec.main_max, spec.main_count
    ranks = bulk_ranks(spec)
    # paid[d, m]: shared money a ticket with m main hits collects in draw d,
    # averaged over a uniformly random special-ball pick. flat: the same at each
    # rank's average payout for the period. weight[m]: per-rank EV share, used to
    # average the model's per-rank multipliers the same way.
    paid = np.zeros((len(after), k + 1))
    flat = np.zeros((len(after), k + 1))
    rank_value: dict[int, float] = {}
    for rank in ranks:
        column = np.nan_to_num(after[f"r{rank}_payout"].to_numpy(dtype=float))
        average = column[column > 0].mean()
        rank_value[rank] = rank_probability(spec, rank) * average
        for m, s in spec.rank_tiers[rank - 1]:
            chance = special_hits_pmf(spec, s)
            paid[:, m] += chance * column
            flat[:, m] += chance * average

    drawn = np.zeros((len(after), pool + 1))
    for i, row in enumerate(after[main_columns(spec)].to_numpy(dtype=int)):
        drawn[i, row] = 1

    order = [number for number, _ in ranked_numbers(model)]
    third = pool // 3
    sources = {
        "least-picked third of numbers": order[:third],
        "all numbers (random)": list(range(1, pool + 1)),
        "most-picked third of numbers": order[-third:],
    }
    rng = np.random.default_rng(seed)
    draw_index = np.arange(len(after))[None, :]
    groups = []
    for label, numbers in sources.items():
        onehot = np.zeros((n_tickets, pool + 1))
        tickets = []
        for i in range(n_tickets):
            pick = rng.choice(numbers, k, replace=False)
            onehot[i, pick] = 1
            tickets.append(tuple(sorted(int(x) for x in pick)))
        hits = (onehot @ drawn.T).astype(int)
        collected = paid[draw_index, hits].sum()
        at_average = flat[draw_index, hits].sum()

        predicted = []
        for ticket in tickets[:300]:
            total = sum(
                rank_value[rank]
                * share_multiplier(
                    model, ticket, spec, spec.rank_tiers[rank - 1][0][0], rank
                )
                for rank in ranks
            )
            predicted.append(total / sum(rank_value.values()))
        wins = float((flat[draw_index, hits] > 0).sum())
        groups.append(
            SharingGroup(
                label=label,
                realised=float(collected / at_average),
                predicted=float(np.mean(predicted)),
                wins=wins,
            )
        )
    return SharingTest(
        game=spec.name,
        cutoff=cutoff,
        n_fit_draws=len(before),
        n_test_draws=len(after),
        groups=groups,
    )
