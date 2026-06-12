"""Expected value & the one *legitimate* selection edge: jackpot-share avoidance.

Picking unpopular numbers does NOT change your probability of winning. But in
pari-mutuel games the jackpot is split among all tickets matching the draw, so
choosing a combination few others play raises your expected payout *conditional on
winning*. This is the Ziemba / Henze-Riedwyl result, implemented transparently.

The popularity model is a deliberately simple, clearly-labelled heuristic: humans
over-pick 'birthday' numbers (1..31) and visual patterns. It is not a claim to know
real ticket-sales distributions — it's a knob to compare two equally-likely tickets
by how much they'd have to share.
"""

from __future__ import annotations

from dataclasses import dataclass

from .combinatorics import jackpot_odds, jackpot_probability, tier_probability
from .games import GameSpec, get

# Heuristic over-pick multiplier for "birthday" numbers.
BIRTHDAY_WEIGHT = 1.8
BASE_WEIGHT = 1.0


def number_weight(n: int) -> float:
    """Heuristic relative popularity weight for one main number."""
    return BIRTHDAY_WEIGHT if n <= 31 else BASE_WEIGHT


def popularity_multiplier(main: tuple[int, ...], spec: GameSpec) -> float:
    """Relative likelihood a random player picks this exact main combination,
    normalised so a uniform pick == 1.0. < 1 means less popular (shares less).
    """
    mean_w = sum(number_weight(n) for n in range(1, spec.main_max + 1)) / spec.main_max
    prod = 1.0
    for n in main:
        prod *= number_weight(n)
    return prod / (mean_w ** len(main))


def _expected_inv_one_plus_binom(n: int, q: float) -> float:
    """E[1/(1+B)] for B ~ Binomial(n, q) = (1-(1-q)^(n+1)) / ((n+1)q)."""
    if q <= 0:
        return 1.0
    return (1 - (1 - q) ** (n + 1)) / ((n + 1) * q)


@dataclass
class EVReport:
    """Expected-value comparison for one ticket's main numbers."""

    game: str
    main: tuple[int, ...]
    popularity_multiplier: float
    expected_jackpot_share: float  # fraction of jackpot you'd keep if you win it
    ev_per_ticket: float
    ev_per_dollar: float
    currency: str

    def __str__(self) -> str:
        return (
            f"EV for {self.game} ticket main={self.main}\n"
            f"  popularity   : {self.popularity_multiplier:.2f}x uniform  "
            f"({'less' if self.popularity_multiplier < 1 else 'more'} shared)\n"
            f"  jackpot share: {self.expected_jackpot_share:.1%} kept if you win it\n"
            f"  EV/ticket    : {self.ev_per_ticket:,.4f} {self.currency} "
            f"(price {self._price():,.2f})\n"
            f"  EV/dollar    : {self.ev_per_dollar:.4f}  "
            f"(< 1 always — you still lose, just less)"
        )

    def _price(self) -> float:
        """Ticket price for this report's game."""
        return get(self.game).price


def ticket_ev(
    main: tuple[int, ...],
    spec: GameSpec,
    *,
    jackpot: float | None = None,
    n_players: int = 10_000_000,
) -> EVReport:
    """Estimate one ticket's EV under the jackpot-sharing popularity model."""
    jackpot = spec.jackpot_estimate if jackpot is None else jackpot
    pop = popularity_multiplier(main, spec)

    # Jackpot sharing: other players who also picked the winning combination.
    q = pop / jackpot_odds(spec)  # prob a random player picks YOUR exact combo
    share = _expected_inv_one_plus_binom(n_players, q)

    ev = 0.0
    for (m, s), payout in spec.prize_table.items():
        ev += tier_probability(spec, m, s) * payout
    # Jackpot tier, sharing-adjusted:
    ev += jackpot_probability(spec) * jackpot * share

    return EVReport(
        game=spec.key,
        main=tuple(sorted(main)),
        popularity_multiplier=pop,
        expected_jackpot_share=share,
        ev_per_ticket=ev,
        ev_per_dollar=ev / spec.price if spec.price else 0.0,
        currency=spec.currency,
    )
