"""Expected value — and the only selection edges that are real.

No choice of numbers changes the probability of winning. Two things do change what
a ticket is *worth*:

* **Who you share with.** Most prizes in the French games are pools split among
  that draw's winners. A combination few others play collects a bigger slice of
  every shared tier — not only the jackpot. (Ziemba; Henze & Riedwyl.) How popular
  each number is comes from ``popularity.py``.
* **When you play.** A rollover jackpot adds value to every ticket in that draw,
  but also attracts more players to split it with. ``Market`` models both from the
  operator's records, so ``evaluate`` can price a ticket at any advertised jackpot.

French lottery winnings are not taxed, so these are the amounts actually received
(annuity prizes are counted at their nominal total). Every figure here is still
below the ticket price in ordinary draws: this module measures how much less you
lose, and says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, sqrt
from numbers import Integral

import pandas as pd

from . import payouts
from .combinatorics import jackpot_odds, jackpot_probability, tier_probability
from .games import GameSpec
from .popularity import (
    PopularityModel,
    combo_popularity,
    heuristic_model,
    load_model,
    share_multiplier,
)

DEFAULT_PLAYERS = 10_000_000


def jackpot_share(expected_cowinners: float, cap: int = 1) -> float:
    """Expected fraction of the top prize a winner keeps: ``E[min(1, cap/(1+B))]``
    with ``B ~ Poisson(expected_cowinners)`` other winning tickets. ``cap`` is how
    many winners are paid in full before the prize is split (1 for a classic
    jackpot)."""
    lam = expected_cowinners
    if lam <= 0:
        return 1.0
    if cap == 1:
        return (1 - exp(-lam)) / lam
    term = exp(-lam)  # P(B = 0)
    total, b = 0.0, 0
    limit = int(lam + 12 * sqrt(lam) + 30)
    while b <= limit:
        total += term * min(1.0, cap / (1 + b))
        b += 1
        term *= lam / b
    return total


@dataclass
class EVReport:
    """Expected value of one ticket, split by where the money comes from."""

    game: str
    main: tuple[int, ...]
    popularity_multiplier: float  # this combination vs a uniform pick
    expected_jackpot_share: float  # fraction of the top prize kept if you win it
    shared_multiplier: float  # shared-tier payout vs the average winner's
    ev_jackpot: float
    ev_shared: float
    ev_fixed: float
    ev_raffle: float
    price: float
    currency: str
    jackpot: float
    n_players: float
    model_note: str = ""

    @property
    def ev_per_ticket(self) -> float:
        """Total expected winnings per ticket."""
        return self.ev_jackpot + self.ev_shared + self.ev_fixed + self.ev_raffle

    @property
    def return_rate(self) -> float:
        """Expected winnings per unit staked (1.0 would be break-even)."""
        return self.ev_per_ticket / self.price if self.price else 0.0

    def __str__(self) -> str:
        cur = self.currency
        lines = [
            f"EV for {self.game} ticket main={self.main}",
            f"  popularity    : {self.popularity_multiplier:.2f}x a random pick  "
            f"({'less' if self.popularity_multiplier < 1 else 'more'} shared)",
            f"  top prize     : {self.jackpot:,.0f} {cur}, "
            f"{self.expected_jackpot_share:.1%} kept if you win it "
            f"(~{self.n_players / 1e6:.1f}M tickets in the draw)",
            f"  shared tiers  : pay {self.shared_multiplier:.3f}x the average "
            "winner's amount for this ticket",
            f"  EV breakdown  : top {self.ev_jackpot:.3f} + shared "
            f"{self.ev_shared:.3f} + fixed {self.ev_fixed:.3f} + raffle "
            f"{self.ev_raffle:.3f}",
            f"  EV / ticket   : {self.ev_per_ticket:,.3f} {cur} "
            f"for a {self.price:,.2f} ticket -> {self.return_rate:.1%} returned",
        ]
        if self.model_note:
            lines.append(f"  popularity model: {self.model_note}")
        return "\n".join(lines)


def ticket_ev(
    main,
    spec: GameSpec,
    *,
    jackpot: float | None = None,
    n_players: float = DEFAULT_PLAYERS,
    model: PopularityModel | None = None,
    raffle: float = 0.0,
) -> EVReport:
    """Expected value of one ticket with these main numbers.

    ``model`` says how popular each number is (default: the birthday heuristic);
    ``n_players`` is the number of other tickets in the draw; ``raffle`` is any
    per-ticket raffle value to add. Special balls are assumed uniformly popular.
    """
    jackpot = spec.jackpot_estimate if jackpot is None else float(jackpot)
    if not isfinite(jackpot) or jackpot < 0:
        raise ValueError("jackpot must be a finite, non-negative amount")
    if isinstance(n_players, bool) or not isfinite(n_players) or n_players < 0:
        raise ValueError("n_players must be a finite, non-negative number")
    if isinstance(n_players, Integral):
        n_players = int(n_players)
    model = model or heuristic_model(spec)
    main = tuple(sorted(main))
    pop = combo_popularity(model, main, spec)

    # Other tickets expected on YOUR exact combination (specials taken as uniform).
    cowinners = n_players * pop / jackpot_odds(spec)
    share = jackpot_share(cowinners, spec.jackpot_winner_cap)
    ev_jackpot = jackpot_probability(spec) * jackpot * share

    shared_tiers = spec.pari_tiers
    multipliers: dict[tuple[int, int | None], float] = {}
    ev_shared = ev_fixed = baseline_shared = 0.0
    for (m, s), payout in spec.prize_table.items():
        if (m, s) == spec.jackpot_tier:
            continue
        value = tier_probability(spec, m, s) * payout
        if (m, s) in shared_tiers:
            key = (m, spec.rank_of((m, s)))
            if key not in multipliers:
                multipliers[key] = share_multiplier(model, main, spec, *key)
            ev_shared += value * multipliers[key]
            baseline_shared += value
        else:
            ev_fixed += value

    return EVReport(
        game=spec.key,
        main=main,
        popularity_multiplier=pop,
        expected_jackpot_share=share,
        shared_multiplier=ev_shared / baseline_shared if baseline_shared else 1.0,
        ev_jackpot=ev_jackpot,
        ev_shared=ev_shared,
        ev_fixed=ev_fixed,
        ev_raffle=raffle,
        price=spec.price,
        currency=spec.currency,
        jackpot=jackpot,
        n_players=float(n_players),
        model_note=model.describe(),
    )


# --- The market: everything measured about one game ------------------------------


@dataclass
class Market:
    """One game as measured from the snapshots on disk: its real prize table, how
    popular each number is, how many tickets are in a draw, and raffle value."""

    spec: GameSpec  # prize table calibrated to observed payouts when available
    model: PopularityModel
    frame: pd.DataFrame | None  # per-draw payouts, or None for games without them
    raffle: float

    @property
    def measured(self) -> bool:
        """Whether this market rests on operator payout data (vs. static guesses)."""
        return self.frame is not None

    def tickets(self, jackpot: float | None = None) -> float:
        """Tickets expected in a draw offering ``jackpot``."""
        if self.frame is None:
            return float(DEFAULT_PLAYERS)
        if jackpot is None:
            return payouts.typical_tickets(self.spec, self.frame)
        return payouts.tickets_at(self.spec, self.frame, jackpot)


def market(game: str) -> Market:
    """Measure a game's market from disk (static fallbacks where data is absent)."""
    frame = payouts.try_load(game)
    spec = payouts.calibrated_spec(game)
    return Market(
        spec=spec,
        model=load_model(game),
        frame=frame,
        raffle=payouts.raffle_ev(spec, frame),
    )


def evaluate(main, mkt: Market, *, jackpot: float | None = None) -> EVReport:
    """Price one ticket in ``mkt`` at an advertised ``jackpot`` (default: the
    game's starting jackpot), with sales scaled to that jackpot."""
    jackpot = mkt.spec.jackpot_estimate if jackpot is None else float(jackpot)
    return ticket_ev(
        main,
        mkt.spec,
        jackpot=jackpot,
        n_players=mkt.tickets(jackpot),
        model=mkt.model,
        raffle=mkt.raffle,
    )


def breakeven_jackpot(main, mkt: Market, *, ceiling: float = 5e9) -> float | None:
    """Smallest jackpot at which this ticket's EV reaches its price, allowing for
    the extra players a bigger jackpot attracts. None if it never does (or the
    game's top prize does not roll over)."""
    spec = mkt.spec
    if not spec.jackpot_rolls:
        return None

    def gap(jackpot: float) -> float:
        return evaluate(main, mkt, jackpot=jackpot).ev_per_ticket - spec.price

    low, high = spec.jackpot_estimate, ceiling
    if gap(low) >= 0:
        return low
    if gap(high) < 0:
        return None
    for _ in range(60):
        mid = (low + high) / 2
        if gap(mid) >= 0:
            high = mid
        else:
            low = mid
    return high


def best_and_worst(mkt: Market) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The least- and most-popular main combinations under the market's model."""
    ranked = sorted(range(1, mkt.spec.main_max + 1), key=mkt.model.weight)
    k = mkt.spec.main_count
    return tuple(sorted(ranked[:k])), tuple(sorted(ranked[-k:]))
