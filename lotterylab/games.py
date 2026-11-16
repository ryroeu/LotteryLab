"""Game definitions — the single source of truth for every lottery's matrix.

Pool sizes, ball counts, prices, prize ranks and (approximate) prize tables live
here so that combinatorics, wheeling, and EV never hardcode a game.

A note on matrices: real lotteries change their ball pools over time (Powerball
went to 5/69 + 1/26 on 2015-10-07; Mega Millions reduced its special pool in 2025).
A CSV downloaded today therefore mixes eras. ``matrix_since`` records when the
complete current matrix took effect; the data layer filters older draws out so odds
and backtests use one consistent matrix.

Prize tables are keyed by ``(main_hits, special_hits)``. The static tables here are
the offline fallback: for the FDJ games they are rounded averages of the operator's
own published per-draw payouts (see ``payouts.py``, which recomputes them from the
snapshots on disk). Pari-mutuel tiers pay a different amount every draw; the
probabilities derived elsewhere are exact, only money figures depend on these.

``rank_tiers`` maps the operator's prize ranks (rank 1 = jackpot) to the tiers that
win them, which is how per-rank winner counts and payouts in the FDJ history files
are joined to the rest of the model. ``pari_ranks`` lists the ranks whose prize is a
shared pool, so a ticket's payout there depends on how many others hold the same
numbers.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

Tier = tuple[int, int]


@dataclass(frozen=True)
class GameSpec:
    """Lottery matrix, prize, and display metadata for one game."""

    key: str
    name: str
    # Main pool: draw ``main_count`` distinct balls from 1..main_max
    main_count: int
    main_max: int
    # Special / bonus pool (Powerball, Mega Ball, Lucky Stars, Dream Number)
    special_count: int
    special_max: int
    special_name: str
    # Economics
    price: float
    currency: str
    # Earliest draw date for which both current ball pools are valid (filtering)
    matrix_since: _dt.date | None
    # Approximate prize table: {(main_hits, special_hits): payout}. The jackpot
    # tier is given separately so ROI uses a finite nominal estimate.
    prize_table: dict[Tier, float] = field(default_factory=dict)
    jackpot_tier: Tier = (0, 0)
    jackpot_estimate: float = 0.0
    draws_per_week: float = 2
    # Operator prize ranks, best first: rank_tiers[0] is rank 1 (the jackpot).
    rank_tiers: tuple[tuple[Tier, ...], ...] = ()
    # Ranks (1-based) paid from a shared pool rather than a fixed amount.
    pari_ranks: frozenset[int] = frozenset()
    # Ranks whose published payout is not a usable lump sum (rollover jackpots,
    # annuities), so the static figure above is kept instead of the data.
    static_ranks: frozenset[int] = frozenset({1})
    # Earliest draw paid under the current prize structure (payout statistics).
    payout_since: _dt.date | None = None
    # Earliest draw for which ``rank_tiers`` describes the operator's ranks
    # (winner counts stay comparable across prize-money changes, not rank changes).
    ranks_since: _dt.date | None = None
    # Winners who can hold the top prize in full before it is split (EuroDreams
    # pays up to three annuities; classic jackpots split from the second winner).
    jackpot_winner_cap: int = 1
    # Guaranteed raffle money per draw for French tickets (My Million, Loto codes).
    raffle_per_draw: float = 0.0
    # Whether the top prize grows when nobody wins it, and where it stops growing.
    jackpot_rolls: bool = True
    jackpot_cap: float | None = None

    def matches_label(self, main_hits: int, special_hits: int) -> str:
        """Human-readable label for a prize tier."""
        s = f"{main_hits} main"
        if self.special_count:
            s += f" + {special_hits} {self.special_name}"
        return s

    def rank_of(self, tier: Tier) -> int | None:
        """Operator prize rank (1-based) that a tier wins, if any."""
        for rank, tiers in enumerate(self.rank_tiers, start=1):
            if tier in tiers:
                return rank
        return None

    @property
    def pari_tiers(self) -> frozenset[Tier]:
        """Tiers whose payout is shared among that draw's winners."""
        return frozenset(
            tier
            for rank, tiers in enumerate(self.rank_tiers, start=1)
            if rank in self.pari_ranks
            for tier in tiers
        )


# --- The registry --------------------------------------------------------------

GAMES: dict[str, GameSpec] = {
    "powerball": GameSpec(
        key="powerball",
        name="Powerball (US)",
        main_count=5,
        main_max=69,
        special_count=1,
        special_max=26,
        special_name="Powerball",
        price=2.00,
        currency="USD",
        matrix_since=_dt.date(2015, 10, 7),
        prize_table={
            (5, 0): 1_000_000,
            (4, 1): 50_000,
            (4, 0): 100,
            (3, 1): 100,
            (3, 0): 7,
            (2, 1): 7,
            (1, 1): 4,
            (0, 1): 4,
        },
        jackpot_tier=(5, 1),
        jackpot_estimate=20_000_000,
        draws_per_week=3,
    ),
    "megamillions": GameSpec(
        key="megamillions",
        name="Mega Millions (US)",
        main_count=5,
        main_max=70,
        special_count=1,
        special_max=24,  # pool reduced 25 -> 24 on 2025-04-08
        special_name="Mega Ball",
        price=5.00,
        currency="USD",
        matrix_since=_dt.date(2025, 4, 8),  # Mega Ball pool changed 25 -> 24
        prize_table={
            # Every play has a random 2x/3x/4x/5x/10x multiplier. Its expected
            # value is 3x, so these are expected rather than guaranteed payouts.
            (5, 0): 3_000_000,
            (4, 1): 30_000,
            (4, 0): 1_500,
            (3, 1): 600,
            (3, 0): 30,
            (2, 1): 30,
            (1, 1): 21,
            (0, 1): 15,
        },
        jackpot_tier=(5, 1),
        jackpot_estimate=50_000_000,
    ),
    "euromillions": GameSpec(
        key="euromillions",
        name="EuroMillions (EU)",
        main_count=5,
        main_max=50,
        special_count=2,
        special_max=12,  # Lucky Stars 1..12 since 2016-09
        special_name="Lucky Star",
        price=2.50,
        currency="EUR",
        matrix_since=_dt.date(2016, 9, 24),
        prize_table={  # mean FDJ payouts, 2020-02 .. 2026-11 (all pari-mutuel)
            (5, 1): 365_000,
            (5, 0): 38_300,
            (4, 2): 1_570,
            (4, 1): 127,
            (3, 2): 61,
            (4, 0): 42,
            (2, 2): 14.6,
            (3, 1): 11.5,
            (3, 0): 9.5,
            (1, 2): 6.9,
            (2, 1): 5.6,
            (2, 0): 4.0,
        },
        jackpot_tier=(5, 2),
        jackpot_estimate=17_000_000,  # the minimum jackpot; rollovers go far higher
        rank_tiers=(
            ((5, 2),),
            ((5, 1),),
            ((5, 0),),
            ((4, 2),),
            ((4, 1),),
            ((3, 2),),
            ((4, 0),),
            ((2, 2),),
            ((3, 1),),
            ((3, 0),),
            ((1, 2),),
            ((2, 1),),
            ((2, 0),),
        ),
        pari_ranks=frozenset(range(1, 14)),
        payout_since=_dt.date(2020, 2, 4),
        ranks_since=_dt.date(2016, 9, 24),
        raffle_per_draw=1_000_000,  # My Million: one French code wins EUR 1M
        jackpot_cap=250_000_000,
    ),
    "eurodreams": GameSpec(
        key="eurodreams",
        name="EuroDreams (EU)",
        main_count=6,
        main_max=40,
        special_count=1,
        special_max=5,
        special_name="Dream Number",
        price=2.50,
        currency="EUR",
        matrix_since=_dt.date(2023, 11, 6),
        # The Dream Number only matters with all six mains. Ranks 3-5 are shared
        # pools (mean FDJ payouts 2025-10 .. 2026-11); two mains refund the ticket.
        prize_table={
            (6, 0): 120_000,  # EUR 2,000/month for 5 years, nominal
            (5, 1): 496,
            (5, 0): 496,
            (4, 1): 34,
            (4, 0): 34,
            (3, 1): 4.9,
            (3, 0): 4.9,
            (2, 1): 2.5,
            (2, 0): 2.5,
        },
        jackpot_tier=(6, 1),
        jackpot_estimate=7_200_000,  # EUR 20,000/month for 30 years, nominal
        rank_tiers=(
            ((6, 1),),
            ((6, 0),),
            ((5, 0), (5, 1)),
            ((4, 0), (4, 1)),
            ((3, 0), (3, 1)),
            ((2, 0), (2, 1)),
        ),
        pari_ranks=frozenset({3, 4, 5}),
        static_ranks=frozenset({1, 2}),
        # FDJ re-weighted the shared ranks on 2025-10-02 (5 mains went from
        # ~EUR 100 to ~EUR 480), so payout statistics start there.
        payout_since=_dt.date(2025, 10, 2),
        ranks_since=_dt.date(2023, 11, 6),
        jackpot_winner_cap=3,
        jackpot_rolls=False,
    ),
    "loto": GameSpec(
        key="loto",
        name="Loto (FR)",
        main_count=5,
        main_max=49,
        special_count=1,
        special_max=10,
        special_name="Numéro Chance",
        price=2.20,
        currency="EUR",
        matrix_since=_dt.date(2008, 10, 6),
        prize_table={  # mean FDJ payouts since the 2019-11 prize structure
            (5, 0): 118_000,
            (4, 1): 2_480,
            (4, 0): 449,
            (3, 1): 54.5,
            (3, 0): 21,
            (2, 1): 10.5,
            (2, 0): 4.4,
            (1, 1): 2.2,
            (0, 1): 2.2,
        },
        jackpot_tier=(5, 1),
        jackpot_estimate=2_000_000,  # the minimum jackpot; +1M per rollover
        draws_per_week=3,
        rank_tiers=(
            ((5, 1),),
            ((5, 0),),
            ((4, 1),),
            ((4, 0),),
            ((3, 1),),
            ((3, 0),),
            ((2, 1),),
            ((2, 0),),
            ((1, 1), (0, 1)),
        ),
        pari_ranks=frozenset(range(1, 9)),
        payout_since=_dt.date(2019, 11, 4),
        ranks_since=_dt.date(2017, 3, 6),
        raffle_per_draw=200_000,  # ten Loto codes win EUR 20,000 each
    ),
}


def get(game: str) -> GameSpec:
    """Look up a game by registry key."""
    try:
        return GAMES[game]
    except KeyError:
        raise KeyError(
            f"Unknown game {game!r}. Known: {', '.join(sorted(GAMES))}"
        ) from None


def all_games() -> list[GameSpec]:
    """Return every registered game in display order."""
    return list(GAMES.values())
