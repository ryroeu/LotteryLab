"""Game definitions — the single source of truth for every lottery's matrix.

Pool sizes, ball counts, prices, and (approximate) prize tables live here so that
combinatorics, wheeling, and EV never hardcode a game.

A note on matrices: real lotteries change their ball pools over time (Powerball
went to 5/69 + 1/26 on 2015-10-07; Mega Millions to 5/70 mains on 2017-10-31).
A CSV downloaded today therefore mixes eras. ``main_matrix_since`` records when
the *current main pool* took effect; the data layer filters older draws out so the
odds and backtests are computed against a single, consistent matrix.

Prize tables are keyed by ``(main_hits, special_hits)``. They are APPROXIMATE and
illustrative — pari-mutuel games (EuroMillions, EuroDreams) pay variable amounts
per draw. The probabilities derived elsewhere are exact; only ROI figures depend
on these prizes, and ROI is a secondary, clearly-labelled metric.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GameSpec:
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
    # Earliest draw date for which the current main matrix is valid (filtering)
    main_matrix_since: _dt.date | None
    # Approximate prize table: {(main_hits, special_hits): payout}. The jackpot
    # tier is given separately so ROI uses a finite nominal estimate.
    prize_table: dict[tuple[int, int], float] = field(default_factory=dict)
    jackpot_tier: tuple[int, int] = (0, 0)
    jackpot_estimate: float = 0.0

    @property
    def special_pool(self) -> int:
        return self.special_max

    def matches_label(self, main_hits: int, special_hits: int) -> str:
        s = f"{main_hits} main"
        if self.special_count:
            s += f" + {special_hits} {self.special_name}"
        return s


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
        main_matrix_since=_dt.date(2015, 10, 7),
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
        main_matrix_since=_dt.date(2017, 10, 31),  # 5/70 mains stable since then
        prize_table={
            (5, 0): 1_000_000,
            (4, 1): 10_000,
            (4, 0): 500,
            (3, 1): 200,
            (3, 0): 10,
            (2, 1): 10,
            (1, 1): 7,
            (0, 1): 5,
        },
        jackpot_tier=(5, 1),
        jackpot_estimate=20_000_000,
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
        main_matrix_since=_dt.date(2016, 9, 24),
        prize_table={  # approximate typical pari-mutuel values
            (5, 1): 130_000,
            (5, 0): 30_000,
            (4, 2): 1_200,
            (4, 1): 120,
            (3, 2): 70,
            (4, 0): 50,
            (2, 2): 15,
            (3, 1): 12,
            (3, 0): 10,
            (1, 2): 8,
            (2, 1): 7,
            (2, 0): 5,
        },
        jackpot_tier=(5, 2),
        jackpot_estimate=50_000_000,
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
        main_matrix_since=_dt.date(2023, 11, 6),
        prize_table={  # approximate; top tiers are annuities shown as lump nominal
            (6, 0): 2_500_000,
            (5, 1): 2_000,
            (5, 0): 200,
            (4, 1): 20,
            (4, 0): 10,
            (3, 1): 6,
            (3, 0): 4,
            (2, 1): 4,
            (1, 1): 2.5,
        },
        jackpot_tier=(6, 1),
        jackpot_estimate=7_200_000,  # ~€20k/month for 30 years, nominal
    ),
}


def get(game: str) -> GameSpec:
    try:
        return GAMES[game]
    except KeyError:
        raise KeyError(
            f"Unknown game {game!r}. Known: {', '.join(sorted(GAMES))}"
        ) from None


def all_games() -> list[GameSpec]:
    return list(GAMES.values())
