"""What the games actually pay — from the operator's own per-draw records.

FDJ publishes, for every draw, how many tickets won each prize rank and what each
winner received. That turns three things the rest of the project would otherwise
have to guess into measurements:

* the **prize table** (mean payout per tier, instead of hand-typed estimates);
* the **number of tickets in play** (winners of a rank ÷ that rank's exact
  probability), which sets how likely a jackpot is to be split;
* the raw material for the **popularity model** in ``popularity.py``.

Only the FDJ games (EuroMillions, EuroDreams, Loto) have this data; for the others
every function here falls back to the static table in ``games.py``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .combinatorics import tier_probability
from .games import GameSpec, get
from .store import load_payouts

# A rank whose payout varies less than this (coefficient of variation) across
# draws is a fixed prize, whatever the rules say.
FIXED_CV = 0.01
# How many recent draws describe "current" ticket sales.
RECENT_DRAWS = 104


def rank_probability(spec: GameSpec, rank: int) -> float:
    """Exact probability that one ticket wins operator prize rank ``rank``."""
    return sum(tier_probability(spec, m, s) for m, s in spec.rank_tiers[rank - 1])


def rank_label(spec: GameSpec, rank: int) -> str:
    """'5 main + 1 Lucky Star' style label for a rank (collapsing 'any special')."""
    tiers = spec.rank_tiers[rank - 1]
    mains = sorted({m for m, _ in tiers})
    specials = {s for _, s in tiers}
    if len(mains) > 1:
        return f"{mains[0]}-{mains[-1]} main + {spec.special_name}"
    if len(specials) == spec.special_count + 1 or len(specials) > 1:
        return f"{mains[0]} main"
    return spec.matches_label(mains[0], next(iter(specials)))


def try_load(game: str, *, current_rules_only: bool = True) -> pd.DataFrame | None:
    """The payout frame for a game, or None when it has no such data."""
    try:
        frame = load_payouts(game, current_rules_only=current_rules_only)
    except (FileNotFoundError, ValueError, KeyError):
        return None
    return frame if len(frame) else None


def rank_summary(spec: GameSpec, frame: pd.DataFrame) -> pd.DataFrame:
    """One row per prize rank: odds, observed payout statistics, and its kind
    (``fixed``, ``shared`` pool, or ``top`` prize kept static)."""
    rows = []
    for rank in range(1, len(spec.rank_tiers) + 1):
        winners = frame[f"r{rank}_winners"].fillna(frame[f"r{rank}_winners_fr"])
        payout = frame[f"r{rank}_payout"]
        paid = payout[(winners > 0) & (payout > 0)]
        mean = float(paid.mean()) if len(paid) else float("nan")
        cv = float(paid.std(ddof=0) / mean) if len(paid) > 1 and mean else 0.0
        if rank in spec.static_ranks:
            kind = "top"
        elif len(paid) and cv < FIXED_CV:
            kind = "fixed"
        else:
            kind = "shared"
        rows.append(
            {
                "rank": rank,
                "label": rank_label(spec, rank),
                "probability": rank_probability(spec, rank),
                "draws_paid": int(len(paid)),
                "mean_payout": mean,
                "median_payout": float(paid.median()) if len(paid) else float("nan"),
                "min_payout": float(paid.min()) if len(paid) else float("nan"),
                "max_payout": float(paid.max()) if len(paid) else float("nan"),
                "cv": cv,
                "mean_winners": float(winners.mean()),
                "kind": kind,
            }
        )
    return pd.DataFrame(rows)


def empirical_prize_table(spec: GameSpec, frame: pd.DataFrame) -> dict:
    """``spec.prize_table`` with every measurable tier replaced by its mean observed
    payout. Top ranks (rollover jackpots, annuities) keep their static figure."""
    table = dict(spec.prize_table)
    summary = rank_summary(spec, frame)
    for row in summary.itertuples():
        if row.rank in spec.static_ranks or not np.isfinite(row.mean_payout):
            continue
        for tier in spec.rank_tiers[row.rank - 1]:
            if tier != spec.jackpot_tier:
                table[tier] = round(row.mean_payout, 2)
    return table


def calibrated_spec(game: str) -> GameSpec:
    """The game's spec with its prize table measured from the snapshots on disk
    (unchanged when the game has no payout data)."""
    spec = get(game)
    frame = try_load(game)
    if frame is None:
        return spec
    return replace(spec, prize_table=empirical_prize_table(spec, frame))


def tickets_per_draw(
    spec: GameSpec, frame: pd.DataFrame, *, french: bool = False
) -> pd.Series:
    """Estimated tickets in each draw: winners of the most-won rank ÷ its exact
    probability. ``french`` counts French tickets only (raffle pools are French).

    The most-won rank has hundreds of thousands of winners, so sampling noise is
    negligible; what remains is a few percent of bias from which numbers came up
    (popular numbers inflate the count), which averaging over draws removes.
    """
    rank = len(spec.rank_tiers)
    column = f"r{rank}_winners_fr" if french else f"r{rank}_winners"
    counts = frame[column]
    if not french:
        counts = counts.fillna(frame[f"r{rank}_winners_fr"])
    return counts / rank_probability(spec, rank)


def typical_tickets(
    spec: GameSpec, frame: pd.DataFrame, *, french: bool = False
) -> float:
    """Median estimated tickets per draw over the most recent draws."""
    series = tickets_per_draw(spec, frame, french=french).dropna()
    return float(series.tail(RECENT_DRAWS).median())


def raffle_ev(spec: GameSpec, frame: pd.DataFrame | None) -> float:
    """Expected raffle winnings per French ticket (My Million, Loto codes): the
    guaranteed raffle money per draw spread over the French tickets in it."""
    if not spec.raffle_per_draw or frame is None:
        return 0.0
    tickets = typical_tickets(spec, frame, french=True)
    return spec.raffle_per_draw / tickets if tickets > 0 else 0.0


def jackpot_history(spec: GameSpec, frame: pd.DataFrame) -> pd.DataFrame:
    """Per draw: the jackpot on offer (where the operator records it even when
    nobody won — Loto does) against estimated tickets sold."""
    winners = frame["r1_winners"].fillna(frame["r1_winners_fr"])
    return pd.DataFrame(
        {
            "date": frame["date"],
            "cycle": frame["cycle"],
            "jackpot": (frame["r1_payout"] * winners.clip(lower=1)).where(
                frame["r1_payout"] > 0
            ),
            "jackpot_winners": winners,
            "tickets": tickets_per_draw(spec, frame),
        }
    )


def sales_model(spec: GameSpec, frame: pd.DataFrame) -> tuple[float, float] | None:
    """``(base, per_unit)``: tickets sold ~ base + per_unit x jackpot, by least
    squares over the draws whose jackpot is on record. None when the jackpot never
    varies or there is too little to fit. Bigger jackpots pull in more players,
    which is exactly what erodes the value of a big jackpot."""
    if not spec.jackpot_rolls:
        return None
    history = jackpot_history(spec, frame).dropna(subset=["jackpot", "tickets"])
    if len(history) < 30 or history["jackpot"].nunique() < 5:
        return None
    slope, base = np.polyfit(history["jackpot"], history["tickets"], 1)
    return float(base), float(max(slope, 0.0))


def tickets_at(spec: GameSpec, frame: pd.DataFrame, jackpot: float) -> float:
    """Estimated tickets in a draw offering ``jackpot`` (typical sales when the
    game has no usable jackpot-sales relationship)."""
    model = sales_model(spec, frame)
    typical = typical_tickets(spec, frame)
    if model is None:
        return typical
    base, per_unit = model
    return max(base + per_unit * jackpot, 0.5 * typical)
