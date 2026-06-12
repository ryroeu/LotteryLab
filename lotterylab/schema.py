"""Canonical draw schema + tidy DataFrame conversion.

One ``Draw`` per drawing. Balls are stored sorted ascending and validated in-range
and unique at construction time, so every downstream consumer can trust them.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import pandas as pd

from .games import GameSpec


@dataclass(frozen=True)
class Draw:
    """One canonical lottery draw with sorted main and special balls."""

    game: str
    date: _dt.date
    draw_id: str | None
    main: tuple[int, ...]
    special: tuple[int, ...]

    def __post_init__(self):
        """Normalize ball ordering after dataclass initialization."""
        # Enforce sorted-ascending storage regardless of source order.
        object.__setattr__(self, "main", tuple(sorted(self.main)))
        object.__setattr__(self, "special", tuple(sorted(self.special)))


def draws_to_frame(draws: list[Draw], spec: GameSpec) -> pd.DataFrame:
    """Tidy frame: game, date, draw_id, m1..mk, s1..sj."""
    rows = []
    for d in draws:
        row = {"game": d.game, "date": d.date, "draw_id": d.draw_id}
        for j in range(spec.main_count):
            row[f"m{j + 1}"] = d.main[j]
        for j in range(spec.special_count):
            row[f"s{j + 1}"] = d.special[j]
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def frame_to_draws(df: pd.DataFrame, spec: GameSpec) -> list[Draw]:
    """Convert a canonical DataFrame back into ``Draw`` objects."""
    main_cols = [f"m{j + 1}" for j in range(spec.main_count)]
    spec_cols = [f"s{j + 1}" for j in range(spec.special_count)]
    out = []
    for _, row in df.iterrows():
        out.append(
            Draw(
                game=spec.key,
                date=row["date"],
                draw_id=row.get("draw_id"),
                main=tuple(int(row[c]) for c in main_cols),
                special=tuple(int(row[c]) for c in spec_cols),
            )
        )
    return out


def main_columns(spec: GameSpec) -> list[str]:
    """Canonical main-number column names for ``spec``."""
    return [f"m{j + 1}" for j in range(spec.main_count)]


def special_columns(spec: GameSpec) -> list[str]:
    """Canonical special-number column names for ``spec``."""
    return [f"s{j + 1}" for j in range(spec.special_count)]
