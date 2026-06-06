"""Per-game CSV adapters: messy source format -> list[Draw].

Each adapter absorbs one game's quirks (column names, date formats, the
space-packed "Winning Numbers" string) and returns canonical, validated draws.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd

from .games import GameSpec, get
from .schema import Draw


def _parse_date(value, fmts: list[str]) -> _dt.date:
    for fmt in fmts:
        try:
            return _dt.datetime.strptime(str(value).strip(), fmt).date()
        except ValueError:
            continue
    # Last resort: let pandas infer (handles ISO cleanly).
    return pd.to_datetime(value).date()


def parse_powerball(raw: pd.DataFrame, spec: GameSpec) -> list[Draw]:
    draws = []
    for _, row in raw.iterrows():
        nums = [int(x) for x in str(row["Winning Numbers"]).split()]
        if len(nums) != spec.main_count + spec.special_count:
            continue  # malformed row
        main = nums[: spec.main_count]
        special = nums[spec.main_count :]
        draws.append(
            Draw(
                game=spec.key,
                date=_parse_date(row["Draw Date"], ["%Y-%m-%d", "%m/%d/%Y"]),
                draw_id=None,
                main=tuple(main),
                special=tuple(special),
            )
        )
    return draws


# Mega Millions uses the same NY layout but a separate "Mega Ball" column.
def parse_megamillions(raw: pd.DataFrame, spec: GameSpec) -> list[Draw]:
    draws = []
    for _, row in raw.iterrows():
        main = [int(x) for x in str(row["Winning Numbers"]).split()][: spec.main_count]
        special = [int(row["Mega Ball"])]
        if len(main) != spec.main_count:
            continue
        draws.append(
            Draw(
                game=spec.key,
                date=_parse_date(row["Draw Date"], ["%Y-%m-%d", "%m/%d/%Y"]),
                draw_id=None,
                main=tuple(main),
                special=tuple(special),
            )
        )
    return draws


def parse_euromillions(raw: pd.DataFrame, spec: GameSpec) -> list[Draw]:
    main_cols = ["Ball 1", "Ball 2", "Ball 3", "Ball 4", "Ball 5"]
    star_cols = ["Lucky Star 1", "Lucky Star 2"]
    draws = []
    for _, row in raw.iterrows():
        draws.append(
            Draw(
                game=spec.key,
                date=_parse_date(row["DrawDate"], ["%d-%b-%Y", "%Y-%m-%d"]),
                draw_id=str(row["DrawNumber"]) if "DrawNumber" in raw.columns else None,
                main=tuple(int(row[c]) for c in main_cols),
                special=tuple(int(row[c]) for c in star_cols),
            )
        )
    return draws


def parse_eurodreams(raw: pd.DataFrame, spec: GameSpec) -> list[Draw]:
    main_cols = [f"Number {i}" for i in range(1, 7)]
    draws = []
    for _, row in raw.iterrows():
        draws.append(
            Draw(
                game=spec.key,
                date=_parse_date(row["Date"], ["%Y-%m-%d", "%d-%b-%Y"]),
                draw_id=None,
                main=tuple(int(row[c]) for c in main_cols),
                special=(int(row["Dream Number"]),),
            )
        )
    return draws


ADAPTERS = {
    "powerball": parse_powerball,
    "megamillions": parse_megamillions,
    "euromillions": parse_euromillions,
    "eurodreams": parse_eurodreams,
}


def parse(game: str, raw: pd.DataFrame) -> list[Draw]:
    spec = get(game)
    adapter = ADAPTERS[game]
    return adapter(raw, spec)
