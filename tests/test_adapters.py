"""Each adapter parses its real snapshot into valid, current-matrix draws."""

import datetime as dt

from lotterylab import games
from lotterylab.store import MIN_DATE, load_canonical
from lotterylab.validate import validate_ticket


def test_powerball_loads_modern_matrix_only():
    """Powerball loads only current-matrix draws after the 2018 floor."""
    df = load_canonical("powerball")
    spec = games.get("powerball")
    assert len(df) > 800  # ~999 draws from 2018-01-01 onward
    # every retained draw must be valid under the current 5/69 + 1/26 matrix
    for _, row in df.iterrows():
        main = tuple(int(row[f"m{i+1}"]) for i in range(spec.main_count))
        special = (int(row["s1"]),)
        validate_ticket(main, special, spec)
    # 2018 floor takes effect (later than the 2015 matrix change)
    assert df["date"].min() >= MIN_DATE
    assert df["date"].min() >= spec.main_matrix_since


def test_all_games_respect_2018_floor():
    """Every canonical history respects the project-wide 2018 floor."""
    for game in ["powerball", "megamillions", "euromillions", "eurodreams"]:
        df = load_canonical(game)
        assert df["date"].min() >= MIN_DATE


def test_megamillions_loads_valid_current_matrix():
    """Mega Millions draws should be valid under the current matrix."""
    df = load_canonical("megamillions")
    spec = games.get("megamillions")
    assert len(df) > 500  # ~844 draws from 2018-01-01 onward
    # every retained draw valid under the current 5/70 + 1/24 matrix
    for _, row in df.iterrows():
        main = tuple(int(row[f"m{i+1}"]) for i in range(spec.main_count))
        validate_ticket(main, (int(row["s1"]),), spec)


def test_euromillions_golden_row():
    """EuroMillions canonical history should contain a known stable row."""
    df = load_canonical("euromillions")
    # FDJ full history (2018+). The 2025-08-15 draw is a stable interior row:
    # balls 13 30 35 36 40, stars 2 6 (cross-checked against national-lottery.co.uk).
    row = df[df["date"] == dt.date(2025, 8, 15)]
    assert len(row) == 1
    row = row.iloc[0]
    assert [row[f"m{i+1}"] for i in range(5)] == [13, 30, 35, 36, 40]
    assert sorted([row["s1"], row["s2"]]) == [2, 6]
    assert len(df) > 500  # full FDJ history, not the old 52-row snapshot
    assert df["date"].min() >= MIN_DATE


def test_eurodreams_golden_row():
    """EuroDreams canonical history should contain a known stable row."""
    df = load_canonical("eurodreams")
    # FDJ full history (2023-11+). 2024-09-26 is a stable interior row,
    # cross-checked against the original Kaggle snapshot.
    row = df[df["date"] == dt.date(2024, 9, 26)]
    assert len(row) == 1
    row = row.iloc[0]
    assert [row[f"m{i+1}"] for i in range(6)] == [9, 19, 21, 31, 32, 39]
    assert row["s1"] == 2
    assert len(df) > 200  # full FDJ history, not the old 94-row snapshot
