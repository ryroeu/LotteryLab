"""Each adapter parses its real snapshot into valid, current-matrix draws."""

import datetime as dt

from lotterylab import games
from lotterylab.store import MIN_DATE, load_canonical
from lotterylab.validate import validate_ticket


def test_powerball_loads_modern_matrix_only():
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
    for game in ["powerball", "euromillions", "eurodreams"]:
        df = load_canonical(game)
        assert df["date"].min() >= MIN_DATE


def test_euromillions_golden_row():
    df = load_canonical("euromillions")
    # newest snapshot's latest draw: 2025-08-15, balls 13 30 35 36 40, stars 2 6
    last = df.iloc[-1]
    assert last["date"] == dt.date(2025, 8, 15)
    assert [last[f"m{i+1}"] for i in range(5)] == [13, 30, 35, 36, 40]
    assert sorted([last["s1"], last["s2"]]) == [2, 6]


def test_eurodreams_golden_row():
    df = load_canonical("eurodreams")
    last = df.iloc[-1]
    assert last["date"] == dt.date(2024, 9, 26)
    assert [last[f"m{i+1}"] for i in range(6)] == [9, 19, 21, 31, 32, 39]
    assert last["s1"] == 2
