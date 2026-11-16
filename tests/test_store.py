"""Snapshots are immutable, ordered, de-duplicated, and never get worse."""

import datetime as dt

import pandas as pd
import pytest

from lotterylab import games, store

from .conftest import FakeSession, fdj_zip

NY_HEADER = "Draw Date,Winning Numbers,Multiplier\n"


def ny_csv(n: int) -> bytes:
    rows = [
        f"{(dt.date(2024, 1, 1) + dt.timedelta(days=3 * i)):%m/%d/%Y},"
        f"{1 + i % 60} {2 + i % 60} {3 + i % 60} {4 + i % 60} {5 + i % 60} {1 + i % 26},2\n"
        for i in range(n)
    ]
    return (NY_HEADER + "".join(rows)).encode()


def test_data_dir_honours_the_environment(data_dir):
    assert store.data_dir() == str(data_dir)
    assert store.raw_dir("loto") == str(data_dir / "raw" / "loto")


def test_snapshots_sort_newest_first_across_naming_schemes(data_dir):
    folder = data_dir / "raw" / "powerball"
    folder.mkdir(parents=True)
    names = [
        "2025-08-13__ny_d6yy-54nr.csv",
        "2026-08-11__8acd8594.csv",
        "2026-08-11__zz-230412__fdj-b4c49b42.csv",
        "2026-11-16T033616__ny-797e2023.csv",
        "2026-11-16T120000__ny-00000000.csv",
    ]
    for name in names:
        (folder / name).write_bytes(b"x")
    found = [p.rsplit("/", 1)[-1] for p in store.all_snapshots("powerball")]
    assert found == names[::-1]


def test_identical_content_writes_nothing_new(data_dir):
    first = store.store_snapshot("powerball", ny_csv(30), "ny")
    again = store.store_snapshot(
        "powerball", ny_csv(30), "ny", now=dt.datetime(2030, 1, 1)
    )
    assert again == first
    assert len(store.all_snapshots("powerball")) == 1
    newer = store.store_snapshot(
        "powerball", ny_csv(31), "ny", now=dt.datetime(2030, 1, 2)
    )
    assert newer != first and store.newest_snapshot("powerball") == newer


def test_bad_downloads_are_rejected_and_leave_no_file(data_dir):
    store.store_snapshot("powerball", ny_csv(100), "ny")
    with pytest.raises(ValueError, match="truncated"):
        store.store_snapshot("powerball", ny_csv(40), "ny")
    with pytest.raises(ValueError, match="did not parse"):
        store.store_snapshot("powerball", b"<xml>latest draw only</xml>", "ny")
    assert len(store.all_snapshots("powerball")) == 1
    assert len(store.load_canonical("powerball")) == 100


def test_loader_skips_a_broken_newest_snapshot(data_dir):
    good = store.store_snapshot("powerball", ny_csv(20), "ny")
    (data_dir / "raw" / "powerball" / "2099-01-01T000000__ny-deadbeef.csv").write_text(
        "not,a,draw\nfile,at,all\n"
    )
    assert store.newest_snapshot("powerball") != good
    assert len(store.load_canonical("powerball")) == 20


@pytest.mark.parametrize(
    "text,value",
    [
        ("1 234,50", 1234.5),
        ("2000,00", 2000.0),
        ("17", 17.0),
        ("", None),
        ("n/a", None),
    ],
)
def test_fdj_number_parsing(text, value):
    assert store._num(text) == value


EURODREAMS_HEADER = (
    ["annee_numero_de_tirage", "jour_de_tirage", "date_de_tirage"]
    + ["numéro_de_tirage_dans_le_cycle", "date_de_forclusion"]
    + [f"boule_{i}" for i in range(1, 7)]
    + ["numero_dream", "boules_gagnantes_en_ordre_croissant"]
    + [
        col
        for rank in range(1, 7)
        for col in (
            f"nombre_de_gagnant_au_rang{rank}_Euro_Dreams_en_france",
            f"nombre_de_gagnant_au_rang{rank}_Euro_Dreams_en_europe",
            f"rapport_du_rang{rank}_Euro_Dreams",
        )
    ]
    + ["nombre_de_codes_gagnants", "rapport_codes_gagnants", "codes_gagnants"]
)


def eurodreams_row(day: dt.date, balls, dream, europe_rank6="505139") -> list[str]:
    ranks = ["0", "0", "0", "0", "1", "2000,00", "32", "146", "492,80"]
    ranks += ["1342", "5888", "34,20", "20190", "86683", "4,80"]
    ranks += ["117076", europe_rank6, "2,50"]
    head = ["26091", "JEUDI", f"{day:%d/%m/%Y}", "0", "11/02/2027"]
    return head + [str(b) for b in balls] + [str(dream), "-"] + ranks + ["", "", ""]


@pytest.mark.parametrize("encoding", ["utf-8", "latin-1"])
def test_fdj_fetch_keeps_winners_and_payouts(data_dir, encoding):
    rows = [
        eurodreams_row(
            dt.date(2026, 11, 2) + dt.timedelta(days=3 * i), balls, 1 + i % 5
        )
        for i, balls in enumerate([(15, 18, 10, 5, 19, 37), (1, 2, 3, 4, 5, 6)])
    ]
    rows.append(eurodreams_row(dt.date(2026, 11, 9), (7, 8, 9, 10, 11, 12), 3, "0"))
    url = store.FDJ_SOURCES["eurodreams"].urls[0]
    session = FakeSession({url: fdj_zip(EURODREAMS_HEADER, rows, encoding)})

    path = store.fetch_raw("eurodreams", session=session)
    assert session.calls == [url]
    assert store.fetch_raw("eurodreams", session=session) == path  # unchanged

    draws = store.load_canonical("eurodreams")
    assert len(draws) == 3
    assert draws.iloc[0][["m1", "m6", "s1"]].tolist() == [5, 37, 1]

    payouts = store.load_payouts("eurodreams")
    assert payouts["r3_payout"].tolist() == [492.8] * 3
    assert payouts["r6_winners_fr"].tolist() == [117076.0] * 3
    # A Europe-wide count below France's is a missing value, not a real zero.
    assert payouts["r6_winners"].tolist()[:2] == [505139.0] * 2
    assert pd.isna(payouts["r6_winners"].iloc[2])


def test_fdj_format_change_raises(data_dir):
    url = store.FDJ_SOURCES["eurodreams"].urls[0]
    session = FakeSession({url: fdj_zip(["something", "else"], [["1", "2"]])})
    with pytest.raises(ValueError, match="format may have changed"):
        store.fetch_raw("eurodreams", session=session)
    assert store.all_snapshots("eurodreams") == []


def test_payouts_unavailable_for_games_without_them(data_dir):
    with pytest.raises(FileNotFoundError):
        store.load_payouts("powerball")
    assert set(store.FETCHABLE) == set(games.GAMES)
