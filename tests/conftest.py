"""Shared fixtures: an isolated data directory and synthetic operator records."""

from __future__ import annotations

import datetime as dt
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from lotterylab.combinatorics import special_hits_pmf
from lotterylab.games import GameSpec
from lotterylab.schema import main_columns, special_columns


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point the store at an empty temporary data directory."""
    monkeypatch.setenv("LOTTERYLAB_DATA", str(tmp_path))
    return tmp_path


def esp(values, size: int) -> float:
    """Elementary symmetric polynomial of ``values`` of order ``size``."""
    poly = [1.0] + [0.0] * size
    for value in values:
        for j in range(size, 0, -1):
            poly[j] += value * poly[j - 1]
    return poly[size]


def make_payout_frame(
    spec: GameSpec, weights, n_draws: int, seed: int = 0, tickets: float = 5e6
) -> pd.DataFrame:
    """Operator-style records from a crowd that picks numbers with ``weights``.

    Players choose main combinations with probability proportional to the product
    of their numbers' weights (the model the library assumes, computed exactly
    here), specials uniformly. Shared ranks split a fixed slice of sales.
    """
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=float)
    k, pool = spec.main_count, spec.main_max
    total = esp(weights, k)
    start = spec.payout_since or dt.date(2020, 1, 1)
    rows = []
    for i in range(n_draws):
        drawn = np.sort(rng.choice(np.arange(1, pool + 1), k, replace=False))
        special = np.sort(
            rng.choice(
                np.arange(1, spec.special_max + 1), spec.special_count, replace=False
            )
        )
        inside = weights[drawn - 1]
        outside = np.delete(weights, drawn - 1)
        row = {"date": start + dt.timedelta(days=3 * i), "cycle": 1.0}
        row.update(dict(zip(main_columns(spec), drawn.tolist())))
        row.update(dict(zip(special_columns(spec), special.tolist())))
        for rank, tiers in enumerate(spec.rank_tiers, start=1):
            chance = sum(
                esp(inside, m) * esp(outside, k - m) / total * special_hits_pmf(spec, s)
                for m, s in tiers
            )
            winners = rng.poisson(tickets * chance)
            if rank in spec.pari_ranks:
                # 4% of sales to each shared rank, split among its winners
                payout = 0.04 * tickets * spec.price / winners if winners else 0.0
            else:
                payout = spec.price
            row[f"r{rank}_winners"] = float(winners)
            row[f"r{rank}_winners_fr"] = float(winners)
            row[f"r{rank}_payout"] = float(payout)
        row["codes_winners"] = row["codes_payout"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fdj_zip(header: list[str], rows: list[list[str]], encoding: str = "utf-8") -> bytes:
    """A zip-of-CSV in FDJ's layout (semicolon-delimited, trailing ';')."""
    text = ";".join(header) + ";\n" + "".join(";".join(r) + ";\n" for r in rows)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("history.csv", text.encode(encoding))
    return buffer.getvalue()


class FakeSession:
    """Stands in for ``requests``: serves canned bytes per URL."""

    def __init__(self, by_url: dict[str, bytes]):
        self.by_url = by_url
        self.calls: list[str] = []

    def get(self, url, **_kwargs):
        self.calls.append(url)
        content = self.by_url[url]

        class Response:
            def raise_for_status(self):
                return None

        response = Response()
        response.content = content
        return response
