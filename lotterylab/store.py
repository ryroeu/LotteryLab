"""Data store — immutable raw snapshots, derived canonical cache.

Source CSVs live under ``data/raw/<game>/<snapshot>.csv`` and are NEVER overwritten
(the old scripts downloaded and clobbered the committed file in place, destroying
provenance). ``load_canonical`` reads the newest snapshot through the game's
adapter, filters to the current matrix era, validates, and returns a tidy frame.
"""

from __future__ import annotations

import datetime as _dt
import glob
import os

import pandas as pd

from . import adapters
from .games import get
from .schema import Draw, draws_to_frame
from .validate import InvalidTicket, validate_ticket

# Repo root = two levels up from this file (lotterylab/store.py -> repo/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
CACHE_DIR = os.path.join(ROOT, "data", "cache")

# Project-wide floor: only ever use draws from 2018-01-01 onward. Combined with
# each game's main_matrix_since (whichever is later) so the uniform "2018 to present"
# cut never re-introduces a stale matrix (every game's current matrix predates 2018).
MIN_DATE = _dt.date(2018, 1, 1)

# Download URLs for the games that expose a public CSV. fetch_raw writes a new
# timestamped snapshot; it never touches existing ones.
FETCH_URLS = {
    "powerball": "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD",
    "megamillions": "https://data.ny.gov/api/views/5xaw-6ayf/rows.csv?accessType=DOWNLOAD",
    "euromillions": "https://www.national-lottery.co.uk/results/euromillions/draw-history/csv",
}


def newest_snapshot(game: str) -> str | None:
    files = sorted(glob.glob(os.path.join(RAW_DIR, game, "*.csv")))
    return files[-1] if files else None


def fetch_raw(game: str, *, today: _dt.date | None = None) -> str:
    """Download a fresh snapshot to data/raw/<game>/. Returns the new path.

    Network side-effect; not exercised by the offline test suite.
    """
    import hashlib

    import requests

    if game not in FETCH_URLS:
        raise ValueError(f"No public fetch URL configured for {game!r}.")
    today = today or _dt.date.today()
    resp = requests.get(FETCH_URLS[game], timeout=60)
    resp.raise_for_status()
    digest = hashlib.sha256(resp.content).hexdigest()[:8]
    os.makedirs(os.path.join(RAW_DIR, game), exist_ok=True)
    path = os.path.join(RAW_DIR, game, f"{today.isoformat()}__{digest}.csv")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(resp.content)
    return path


def load_canonical(
    game: str,
    *,
    modern_only: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Load the newest snapshot as a validated, tidy, chronologically-sorted frame.

    ``modern_only`` filters to draws on/after ``spec.main_matrix_since`` and drops
    any row that is not a valid draw under the current matrix (a robust safety net
    for mixed-era files).
    """
    spec = get(game)
    path = newest_snapshot(game)
    if path is None:
        raise FileNotFoundError(
            f"No raw snapshot for {game!r} in {os.path.join(RAW_DIR, game)}. "
            f"Run fetch_raw({game!r}) or add a CSV."
        )
    raw = pd.read_csv(path)
    draws = adapters.parse(game, raw)

    # Effective cutoff = the later of the 2018 floor and the game's matrix change.
    cutoff = None
    if modern_only:
        cutoff = MIN_DATE
        if spec.main_matrix_since and spec.main_matrix_since > cutoff:
            cutoff = spec.main_matrix_since

    kept: list[Draw] = []
    dropped_era = dropped_invalid = 0
    for d in draws:
        if cutoff and d.date < cutoff:
            dropped_era += 1
            continue
        try:
            validate_ticket(d.main, d.special, spec)
        except InvalidTicket:
            dropped_invalid += 1
            continue
        kept.append(d)

    if verbose:
        print(
            f"[{game}] loaded {len(kept)} draws from {os.path.basename(path)} "
            f"(dropped {dropped_era} pre-{cutoff} + "
            f"{dropped_invalid} invalid-under-current-matrix)"
        )
    return draws_to_frame(kept, spec)


def write_cache(game: str, df: pd.DataFrame) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{game}.csv")
    df.to_csv(path, index=False)
    return path
