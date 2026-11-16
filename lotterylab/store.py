"""Data store — immutable raw snapshots of official draw history.

Source CSVs live under ``<data dir>/raw/<game>/<snapshot>.csv`` and are NEVER
overwritten, so every result can be traced to the file it came from.
``load_canonical`` reads the newest snapshot through the
game's adapter, filters to the current matrix era, validates, and returns a tidy
frame. ``load_payouts`` returns the per-rank winner counts and prize amounts that
the FDJ history files publish alongside the balls.

The data directory is, in order: ``$LOTTERYLAB_DATA``; the ``data/`` folder next to
the package when running from a source checkout; otherwise ``~/.lotterylab/data``
(so an installed ``lotterylab`` command still has somewhere to fetch into).
"""

from __future__ import annotations

import csv as _csv
import datetime as _dt
import glob
import hashlib
import io
import os
import re
import warnings
import zipfile
from dataclasses import dataclass

import pandas as pd
import requests

from . import adapters
from .games import GameSpec, get
from .schema import Draw, draws_to_frame, main_columns, special_columns
from .validate import InvalidTicket, validate_ticket

_UA = {"User-Agent": "Mozilla/5.0 (compatible; lotterylab/0.2)"}
SNAPSHOT_PARSE_ERRORS = (
    KeyError,
    OSError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
)

# A fresh download must keep at least this share of the draws the newest snapshot
# on disk already has, or it is treated as truncated and discarded.
MIN_KEEP_FRACTION = 0.9

# Single-CSV download URLs (NY Open Data).
FETCH_URLS = {
    "powerball": "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD",
    "megamillions": "https://data.ny.gov/api/views/5xaw-6ayf/rows.csv?accessType=DOWNLOAD",
}

_FDJ_DOC = "https://www.sto.api.fdj.fr/anonymous/service-draw-info/v3/documentations/"


def _fdj(*suffixes: str) -> tuple[str, ...]:
    return tuple(f"{_FDJ_DOC}1a2b3c4d-9876-4562-b3fc-2c963f66{s}" for s in suffixes)


@dataclass(frozen=True)
class FdjSource:
    """How one game's FDJ history files map onto a combined snapshot.

    FDJ (the French operator) publishes official history split by rule-era; the
    current-era document is updated in place after every draw. ``rank_label`` is the
    infix FDJ puts in its per-rank columns (``nombre_de_gagnant_au_rang1_<label>_en_
    france``); Loto has none.
    """

    urls: tuple[str, ...]
    fdj_special_cols: tuple[str, ...]
    rank_label: str | None
    date_col: str
    main_cols: tuple[str, ...]
    special_cols: tuple[str, ...]


FDJ_SOURCES: dict[str, FdjSource] = {
    # Star pool changed over the years, so EuroMillions is chunked into eras;
    # together these cover 2016-09 to present.
    "euromillions": FdjSource(
        urls=_fdj("afc6", "afd6", "afe6"),
        fdj_special_cols=("etoile_1", "etoile_2"),
        rank_label="Euro_Millions",
        date_col="DrawDate",
        main_cols=("Ball 1", "Ball 2", "Ball 3", "Ball 4", "Ball 5"),
        special_cols=("Lucky Star 1", "Lucky Star 2"),
    ),
    # EuroDreams launched 2023-11 with one stable matrix: a single live file.
    "eurodreams": FdjSource(
        urls=_fdj("afa5"),
        fdj_special_cols=("numero_dream",),
        rank_label="Euro_Dreams",
        date_col="Date",
        main_cols=tuple(f"Number {i}" for i in range(1, 7)),
        special_cols=("Dream Number",),
    ),
    # Loto has been 5/49 + 1/10 since 2008-10; four files cover it to present.
    "loto": FdjSource(
        urls=_fdj("afm6", "afn6", "afo6", "afp6"),
        fdj_special_cols=("numero_chance",),
        rank_label=None,
        date_col="Date",
        main_cols=tuple(f"Ball {i}" for i in range(1, 6)),
        special_cols=("Chance",),
    ),
}

FETCHABLE = tuple(FETCH_URLS) + tuple(FDJ_SOURCES)


# --- Locations -------------------------------------------------------------------


def data_dir() -> str:
    """Directory holding ``raw/<game>/`` snapshots (see the module docstring)."""
    override = os.environ.get("LOTTERYLAB_DATA")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    checkout = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    if os.path.isdir(checkout):
        return checkout
    return os.path.join(os.path.expanduser("~"), ".lotterylab", "data")


def raw_dir(game: str | None = None) -> str:
    """The raw-snapshot directory, optionally for one game."""
    base = os.path.join(data_dir(), "raw")
    return os.path.join(base, game) if game else base


_SNAPSHOT_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:T(\d{6})|__zz-(\d{6}))?")


def _snapshot_key(path: str) -> tuple[str, str, str]:
    """Sort key: (date, time-of-day or '', name). Understands both name schemes
    (``2026-11-16T093000__fdj-ab12cd34.csv`` and the older ``…__zz-093000__…``)."""
    name = os.path.basename(path)
    match = _SNAPSHOT_STAMP.match(name)
    if not match:
        return ("", "", name)
    return (match.group(1), match.group(2) or match.group(3) or "", name)


def all_snapshots(game: str) -> list[str]:
    """All raw snapshots for a game, newest first."""
    paths = glob.glob(os.path.join(raw_dir(game), "*.csv"))
    return sorted(paths, key=_snapshot_key, reverse=True)


def newest_snapshot(game: str) -> str | None:
    """Newest raw snapshot path for a game, if one exists."""
    snaps = all_snapshots(game)
    return snaps[0] if snaps else None


# --- FDJ parsing -----------------------------------------------------------------


def _num(value) -> float | None:
    """Parse an FDJ number ('1 234,50', '2000,00', '') to float, or None."""
    if value is None:
        return None
    text = str(value).strip().replace(" ", "").replace(" ", "")
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def _fdj_date(value) -> _dt.date:
    text = str(value).strip()
    for fmt in ("%d/%m/%Y", "%Y%m%d", "%d/%m/%y"):
        try:
            return _dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised FDJ date {text!r}")


def _read_fdj_table(content: bytes) -> pd.DataFrame:
    """One FDJ file (zip-of-CSV or raw CSV) as an all-strings frame.

    Semicolon-delimited; older files are latin-1, newer ones UTF-8.
    ``index_col=False`` because the rows carry a trailing ';' that would otherwise
    make pandas adopt the first column as the index and shift every column left.
    """
    raw = content
    if content[:2] == b"PK":  # zip magic
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            raw = archive.read(archive.namelist()[0])
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    with warnings.catch_warnings():
        # One era file has a spare trailing ';' on every data row; pandas warns
        # that it drops that (empty) extra field.
        warnings.simplefilter("ignore", pd.errors.ParserWarning)
        return pd.read_csv(
            io.StringIO(text),
            sep=";",
            index_col=False,
            dtype=str,
            keep_default_na=False,
        )


def _parse_fdj(content: bytes, n_main: int, source: FdjSource) -> list[dict]:
    """Parse one FDJ draw file to draw dicts (balls + per-rank winners/payouts).

    Returns [] if the ball columns are absent. Rank columns are optional and vary
    by era; whatever is present is kept.
    """
    df = _read_fdj_table(content)
    main_cols = [f"boule_{i}" for i in range(1, n_main + 1)]
    need = ["date_de_tirage", *main_cols, *source.fdj_special_cols]
    if not all(c in df.columns for c in need):
        return []

    label = f"_{source.rank_label}" if source.rank_label else ""
    rank_cols: dict[int, tuple[str | None, str | None, str | None]] = {}
    for rank in range(1, 21):
        if label:
            total = f"nombre_de_gagnant_au_rang{rank}{label}_en_europe"
            french = f"nombre_de_gagnant_au_rang{rank}{label}_en_france"
        else:
            total = french = f"nombre_de_gagnant_au_rang{rank}"
        payout = f"rapport_du_rang{rank}{label}"
        if payout not in df.columns:
            break
        rank_cols[rank] = (
            total if total in df.columns else None,
            french if french in df.columns else None,
            payout,
        )
    cycle_col = next((c for c in df.columns if "de_tirage_dans_le_cycle" in c), None)
    has_codes = "nombre_de_codes_gagnants" in df.columns

    rows = []
    for record in df.to_dict("records"):
        try:
            date = _fdj_date(record["date_de_tirage"])
            main = sorted(int(record[c]) for c in main_cols)
            special = sorted(int(record[c]) for c in source.fdj_special_cols)
        except (ValueError, TypeError):
            continue
        ranks = {}
        for rank, (total_col, french_col, payout_col) in rank_cols.items():
            total = _num(record[total_col]) if total_col else None
            french = _num(record[french_col]) if french_col else None
            # Some files leave the Europe-wide count at 0 while France has
            # winners: that is a missing value, not zero.
            if total is not None and french is not None and total < french:
                total = None
            ranks[rank] = (total, french, _num(record[payout_col]))
        rows.append(
            {
                "date": date,
                "main": main,
                "special": special,
                "draw": str(record.get("annee_numero_de_tirage", "")).strip(),
                "cycle": _num(record[cycle_col]) if cycle_col else None,
                "ranks": ranks,
                "codes_winners": (
                    _num(record["nombre_de_codes_gagnants"]) if has_codes else None
                ),
                "codes_payout": (
                    _num(record.get("rapport_codes_gagnants")) if has_codes else None
                ),
            }
        )
    return rows


def _download_fdj(game: str, session=None) -> dict:
    """Download a game's FDJ files and merge their draws, de-duped by draw date."""
    source = FDJ_SOURCES[game]
    spec = get(game)
    http = session or requests
    by_date: dict[_dt.date, dict] = {}
    for url in source.urls:
        resp = http.get(url, headers=_UA, timeout=60)
        resp.raise_for_status()
        rows = _parse_fdj(resp.content, spec.main_count, source)
        if not rows:
            raise ValueError(
                f"FDJ source produced no usable draws; its format may have changed: {url}"
            )
        for row in rows:
            by_date[row["date"]] = row
    return by_date


def _cell(value) -> str:
    if value is None:
        return ""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _fdj_snapshot_bytes(game: str, by_date: dict) -> bytes:
    """Merged FDJ draws as one CSV in the game's adapter layout, plus rank columns."""
    source = FDJ_SOURCES[game]
    n_ranks = len(get(game).rank_tiers)
    header = [source.date_col, *source.main_cols, *source.special_cols]
    header += ["DrawNumber", "cycle"]
    for rank in range(1, n_ranks + 1):
        header += [f"r{rank}_winners", f"r{rank}_winners_fr", f"r{rank}_payout"]
    header += ["codes_winners", "codes_payout"]

    buffer = io.StringIO(newline="")
    writer = _csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in sorted(by_date.values(), key=lambda r: r["date"]):
        line = [row["date"].isoformat(), *row["main"], *row["special"]]
        line += [row["draw"], _cell(row["cycle"])]
        for rank in range(1, n_ranks + 1):
            line += [_cell(v) for v in row["ranks"].get(rank, (None, None, None))]
        line += [_cell(row["codes_winners"]), _cell(row["codes_payout"])]
        writer.writerow(line)
    return buffer.getvalue().encode("utf-8")


# --- Fetching --------------------------------------------------------------------


def _count_draws(game: str, content: bytes) -> int:
    """How many draws these snapshot bytes yield through the adapter (0 if none)."""
    try:
        return len(adapters.parse(game, pd.read_csv(io.BytesIO(content))))
    except SNAPSHOT_PARSE_ERRORS:
        return 0


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def store_snapshot(
    game: str,
    content: bytes,
    source: str,
    *,
    now: _dt.datetime | None = None,
    validate: bool = True,
) -> str:
    """Write ``content`` as a new immutable snapshot and return its path.

    Returns the existing newest snapshot instead when the content is identical to
    it, so refreshing on a quiet day adds nothing. With ``validate`` the content
    must parse to draws, and must not have lost draws relative to the newest
    snapshot on disk — a changed, expired, or truncated source can never leave a
    worse snapshot in front of a good one.
    """
    now = now or _dt.datetime.now()
    newest = newest_snapshot(game)
    previous = _read_bytes(newest) if newest else None
    if previous == content:
        return newest

    if validate:
        n_new = _count_draws(game, content)
        if n_new == 0:
            raise ValueError(
                f"Fetched {game!r} data did not parse to any draws — the source "
                f"format may have changed. Nothing written."
            )
        n_old = _count_draws(game, previous) if previous else 0
        if n_new < MIN_KEEP_FRACTION * n_old:
            raise ValueError(
                f"Fetched {game!r} data has {n_new} draws but the newest snapshot "
                f"has {n_old} — it looks truncated. Nothing written."
            )

    digest = hashlib.sha256(content).hexdigest()[:8]
    folder = raw_dir(game)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{now:%Y-%m-%dT%H%M%S}__{source}-{digest}.csv")
    with open(path, "xb") as output:
        output.write(content)
    return path


def fetch_raw(
    game: str,
    *,
    now: _dt.datetime | None = None,
    validate: bool = True,
    session=None,
) -> str:
    """Download fresh official history and store it as a snapshot. Returns its path
    (the existing newest snapshot when nothing changed).

    Network side-effect. ``session`` is anything with a ``requests``-style ``get``.
    """
    if game in FDJ_SOURCES:
        content = _fdj_snapshot_bytes(game, _download_fdj(game, session))
        source = "fdj"
    elif game in FETCH_URLS:
        resp = (session or requests).get(FETCH_URLS[game], headers=_UA, timeout=60)
        resp.raise_for_status()
        content, source = resp.content, "ny"
    else:
        raise ValueError(f"No fetch source configured for {game!r}.")
    return store_snapshot(game, content, source, now=now, validate=validate)


# --- Loading ---------------------------------------------------------------------


def load_canonical(game: str, *, modern_only: bool = True) -> pd.DataFrame:
    """Load the newest snapshot as a validated, tidy, chronologically-sorted frame.

    ``modern_only`` keeps draws on/after the game's current-matrix date and drops
    any row not valid under the current matrix (a safety net for mixed-era files).
    Snapshots are tried newest-first; a snapshot that fails to parse or yields zero
    draws is skipped, so one bad file never breaks the pipeline.
    """
    spec = get(game)
    snapshots = all_snapshots(game)
    if not snapshots:
        raise FileNotFoundError(
            f"No raw snapshot for {game!r} in {raw_dir(game)}. "
            f"Run `lotterylab fetch {game}` or add a CSV."
        )
    cutoff = spec.matrix_since if modern_only else None

    for path in snapshots:
        try:
            draws = adapters.parse(game, pd.read_csv(path))
        except SNAPSHOT_PARSE_ERRORS:
            continue
        kept: list[Draw] = []
        for d in draws:
            if cutoff and d.date < cutoff:
                continue
            try:
                validate_ticket(d.main, d.special, spec)
            except InvalidTicket:
                continue
            kept.append(d)
        if kept:
            return draws_to_frame(kept, spec)

    raise ValueError(
        f"No usable snapshot for {game!r}: all {len(snapshots)} file(s) failed to "
        f"parse into draws. The source format may have changed."
    )


def payout_columns(spec: GameSpec) -> list[str]:
    """Per-rank columns a payout frame carries for ``spec``."""
    cols = []
    for rank in range(1, len(spec.rank_tiers) + 1):
        cols += [f"r{rank}_winners", f"r{rank}_winners_fr", f"r{rank}_payout"]
    return cols


def load_payouts(game: str, *, current_rules_only: bool = True) -> pd.DataFrame:
    """Per-draw winner counts and prize amounts by rank, from the newest snapshot
    that carries them.

    Columns: ``date``, ``cycle`` (draw number within the rollover cycle),
    ``m1..mk``/``s1..sj`` (the balls), ``r<rank>_winners`` (all participating
    countries), ``r<rank>_winners_fr`` (France), ``r<rank>_payout`` (per winner),
    ``codes_winners``/``codes_payout`` (raffle codes). ``current_rules_only`` keeps
    draws since the game's present prize structure (``spec.payout_since``);
    otherwise every draw whose ranks mean what they mean today
    (``spec.ranks_since``) — winner counts stay comparable across prize changes.

    Raises ``FileNotFoundError`` when the game publishes no such data (the US
    sources) or no snapshot has it yet (fetch a fresh one).
    """
    spec = get(game)
    source = FDJ_SOURCES.get(game)
    if source is None or not spec.rank_tiers:
        raise FileNotFoundError(f"{spec.name} has no per-rank payout data source.")
    wanted = payout_columns(spec)
    for path in all_snapshots(game):
        try:
            raw = pd.read_csv(path)
        except SNAPSHOT_PARSE_ERRORS:
            continue
        if "r1_payout" not in raw.columns:
            continue
        frame = pd.DataFrame({"date": pd.to_datetime(raw[source.date_col]).dt.date})
        frame["cycle"] = raw.get("cycle")
        for name, col in zip(main_columns(spec), source.main_cols):
            frame[name] = raw[col].astype(int)
        for name, col in zip(special_columns(spec), source.special_cols):
            frame[name] = raw[col].astype(int)
        for col in [*wanted, "codes_winners", "codes_payout"]:
            frame[col] = pd.to_numeric(raw.get(col), errors="coerce")
        since = spec.payout_since if current_rules_only else spec.ranks_since
        if since:
            frame = frame[frame["date"] >= since]
        return frame.sort_values("date").reset_index(drop=True)
    raise FileNotFoundError(
        f"No {spec.name} snapshot with payout columns yet. "
        f"Run `lotterylab fetch {game}`."
    )
