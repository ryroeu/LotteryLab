"""Shared plumbing for the Streamlit app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from lotterylab import games
from lotterylab.backtest import BacktestResult, backtest
from lotterylab.store import SNAPSHOT_PARSE_ERRORS, load_canonical
from lotterylab.strategy import get_strategy
from lotterylab.synth import synth_history

GAME_KEYS: list[str] = list(games.GAMES)

CURRENCY_SYMBOL = {"USD": "$", "EUR": "€"}
DATA_LOAD_ERRORS = SNAPSHOT_PARSE_ERRORS

DISCLAIMER = (
    "Lottery Lab is for fun and learning. Nothing here improves the odds of "
    "winning, because nothing can. The expected value of every ticket is well "
    "under its price — please gamble responsibly."
)


def symbol(currency: str) -> str:
    """Return the display symbol for a currency code."""
    return CURRENCY_SYMBOL.get(currency, currency + " ")


def fmt_money(x: float, currency: str, decimals: int = 2) -> str:
    """Format a currency amount for app display."""
    return f"{symbol(currency)}{x:,.{decimals}f}"


def fmt_one_in(p: float) -> str:
    """Format a probability as a one-in-N odds string."""
    if p <= 0:
        return "never"
    return f"1 in {round(1 / p):,}"


def game_selector(default: str = "eurodreams", key: str = "game") -> str:
    """A selectbox over the game registry; returns the game key."""
    idx = GAME_KEYS.index(default) if default in GAME_KEYS else 0
    return st.selectbox(
        "Game",
        GAME_KEYS,
        index=idx,
        format_func=lambda k: games.get(k).name,
        key=key,
    )


def synth_controls(default_n: int = 1500, max_n: int = 50_000) -> tuple[bool, int]:
    """The 'use provably-fair synthetic draws' toggle + size slider."""
    synth = st.toggle(
        "Synthetic fair draws",
        value=False,
        help=(
            "Replace real history with provably-fair generated draws. Because "
            "they are uniform by construction, every strategy MUST land on the "
            "chance line — a self-checking ground truth (and it works offline)."
        ),
    )
    n = default_n
    if synth:
        n = st.slider(
            "Synthetic draws", min_value=500, max_value=max_n, value=default_n, step=500
        )
    return synth, n


@st.cache_data(show_spinner=False)
def load_history(game: str, synth: bool = False, synth_n: int = 1500) -> pd.DataFrame:
    """Draw history as a tidy frame — real snapshots or synthetic fair draws."""
    spec = games.get(game)
    if synth:
        return synth_history(spec, synth_n, seed=0)
    return load_canonical(game)


@st.cache_data(show_spinner=False)
def history_status(game: str) -> dict | None:
    """Draw count + date range of the loadable history, or None if unavailable."""
    try:
        hist = load_history(game)
    except DATA_LOAD_ERRORS:
        return None
    return {
        "draws": len(hist),
        "first": hist["date"].iloc[0],
        "last": hist["date"].iloc[-1],
    }


def require_history(game: str, synth: bool, synth_n: int) -> pd.DataFrame:
    """Load history for a page, or show the load error and stop the app."""
    spec = games.get(game)
    try:
        hist = load_history(game, synth, synth_n)
    except DATA_LOAD_ERRORS as error:
        st.error(f"Could not load history for {spec.name}: {error}")
        st.stop()
    data_source_caption(game, synth, synth_n)
    return hist


@st.cache_data(show_spinner="Walking the backtest forward…")
def run_backtest(
    game: str, strategy: str, n_tickets: int, synth: bool, synth_n: int
) -> BacktestResult:
    """Run and cache one walk-forward strategy backtest."""
    spec = games.get(game)
    hist = load_history(game, synth, synth_n)
    return backtest(get_strategy(strategy), hist, spec, n_tickets=n_tickets, seed=1)


def data_source_caption(game: str, synth: bool, synth_n: int) -> None:
    """One consistent line under each page's controls saying what data is in use."""
    if synth:
        st.caption(
            f"Using **{synth_n:,} synthetic fair draws** (uniform by construction)."
        )
    else:
        status = history_status(game)
        if status:
            st.caption(
                f"Using **real history**: {status['draws']:,} draws, "
                f"{status['first']} → {status['last']}."
            )


def footer() -> None:
    """Render the shared page footer."""
    st.divider()
    st.caption(f"🎰 {DISCLAIMER}")
