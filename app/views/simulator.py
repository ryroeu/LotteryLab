"""Time & Variance — make the abstract odds *felt*."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app import shared
from lotterylab import games
from lotterylab.simulate import time_to_match3, variance_samples

st.title("⏳ Time & Variance")
st.caption(
    "How long until a single ticket matches 3 — and what a season of playing really feels like."
)

# --- Waiting time ------------------------------------------------------------------

st.subheader("How long until a 3-match?")
waits = [time_to_match3(spec) for spec in games.all_games()]
wait_df = pd.DataFrame(
    {
        "Game": [w.game for w in waits],
        "Match-3 odds": [f"1 in {w.one_in:,}" for w in waits],
        "Draws / week": [w.draws_per_week for w in waits],
        "Expected wait (draws)": [round(w.expected_draws) for w in waits],
        "Expected wait (years)": [w.expected_years for w in waits],
        "Median wait (years)": [w.median_years for w in waits],
    }
)
st.dataframe(
    wait_df,
    hide_index=True,
    width="stretch",
    column_config={
        "Expected wait (years)": st.column_config.NumberColumn(format="%.1f"),
        "Median wait (years)": st.column_config.NumberColumn(format="%.2f"),
    },
)
st.caption(
    "One ticket per draw. EuroDreams' median wait is ~3 months; Powerball's is "
    "over a year — that's what *an order of magnitude friendlier* means in practice."
)

st.divider()

# --- Variance simulator --------------------------------------------------------------

st.subheader("A season of playing, simulated honestly")
st.caption(
    "Monte-Carlo a uniform one-ticket-per-draw player over many independent seasons, "
    "using the exact tier probabilities. Net = winnings − spend."
)

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    game = shared.game_selector()
with c2:
    draws = st.slider(
        "Draws per season",
        26,
        520,
        104,
        step=26,
        help="104 ≈ a year at 2 draws/week",
    )
with c3:
    seasons = st.slider("Seasons", 500, 10_000, 2000, step=500)

spec = games.get(game)


@st.cache_data(show_spinner="Simulating seasons…")
def run_variance(game_key: str, n_draws: int, n_seasons: int):
    """Simulate repeated one-ticket seasons for a game."""
    return variance_samples(
        games.get(game_key), draws=n_draws, n_seasons=n_seasons, seed=0
    )


nets, any3_fraction = run_variance(game, draws, seasons)
spend = draws * spec.price

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Season spend", shared.fmt_money(spend, spec.currency, 0), border=True)
m2.metric(
    "Mean net", shared.fmt_money(float(nets.mean()), spec.currency, 0), border=True
)
m3.metric(
    "Median net",
    shared.fmt_money(float(np.percentile(nets, 50)), spec.currency, 0),
    border=True,
)
m4.metric(
    "Best season", shared.fmt_money(float(nets.max()), spec.currency, 0), border=True
)
m5.metric(
    "Seasons with a 3-match",
    f"{any3_fraction:.1%}",
    help="Fraction of seasons containing at least one ≥3-main hit.",
    border=True,
)

hist_df = pd.DataFrame({"net": nets})
bars = (
    alt.Chart(hist_df)
    .mark_bar(color=shared.ACCENT, opacity=0.85)
    .encode(
        x=alt.X(
            "net:Q",
            bin=alt.Bin(maxbins=60),
            title=f"Net result per season ({spec.currency})",
        ),
        y=alt.Y("count():Q", title="Seasons"),
        tooltip=[alt.Tooltip("count():Q", title="Seasons")],
    )
)
break_even = (
    alt.Chart(pd.DataFrame({"x": [0.0]}))
    .mark_rule(color="#FF6B6B", strokeDash=[6, 4], size=2)
    .encode(x="x:Q")
)
st.altair_chart((bars + break_even).properties(height=300), width="stretch")
st.caption(
    f"Dashed line = break-even. Across {seasons:,} simulated seasons "
    f"[p05 {shared.fmt_money(float(np.percentile(nets, 5)), spec.currency, 0)} · "
    f"p95 {shared.fmt_money(float(np.percentile(nets, 95)), spec.currency, 0)}] — "
    "the mass sits firmly left of it, with a long thin lucky tail. That asymmetry *is* the lottery."
)

shared.footer()
