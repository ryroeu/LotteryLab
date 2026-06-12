"""The Odds — landing page: the premise, the exact numbers, the three real ideas."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import shared
from lotterylab import games
from lotterylab.combinatorics import odds_table

st.title("🎰 Lottery Lab")
st.caption("An honest statistical sandbox for lottery numbers.")

st.markdown(
    "> **The premise:** nobody can predict a fair lottery draw — every combination "
    "is equally likely, and past draws carry *zero* information about the next one. "
    "So this app does the two things that are actually real: **proves** on real "
    "history that no strategy beats chance, and **engineers** the only genuine "
    "levers — covering designs and jackpot-share expected value."
)

# --- Headline odds ---------------------------------------------------------------

rows = odds_table()

cols = st.columns(len(rows))
for col, r in zip(cols, rows):
    with col:
        st.metric(
            r["game"],
            f"1 in {r['match3_one_in']:,}",
            help=(
                "Exact odds that one ticket matches 3 main numbers. "
                f"Jackpot: 1 in {r['jackpot_one_in']:,}."
            ),
            border=True,
        )
st.caption(
    "Cards show **match-3 odds** — the realistic small win, exact to the draw matrix."
)

df = pd.DataFrame(
    {
        "Game": [r["game"] for r in rows],
        "Matrix": [r["matrix"] for r in rows],
        "Match 3 main": [f"1 in {r['match3_one_in']:,}" for r in rows],
        "Jackpot": [f"1 in {r['jackpot_one_in']:,}" for r in rows],
    }
)
st.dataframe(df, hide_index=True, width="stretch")

st.info(
    "**EuroDreams is the realistic target.** Its match-3 is an order of magnitude "
    "friendlier than Powerball's — a single ticket is expected to match 3 about "
    "every 32 draws (median wait ~3 months at two draws a week).",
    icon="🎯",
)

# --- The three real ideas --------------------------------------------------------

st.subheader("The three real ideas")
c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("##### ⚖️ You can't predict the draw")
        st.markdown(
            "Run `random`, `hot`, `cold`, `last_echo` and the ghost of the old "
            "LSTM through a walk-forward backtest: they **all** land within noise "
            "of the exact hypergeometric baseline. *(→ Strategies vs Chance)*"
        )
with c2:
    with st.container(border=True):
        st.markdown("##### 🛞 Wheeling guarantees coverage, not profit")
        st.markdown(
            "A covering design guarantees a 3-match *if* ≥3 of your chosen numbers "
            "are drawn. It never changes the odds or per-ticket EV — it buys "
            "**determinism** at a fixed, honest cost. *(→ Wheeling)*"
        )
with c3:
    with st.container(border=True):
        st.markdown("##### 💰 Unpopular numbers raise *payout*, not probability")
        st.markdown(
            "Jackpots are split among matching tickets; avoiding birthdays (≤31) "
            "means sharing with fewer people **if** you win. Real, but only for "
            "pari-mutuel tiers. *(→ Expected Value)*"
        )

# --- Data on disk ----------------------------------------------------------------

st.subheader("Data on disk")
status_cols = st.columns(len(shared.GAME_KEYS))
for col, key in zip(status_cols, shared.GAME_KEYS):
    spec = games.get(key)
    with col, st.container(border=True):
        status = shared.history_status(key)
        if status:
            st.metric(spec.name, f"{status['draws']:,}")
            st.caption(f"draws · {status['first']} → {status['last']}")
        else:
            st.metric(spec.name, "no data")
            st.caption("Fetch a snapshot on the **Data** page.")

shared.footer()
