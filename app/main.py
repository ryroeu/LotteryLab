"""Lottery Lab — the web UI. Run with:  streamlit run app/main.py

A clean front-end over the lotterylab library, organised the way the project
thinks: understand the odds, *prove* no strategy beats them, then engineer the
two levers that are real (covering designs and jackpot-share EV).
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Lottery Lab",
    page_icon="🎰",
    layout="wide",
    menu_items={
        "About": (
            "**Lottery Lab** — an honest statistical sandbox for lottery numbers. "
            "Nothing predicts a fair draw; this app proves it, then engineers the "
            "only levers that are real."
        )
    },
)

pages = {
    "Understand": [
        st.Page("views/overview.py", title="The Odds", icon="🎲", default=True),
        st.Page("views/frequency.py", title="Frequency", icon="📊"),
    ],
    "Prove": [
        st.Page("views/prove.py", title="Strategies vs Chance", icon="⚖️"),
    ],
    "Engineer": [
        st.Page("views/wheel.py", title="Wheeling", icon="🛞"),
        st.Page("views/ev.py", title="Expected Value", icon="💰"),
    ],
    "Feel It": [
        st.Page("views/simulator.py", title="Time & Variance", icon="⏳"),
    ],
    "Admin": [
        st.Page("views/data.py", title="Data", icon="🗃️"),
    ],
}

nav = st.navigation(pages)

with st.sidebar:
    st.caption(
        "**The premise:** every combination is equally likely and past draws "
        "carry zero information about the next one. The fun is proving it — "
        "and engineering what's actually real."
    )

nav.run()
