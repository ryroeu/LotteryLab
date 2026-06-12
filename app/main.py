"""Lottery Lab — the web UI. Run with:  streamlit run app/main.py

A clean front-end over the lotterylab library, organised the way the project
thinks: understand the odds, *prove* no strategy beats them, then engineer the
two levers that are real (covering designs and jackpot-share EV).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts only this script's directory (app/) on sys.path. Add the repo
# root so `from app import shared` and `import lotterylab` resolve in every
# view this entry point runs — without requiring a pip install.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st  # noqa: E402

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
