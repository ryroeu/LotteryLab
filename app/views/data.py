"""Data — immutable raw snapshots: what's on disk, and one-click refresh."""

from __future__ import annotations

import os

import requests
import streamlit as st

from app import shared
from lotterylab import games, store

st.title("🗃️ Data")
st.caption(
    "Source CSVs are immutable, timestamped snapshots under `data/raw/<game>/` — "
    "never overwritten. The loader tries snapshots newest-first and keeps draws "
    "since each game's current matrix took effect, so odds and backtests always "
    "run against one consistent set of rules."
)

for key in shared.GAME_KEYS:
    spec = games.get(key)
    snapshots = store.all_snapshots(key)
    status = shared.history_status(key)

    with st.container(border=True):
        info_col, action_col = st.columns([3, 1], vertical_alignment="center")
        with info_col:
            st.markdown(f"##### {spec.name}")
            if status:
                st.markdown(
                    f"**{status['draws']:,} draws** loaded · {status['first']} → "
                    f"{status['last']} · {len(snapshots)} snapshot(s) on disk"
                )
            else:
                st.markdown(
                    f"**No usable data** · {len(snapshots)} snapshot(s) on disk — "
                    "fetch one below."
                )
        with action_col:
            fetch = st.button(
                "Fetch fresh snapshot",
                key=f"fetch_{key}",
                width="stretch",
                help="Downloads from the official source and writes a NEW snapshot "
                "(nothing is written if the data is unchanged, unparsable or "
                "truncated).",
            )

        if fetch:
            with st.spinner(f"Downloading {spec.name} history…"):
                try:
                    path = store.fetch_raw(key)
                except (
                    OSError,
                    ValueError,
                    requests.RequestException,
                ) as error:
                    st.error(f"Fetch failed: {error}")
                else:
                    st.success(f"Newest snapshot: `{os.path.basename(path)}`.")
                    shared.clear_data_caches()
                    st.rerun()

        if snapshots:
            with st.expander(f"Snapshots ({len(snapshots)}, newest first)"):
                for p in snapshots:
                    size_kb = os.path.getsize(p) / 1024
                    st.markdown(f"- `{os.path.basename(p)}` ({size_kb:,.0f} KB)")

st.info(
    "Powerball and Mega Millions come from NY Open Data; EuroMillions, EuroDreams "
    "and Loto from FDJ (the French operator), whose era-split files are merged and "
    "de-duped into one combined snapshot per fetch — including how many tickets won "
    "each prize rank and what they were paid, which the Expected Value page uses.",
    icon="🌐",
)

shared.footer()
