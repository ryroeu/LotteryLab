"""Data — immutable raw snapshots: what's on disk, and one-click refresh."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_APP = str(Path(__file__).resolve().parents[1])
if _APP not in sys.path:
    sys.path.insert(0, _APP)

import streamlit as st  # noqa: E402

import shared  # noqa: E402
from lotterylab import games, store  # noqa: E402

st.title("🗃️ Data")
st.caption(
    "Source CSVs are immutable, timestamped snapshots under `data/raw/<game>/` — "
    "never overwritten. The loader tries snapshots newest-first and applies the "
    "2018 floor + each game's matrix-change date, so odds and backtests always "
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
                "(rejects a download that doesn't parse to any draws).",
            )

        if fetch:
            with st.spinner(f"Downloading {spec.name} history…"):
                try:
                    path = store.fetch_raw(key)
                except Exception as e:
                    st.error(f"Fetch failed: {e}")
                else:
                    st.success(f"Wrote `{os.path.basename(path)}`.")
                    shared.load_history.clear()
                    shared.history_status.clear()
                    st.rerun()

        if snapshots:
            with st.expander(f"Snapshots ({len(snapshots)}, newest first)"):
                for p in snapshots:
                    size_kb = os.path.getsize(p) / 1024
                    st.markdown(f"- `{os.path.basename(p)}` ({size_kb:,.0f} KB)")

st.info(
    "Powerball and Mega Millions come from NY Open Data; EuroMillions and EuroDreams "
    "from FDJ (the French operator), whose era-split files are merged and de-duped "
    "into one combined snapshot per fetch.",
    icon="🌐",
)

shared.footer()
