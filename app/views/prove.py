"""Strategies vs Chance — every built-in strategy, walked forward, hugging the line."""

from __future__ import annotations

import sys
from pathlib import Path

_APP = str(Path(__file__).resolve().parents[1])
if _APP not in sys.path:
    sys.path.insert(0, _APP)

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import shared  # noqa: E402
from lotterylab import games  # noqa: E402
from lotterylab.strategy import BUILTIN_STRATEGIES  # noqa: E402

st.title("⚖️ Strategies vs Chance")
st.caption(
    "Walk every strategy forward over draws it could not have seen — "
    "if the game is fair, they all sit within ~2 standard errors of the exact chance line."
)

left, mid, right = st.columns([2, 1, 1])
with left:
    game = shared.game_selector()
with mid:
    n_tickets = st.slider("Tickets per draw", 1, 10, 1)
with right:
    synth, synth_n = shared.synth_controls()

spec = games.get(game)

try:
    hist = shared.load_history(game, synth, synth_n)
except Exception as e:
    st.error(f"Could not load history for {spec.name}: {e}")
    st.stop()

shared.data_source_caption(game, synth, synth_n)


def verdict_of(z: float) -> str:
    # An "edge" would be ABOVE baseline. With this many strategies, |z|~2 is
    # expected by chance, so only a large positive surprise would matter.
    if z > 3:
        return "🔍 surprise (check)"
    if z < -2:
        return "🥶 below chance"
    return "✅ no edge"


results = {
    name: shared.run_backtest(game, name, n_tickets, synth, synth_n)
    for name in BUILTIN_STRATEGIES
}

any_res = next(iter(results.values()))
p3 = any_res.baseline_three_plus_rate
opportunities = any_res.n_draws * n_tickets
se_rate = (p3 * (1 - p3) / opportunities) ** 0.5 if opportunities else 0.0

b1, b2, b3 = st.columns(3)
b1.metric("Chance line (≥3 main)", f"{p3:.5f}", help=shared.fmt_one_in(p3), border=True)
b2.metric("Draws evaluated", f"{any_res.n_draws:,}", border=True)
b3.metric(
    "Expected ≥3 hits per strategy",
    f"{opportunities * p3:.1f}",
    help="opportunities × baseline probability",
    border=True,
)

table = pd.DataFrame(
    {
        "strategy": list(results),
        "hits": [r.three_plus_hits for r in results.values()],
        "rate": [r.three_plus_rate for r in results.values()],
        "z": [r.z_vs_baseline for r in results.values()],
        "verdict": [verdict_of(r.z_vs_baseline) for r in results.values()],
    }
)

band = (
    alt.Chart(pd.DataFrame({"lo": [max(0.0, p3 - 2 * se_rate)], "hi": [p3 + 2 * se_rate]}))
    .mark_rect(opacity=0.15, color="#9AA4B2")
    .encode(x="lo:Q", x2="hi:Q")
)
rule = (
    alt.Chart(pd.DataFrame({"baseline": [p3]}))
    .mark_rule(color="#FF6B6B", size=2)
    .encode(x="baseline:Q")
)
points = (
    alt.Chart(table)
    .mark_circle(size=180, color="#F2B636")
    .encode(
        x=alt.X("rate:Q", title="≥3-main hit rate", scale=alt.Scale(zero=False),
                axis=alt.Axis(format=".4f")),
        y=alt.Y("strategy:N", title=None, sort=table["strategy"].tolist()),
        tooltip=[
            alt.Tooltip("strategy:N"),
            alt.Tooltip("hits:Q", title="≥3 hits"),
            alt.Tooltip("rate:Q", format=".5f"),
            alt.Tooltip("z:Q", format="+.2f"),
        ],
    )
)
st.altair_chart((band + rule + points).properties(height=280), width="stretch")
st.caption(
    "Red line = the exact hypergeometric baseline; grey band = ±2 standard errors. "
    "`order_stat_mean` **is** the old LSTM (`lstm_ghost` is its alias) — no better than `random`."
)

st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={
        "strategy": st.column_config.TextColumn("Strategy"),
        "hits": st.column_config.NumberColumn("≥3 hits"),
        "rate": st.column_config.NumberColumn("Hit rate", format="%.5f"),
        "z": st.column_config.NumberColumn("z vs baseline", format="%+.2f"),
        "verdict": st.column_config.TextColumn("Verdict"),
    },
)

expected_hits = opportunities * p3
if expected_hits < 10:
    st.info(
        f"**Low-power note:** only ~{any_res.n_draws} draws evaluated and match-3 is rare "
        f"here (~{expected_hits:.1f} expected hits per strategy), so individual z-scores "
        "are noisy by nature. For a high-power demo, flip on synthetic draws and raise the count.",
        icon="🔬",
    )

# --- Single-strategy detail --------------------------------------------------------

st.subheader("Strategy detail")
chosen = st.selectbox("Strategy", list(BUILTIN_STRATEGIES), format_func=lambda s: s)
res = results[chosen]

d1, d2, d3, d4 = st.columns(4)
d1.metric("Spent", shared.fmt_money(res.spent, res.currency, 0), border=True)
d2.metric("Won", shared.fmt_money(res.won, res.currency, 0), border=True)
d3.metric("ROI", f"{res.roi:.3f}", help="winnings ÷ spend — fair games sit well below 1", border=True)
d4.metric("z vs baseline", f"{res.z_vs_baseline:+.2f}", border=True)

tiers = [
    {"Tier": spec.matches_label(m, s), "Hits": c}
    for (m, s), c in sorted(res.tier_hits.items(), reverse=True)
    if m + s > 0
]
if tiers:
    st.dataframe(pd.DataFrame(tiers), hide_index=True, width="stretch")
else:
    st.caption("No winning tiers hit in this run.")

st.markdown(
    "> With several strategies, a |z| of ~2 somewhere is *expected* by chance. "
    "What would matter is a large **positive** z that persists across seeds and games. "
    "It never does — that's the whole point."
)

shared.footer()
