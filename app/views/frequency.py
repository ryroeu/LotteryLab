"""Frequency dashboard — fun to look at, zero predictive power (and it says so)."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app import shared
from lotterylab import analytics, games

st.title("📊 Frequency")
st.caption(
    "How often each main number has been drawn — plus the test that shows it means nothing."
)

st.info(analytics.DISCLAIMER, icon="🧊")

left, right = st.columns([2, 1])
with left:
    game = shared.game_selector()
with right:
    synth, synth_n = shared.synth_controls()

spec = games.get(game)

hist = shared.require_history(game, synth, synth_n)

counts = analytics.frequency(hist, spec)[1:]  # index 0 unused
expected = counts.sum() / spec.main_max

freq_df = pd.DataFrame({"number": range(1, spec.main_max + 1), "count": counts})
axis_values = [1] + list(range(5, spec.main_max + 1, 5))

bars = (
    alt.Chart(freq_df)
    .mark_bar(opacity=0.9)
    .encode(
        x=alt.X(
            "number:O",
            title="Main number",
            axis=alt.Axis(labelAngle=0, values=axis_values),
        ),
        y=alt.Y("count:Q", title="Times drawn"),
        tooltip=[
            alt.Tooltip("number:O", title="Number"),
            alt.Tooltip("count:Q", title="Times drawn"),
        ],
        color=alt.value(shared.ACCENT),
    )
)
rule = (
    alt.Chart(pd.DataFrame({"expected": [expected]}))
    .mark_rule(color="#FF6B6B", strokeDash=[6, 4], size=2)
    .encode(y="expected:Q")
)
st.altair_chart((bars + rule).properties(height=320), width="stretch")
st.caption(
    f"Dashed line = expected count per number if perfectly uniform "
    f"({expected:.1f}). The wobble around it is exactly what fair sampling looks like."
)

# --- Chi-square uniformity test ----------------------------------------------------

st.subheader("Is it uniform? (chi-square goodness of fit)")
test = analytics.uniformity_test(hist, spec)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Balls observed", f"{test['n_balls_observed']:,}", border=True)
m2.metric("Chi-square", f"{test['chi2']:.1f}", border=True)
m3.metric("Degrees of freedom", test["dof"], border=True)
m4.metric("p-value", f"{test['p_value']:.3f}", border=True)

if test["p_value"] > 0.05:
    st.success(
        f"Verdict: **{test['verdict']}** — 'hot' and 'cold' numbers are sampling noise.",
        icon="✅",
    )
else:
    st.error(f"Verdict: **{test['verdict']}**", icon="⚠️")

shared.footer()
