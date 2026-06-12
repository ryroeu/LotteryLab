"""Wheeling — covering designs, the only mechanism that *guarantees* a 3-match."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import shared
from lotterylab import games
from lotterylab.wheeling import cycled_specials, spread_numbers, wheel_report

MAX_WHEEL = 14  # covering designs grow fast beyond this

st.title("🛞 Wheeling")
st.caption(
    "Pick K numbers, and a covering design builds the smallest greedy block of tickets "
    "such that **any** 3 of your K appearing in the draw guarantees a 3-match ticket."
)

left, right = st.columns([1, 2])
with left:
    game = shared.game_selector()
spec = games.get(game)

with right:
    mode = st.segmented_control(
        "Numbers to wheel",
        ["Even spread", "Pick my own"],
        default="Even spread",
    ) or "Even spread"

if mode == "Even spread":
    n = st.slider("How many numbers (K)", spec.main_count, MAX_WHEEL, 9)
    chosen = spread_numbers(n, 1, spec.main_max)
    st.caption(f"An even spread across 1–{spec.main_max}: **{', '.join(map(str, chosen))}**")
else:
    chosen = sorted(
        st.multiselect(
            (
                f"Your numbers (1–{spec.main_max}, at least {spec.main_count}, "
                f"at most {MAX_WHEEL})"
            ),
            options=list(range(1, spec.main_max + 1)),
            default=spread_numbers(9, 1, spec.main_max),
            max_selections=MAX_WHEEL,
        )
    )
    if len(chosen) < spec.main_count:
        st.info(f"Pick at least {spec.main_count} numbers to build a wheel for {spec.name}.")
        st.stop()


@st.cache_data(show_spinner="Building the covering design…")
def build_wheel(game_key: str, numbers: tuple[int, ...]):
    """Build and cache a wheel report for a selected number pool."""
    return wheel_report(games.get(game_key), list(numbers))


report = build_wheel(game, tuple(chosen))

three_match_pay = spec.prize_table.get((3, 0), 0.0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tickets", report.n_tickets, help=f"k = {report.k} numbers per ticket", border=True)
m2.metric(
    "Cost per draw",
    shared.fmt_money(report.cost, report.currency),
    delta=f"-{shared.fmt_money(report.cost - three_match_pay, report.currency)} vs a 3-match prize",
    delta_color="inverse",
    help=f"A 3-main match pays ≈ {shared.fmt_money(three_match_pay, report.currency)} here.",
    border=True,
)
m3.metric(
    "P(guarantee fires)",
    f"{report.p_condition:.1%}",
    help=(
        f"Probability that ≥{report.t} of your {len(report.chosen)} numbers "
        "are among the drawn balls."
    ),
    border=True,
)
m4.metric("≈ draws between hits", f"{report.expected_draws_between_hits:.1f}", border=True)

st.success(f"**Guarantee:** {report.guarantee}.", icon="🔒")
st.warning(
    f"**Honest note:** the guarantee covers the {spec.main_count} main numbers only"
    + (
        f" — the {spec.special_name} (1–{spec.special_max}) is an independent pick "
        "the wheel can't help with."
        if spec.special_count
        else "."
    )
    + " It holds only when the condition is met, and the block's cost exceeds a "
    "3-match prize: wheeling buys **determinism, not profit**.",
    icon="📏",
)

# --- The tickets -------------------------------------------------------------------

st.subheader(f"The {report.n_tickets} tickets")
rows = []
for i, tk in enumerate(report.tickets):
    row = {
        "Ticket": i + 1,
        "Main numbers": " · ".join(f"{x}" for x in tk),
    }
    if spec.special_count:
        row[f"{spec.special_name} (cycled)"] = " · ".join(
            str(x) for x in cycled_specials(i, spec)
        )
    rows.append(row)
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
if spec.special_count:
    st.caption(
        f"Each ticket also needs {spec.special_count} {spec.special_name}(s) from "
        f"1–{spec.special_max}; they're cycled so the block spreads that pick too."
    )

with st.expander("What exactly is a covering design?"):
    st.markdown(
        f"""
A lottery wheel is a **covering design** `C(K, k, t)`: from your K chosen numbers,
generate k-number tickets so that *every* t-subset of your K appears on at least one
ticket (here t = {report.t}, k = {report.k}). The construction is greedy — not always
minimal (minimal covers are NP-hard; see the La Jolla Covering Repository) — but the
guarantee is brute-force **verified** before it's reported.

What it does *not* do: change the draw, the per-ticket win probability, or the
per-ticket expected value. Guaranteeing a 3-match on *every* draw would mean covering
the whole pool, which costs far more in tickets than a 3-match pays.
"""
    )

shared.footer()
