"""Expected Value — the one legitimate selection edge: jackpot-share avoidance."""

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
from lotterylab.ev import BIRTHDAY_WEIGHT, ticket_ev  # noqa: E402

st.title("💰 Expected Value")
st.caption(
    "Picking unpopular numbers does **not** change your odds of winning — but jackpots "
    "are split, so numbers fewer people play raise your expected payout *if* you win. "
    "(Ziemba; Henze & Riedwyl.)"
)

game = shared.game_selector()
spec = games.get(game)
k = spec.main_count

# A plausible date-heavy ticket (all ≤ 31) as the starting point.
BIRTHDAYISH = [7, 11, 14, 21, 27, 31]

user_main = st.multiselect(
    f"Your {k} main numbers (1–{spec.main_max})",
    options=list(range(1, spec.main_max + 1)),
    default=BIRTHDAYISH[:k],
    max_selections=k,
)

with st.expander("Model assumptions"):
    jackpot = st.number_input(
        f"Jackpot estimate ({spec.currency})",
        min_value=1_000_000,
        value=int(spec.jackpot_estimate),
        step=1_000_000,
    )
    n_players = st.number_input(
        "Other tickets in the draw",
        min_value=100_000,
        value=10_000_000,
        step=1_000_000,
    )
    st.caption(
        f"Popularity is a deliberately simple, clearly-labelled heuristic: 'birthday' "
        f"numbers (≤31) are over-picked {BIRTHDAY_WEIGHT}× relative to the rest. It is "
        "not a claim to know real ticket-sales distributions — it's a knob to compare "
        "two equally-likely tickets by how much they'd have to share."
    )

if len(user_main) < k:
    st.info(f"Pick exactly {k} numbers ({k - len(user_main)} to go).")
    st.stop()

# The two canonical reference tickets from the CLI.
birthday_ref = tuple(range(1, k + 1))
high_ref = tuple(range(spec.main_max - k + 1, spec.main_max + 1))

tickets = {
    "Your ticket": tuple(sorted(user_main)),
    "All-birthday (1…)": birthday_ref,
    "All-high (…max)": high_ref,
}
reports = {
    label: ticket_ev(main, spec, jackpot=float(jackpot), n_players=int(n_players))
    for label, main in tickets.items()
}

cols = st.columns(3)
for col, (label, rep) in zip(cols, reports.items()):
    with col, st.container(border=True):
        st.markdown(f"##### {label}")
        st.code(" · ".join(map(str, rep.main)), language=None)
        st.metric(
            "Popularity vs uniform",
            f"{rep.popularity_multiplier:.2f}×",
            delta="shared less" if rep.popularity_multiplier < 1 else "shared more",
            delta_color="normal" if rep.popularity_multiplier < 1 else "inverse",
        )
        st.metric(
            "Jackpot kept if you win",
            f"{rep.expected_jackpot_share:.1%}",
            help="Expected fraction after splitting with co-winners on the same combination.",
        )
        st.metric(
            "EV per ticket",
            shared.fmt_money(rep.ev_per_ticket, rep.currency, 4),
            delta=f"{rep.ev_per_ticket - spec.price:+,.2f} vs the {shared.fmt_money(spec.price, spec.currency)} price",
            delta_color="normal",
        )

ev_df = pd.DataFrame(
    {
        "ticket": list(reports),
        "ev": [r.ev_per_ticket for r in reports.values()],
    }
)
bars = (
    alt.Chart(ev_df)
    .mark_bar(color="#F2B636", opacity=0.9)
    .encode(
        x=alt.X("ev:Q", title=f"Expected value per ticket ({spec.currency})"),
        y=alt.Y("ticket:N", title=None, sort=list(reports)),
        tooltip=[alt.Tooltip("ticket:N"), alt.Tooltip("ev:Q", format=",.4f")],
    )
)
price_rule = (
    alt.Chart(pd.DataFrame({"price": [spec.price]}))
    .mark_rule(color="#FF6B6B", strokeDash=[6, 4], size=2)
    .encode(x="price:Q")
)
st.altair_chart((bars + price_rule).properties(height=160), width="stretch")
st.caption(
    f"Dashed line = the ticket price ({shared.fmt_money(spec.price, spec.currency)}). "
    "Every bar stays under it: EV per unit spent is always < 1 — you still lose, just less."
)

st.warning(
    "**Honest note:** the probability of winning is identical for every ticket above. "
    "Only the expected *share* of a pari-mutuel jackpot differs — it does nothing for "
    "fixed lower-tier prizes.",
    icon="📏",
)

shared.footer()
