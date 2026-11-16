"""Expected Value — who you share with, and when the jackpot is worth more."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app import shared
from lotterylab import ev, popularity
from lotterylab.sharing import sharing_backtest
from lotterylab.store import FDJ_SOURCES

st.title("💰 Expected Value")
st.caption(
    "Picking unpopular numbers does **not** change your odds of winning — but most "
    "prizes are pools split among that draw's winners, so numbers fewer people play "
    "are worth more *if* they win. (Ziemba; Henze & Riedwyl.)"
)

game = shared.game_selector(default="loto", key="ev_game")
mkt = shared.market(game)
spec = mkt.spec
k = spec.main_count
money = lambda x, d=2: shared.fmt_money(x, spec.currency, d)  # noqa: E731

# A plausible date-heavy ticket (all ≤ 31) as the starting point.
BIRTHDAYISH = [7, 11, 14, 21, 27, 31]

left, right = st.columns([2, 1])
with left:
    user_main = st.multiselect(
        f"Your {k} main numbers (1–{spec.main_max})",
        options=list(range(1, spec.main_max + 1)),
        default=BIRTHDAYISH[:k],
        max_selections=k,
    )
with right:
    jackpot = st.number_input(
        f"Advertised jackpot ({spec.currency})",
        min_value=float(spec.jackpot_estimate),
        value=float(spec.jackpot_estimate),
        step=1_000_000.0,
        disabled=not spec.jackpot_rolls,
        help=(
            "The top prize on offer for the draw you'd play. Ticket sales are "
            "scaled to it — bigger jackpots attract more players to share with."
            if spec.jackpot_rolls
            else "This game's top prize is fixed; it does not roll over."
        ),
    )

if mkt.measured:
    st.caption(
        f"Measured from FDJ's records: prize table from {len(mkt.frame):,} draws, "
        f"≈{mkt.tickets(jackpot) / 1e6:.1f}M tickets in a draw like this, number "
        f"popularity {mkt.model.describe()}."
    )
else:
    st.caption(
        "No operator payout data for this game: prizes are the static table and "
        f"popularity is a {mkt.model.describe()}."
    )

if len(user_main) < k:
    st.info(f"Pick exactly {k} numbers ({k - len(user_main)} to go).")
    st.stop()

least, most = ev.best_and_worst(mkt)
tickets = {
    "Your ticket": tuple(sorted(user_main)),
    "Least-picked numbers": least,
    "Most-picked numbers": most,
}
reports = {
    label: ev.evaluate(main, mkt, jackpot=jackpot) for label, main in tickets.items()
}

cols = st.columns(3)
for col, (label, rep) in zip(cols, reports.items()):
    with col, st.container(border=True):
        st.markdown(f"##### {label}")
        st.code(" · ".join(map(str, rep.main)), language=None)
        st.metric(
            "Popularity vs a random pick",
            f"{rep.popularity_multiplier:.2f}×",
            delta="shared less" if rep.popularity_multiplier < 1 else "shared more",
            delta_color="normal" if rep.popularity_multiplier < 1 else "inverse",
        )
        st.metric(
            "Shared prizes pay",
            f"{rep.shared_multiplier:.2f}×",
            help="This ticket's expected payout on shared-pool ranks, relative to "
            "the average winner of those ranks.",
        )
        st.metric(
            "Top prize kept if you win",
            f"{rep.expected_jackpot_share:.1%}",
            help="Expected fraction after splitting with co-winners on the same "
            "combination.",
        )
        st.metric(
            "EV per ticket",
            money(rep.ev_per_ticket, 3),
            delta=f"{rep.return_rate:.1%} of the {money(spec.price)} price",
            delta_color="off",
        )

parts = {
    "Top prize": "ev_jackpot",
    "Shared ranks": "ev_shared",
    "Fixed prizes": "ev_fixed",
    "Raffle": "ev_raffle",
}
ev_df = pd.DataFrame(
    [
        {"ticket": label, "source": part, "ev": getattr(rep, attr)}
        for label, rep in reports.items()
        for part, attr in parts.items()
    ]
)
bars = (
    alt.Chart(ev_df)
    .mark_bar(opacity=0.9)
    .encode(
        x=alt.X("sum(ev):Q", title=f"Expected value per ticket ({spec.currency})"),
        y=alt.Y("ticket:N", title=None, sort=list(reports)),
        color=alt.Color("source:N", title=None, sort=list(parts)),
        order=alt.Order("source:N", sort="descending"),
        tooltip=[
            alt.Tooltip("ticket:N"),
            alt.Tooltip("source:N"),
            alt.Tooltip("ev:Q", format=",.3f"),
        ],
    )
)
price_rule = (
    alt.Chart(pd.DataFrame({"price": [spec.price]}))
    .mark_rule(color="#FF6B6B", strokeDash=[6, 4], size=2)
    .encode(x="price:Q")
)
st.altair_chart((bars + price_rule).properties(height=180), width="stretch")
st.caption(f"Dashed line = the ticket price ({money(spec.price)}).")

if spec.jackpot_rolls:
    notes = []
    for label, main in tickets.items():
        even = ev.breakeven_jackpot(main, mkt)
        if even is None:
            notes.append(f"**{label}**: never reaches break-even")
        elif spec.jackpot_cap and even > spec.jackpot_cap:
            notes.append(
                f"**{label}**: would need {money(even, 0)} — above the "
                f"{money(spec.jackpot_cap, 0)} cap"
            )
        else:
            notes.append(f"**{label}**: breaks even near {money(even, 0)}")
    st.markdown("**Break-even jackpot** — " + " · ".join(notes))

st.info(
    "**Honest note:** every ticket above has identical odds; only who you'd share "
    "with differs. 'Break-even' is an average over millions of draws — almost all "
    "of that value sits in a jackpot you will almost certainly never hit, so the "
    "typical outcome of playing is still a loss.",
    icon="📏",
)

# --- Which numbers are popular -----------------------------------------------------

st.subheader("Which numbers do other players pick?")
pop_df = pd.DataFrame(
    {
        "number": range(1, spec.main_max + 1),
        "popularity": list(mkt.model.weights),
    }
)
pop_bars = (
    alt.Chart(pop_df)
    .mark_bar(color=shared.ACCENT, opacity=0.9)
    .encode(
        x=alt.X(
            "number:O",
            title="Main number",
            axis=alt.Axis(labelAngle=0, values=[1, *range(5, spec.main_max + 1, 5)]),
        ),
        y=alt.Y(
            "popularity:Q",
            title="Relative popularity",
            scale=alt.Scale(zero=False),
        ),
        tooltip=[
            alt.Tooltip("number:O"),
            alt.Tooltip("popularity:Q", format=".2f"),
        ],
    )
)
average = (
    alt.Chart(pd.DataFrame({"y": [1.0]}))
    .mark_rule(color="#FF6B6B", strokeDash=[6, 4], size=2)
    .encode(y="y:Q")
)
st.altair_chart((pop_bars + average).properties(height=260), width="stretch")
st.caption(
    f"1.00 = picked as often as average. Source: {mkt.model.describe()}. "
    "This describes the crowd, not the draw — every number is equally likely to "
    "come up."
)

if game in FDJ_SOURCES and mkt.model.source == "fitted":

    @st.cache_data(show_spinner="Checking the model against held-out draws…")
    def run_sharing(game_key: str, _snapshot_count: int):
        """Walk-forward sharing test, cached per game and data state."""
        return sharing_backtest(game_key)

    with st.expander("Does it hold up? A walk-forward test on real payouts"):
        try:
            test = run_sharing(game, len(mkt.frame))
        except ValueError as error:
            st.caption(f"Not enough data to test yet: {error}")
        else:
            st.markdown(
                f"Popularity was fitted on the **{test.n_fit_draws:,} draws before "
                f"{test.cutoff}**, then asked what hypothetical tickets would have "
                f"collected in the **{test.n_test_draws:,} draws after it**, using "
                "the payouts FDJ actually published. 1.00 = paid like the average "
                "winner; all groups win equally often."
            )
            st.dataframe(
                pd.DataFrame(
                    {
                        "Tickets drawn from": [g.label for g in test.groups],
                        "Realised": [g.realised for g in test.groups],
                        "Model predicted": [g.predicted for g in test.groups],
                    }
                ),
                hide_index=True,
                width="stretch",
                column_config={
                    "Realised": st.column_config.NumberColumn(format="%.3f"),
                    "Model predicted": st.column_config.NumberColumn(format="%.3f"),
                },
            )
            st.caption(
                f"`{popularity.__name__}` fits the weights; only the high-volume "
                "shared ranks are scored, so a few lucky top-rank payouts can't "
                "drown the result."
            )

shared.footer()
