# 🎰 Lottery Lab

An honest statistical sandbox for lottery numbers, pointed at the French games.

> **The premise:** nobody can predict a fair lottery draw — every combination is
> equally likely, and past draws carry *zero* information about the next one. So
> this project does the things that are actually real instead:
>
> 1. **Proves** — empirically, on real draw history — that no "strategy" beats
>    chance. (Watching clever ideas faceplant onto the chance line is the fun part.)
> 2. **Measures** the two levers that change what a ticket is *worth*: **who you
>    share with** (most French prizes are pools split among that draw's winners)
>    and **when you play** (a rollover jackpot adds value, and attracts company).
>    Both come from the operator's own records, and both are tested out of sample.
> 3. **Engineers** **covering designs** (wheeling) that *guarantee* a small-prize
>    match at a known cost.

## The odds (exact)

| Game | Matrix | Match 3 main | Jackpot |
|---|---|---|---|
| **Loto (FR)** | 5/49 + 1/10 | 1 in 202 | 1 in 19,068,840 |
| EuroMillions (EU) | 5/50 + 2/12 | 1 in 214 | 1 in 139,838,160 |
| **EuroDreams (EU)** | 6/40 + 1/5 | **1 in 32** | 1 in 19,191,900 |
| Powerball (US) | 5/69 + 1/26 | 1 in 557 | 1 in 292,201,338 |
| Mega Millions (US) | 5/70 + 1/24 | 1 in 582 | 1 in 290,472,336 |

**EuroDreams is the friendliest for small wins.** A single ticket is expected to
match 3 about every 32 draws (median wait ~3 months at two draws a week), and two
numbers refund the ticket.

## Quick start

```bash
pip install -e ".[ui,dev]"             # or: pip install -r requirements.txt
streamlit run app/main.py              # ← the web UI (everything below, pointable & clickable)
python -m lotterylab odds              # the table above
python -m lotterylab prove eurodreams  # run every strategy vs the chance line
python -m lotterylab payouts loto      # what each prize rank really pays
python -m lotterylab popular loto      # which numbers other players over-pick
python -m lotterylab ev loto --jackpot 13000000   # price a ticket at tonight's jackpot
python -m lotterylab sharing loto      # walk-forward proof of the sharing effect
python -m lotterylab wheel eurodreams -n 8        # a covering design that guarantees 3-matches
python -m lotterylab fetch             # download fresh official history
pytest                                 # the test suite
```

## What the French games really pay

FDJ publishes, for every draw, how many tickets won each prize rank and what each
winner received. `fetch` keeps those columns, and three things that used to be
guesses become measurements:

- **The prize table.** `payouts <game>` shows the mean, lowest and highest payout
  per rank and whether the rank is fixed or a shared pool. (The hand-typed
  EuroDreams table this replaced paid €2.5M for six numbers without the Dream
  Number; the real prize is €2,000 a month for five years, and the app overstated
  the game's return by about a third.)
- **How many tickets are in a draw** — winners of the most-won rank ÷ its exact
  odds — which sets how likely a jackpot is to be split, and how sales swell as
  the jackpot grows.
- **Which numbers the crowd picks.** Draws full of popular numbers produce more
  winners. `popularity.py` fits a per-number weight to those swings (ridge
  regression on winner-count ratios, so unknown sales cancel out), then measures
  how strongly each shared rank's payout reacts.

### The sharing effect, tested on real money

`sharing <game>` fits popularity on draws before a cut-off, then looks up what
hypothetical tickets would *actually have been paid* in the draws after it:

```
Sharing test for Loto (FR): popularity fitted on 1077 draws before 2024-01-24, scored on the 440 after it.
  Shared-rank money actually paid, relative to the average winner (1.00):
    tickets drawn from              realised  predicted
    least-picked third of numbers      1.203      1.168
    all numbers (random)               1.000      1.000
    most-picked third of numbers       0.843      0.867
```

All three groups win equally often — that is the premise. What differs is what a
win is worth: in Loto, tickets built from neglected numbers collected about 20%
more per win than average and popular ones about 16% less, on draws the model
never saw. EuroMillions shows the same pattern at about ±8%. (EuroDreams has only
~120 draws under its current prize rules, and there the model currently
over-predicts: 1.13 realised against 1.27 forecast.)

### What that is and isn't

- It is **not** a way to win more often. Nothing is.
- It **is** a measurable difference in expected value between tickets with
  identical odds. On current data a Loto ticket of neglected numbers returns about
  57% of its price at the minimum jackpot, against about 34% for the most popular
  combination and about 44% for a random one.
- `ev <game> --jackpot N` adds the rollover: the top prize's value grows with the
  jackpot while extra players dilute it. For Loto the model puts break-even for the
  least-picked numbers near a €21M jackpot (29 of the 1,100 draws since
  November 2019 offered that much); for EuroMillions it sits above the €250M cap,
  so never.
- Read "break-even" as an average over millions of draws. Almost all of that value
  sits in a jackpot you will almost certainly never hit, so the typical outcome of
  playing is still a loss. French lottery winnings are untaxed, so the figures are
  what a winner receives; annuities are counted at nominal value.

## The web UI

`streamlit run app/main.py` serves a dark dashboard over the same library the CLI
uses — same numbers, same honest framing, organised the way the project thinks:

| Section | Page | What you can do |
|---|---|---|
| Understand | **The Odds** | the exact odds per game, the premise, data status |
| Understand | **Frequency** | draw-frequency chart + live chi-square uniformity test |
| Prove | **Strategies vs Chance** | run every strategy walk-forward; see them hug the baseline (±2 SE band), per-strategy tier breakdown; synthetic-data toggle |
| Engineer | **Wheeling** | build a covering design from a spread or your own numbers; guarantee, cost, and the full ticket block |
| Engineer | **Expected Value** | price your numbers against the least- and most-picked combinations at any jackpot; number-popularity chart; the walk-forward sharing test |
| Feel It | **Time & Variance** | per-game wait until a 3-match; Monte-Carlo season simulator with net-result histogram |
| Admin | **Data** | snapshots on disk + one-click fetch of fresh official history |

Everything is cached, so the heavy bits (backtests, covering designs, simulations)
run once and then respond instantly.

## What's in the box

| Command | What it does |
|---|---|
| `odds` | Exact match-3 and jackpot odds for every game. |
| `prove [game]` | Runs all strategies through the backtest; shows them all hugging chance. Add `--synth --synth-n 50000` for a high-power run on provably-fair draws (seconds). |
| `backtest <game> <strategy>` | One strategy, walked forward over real history, scored vs the exact baseline (with a z-score). |
| `payouts <game>` | Per-rank odds and observed payouts, tickets per draw, raffle value. |
| `popular <game>` | Least- and most-picked numbers, and each shared rank's payout elasticity. |
| `ev <game>` | Prices tickets (yours with `--numbers`) by expected payout at any `--jackpot`, with break-even. |
| `sharing <game>` | Walk-forward test of the sharing effect against actual payouts. |
| `wheel <game> -n N` | Builds a covering design over N numbers and reports the **guarantee**, ticket count, and honest cost. |
| `freq <game>` | Frequency dashboard + a chi-square uniformity test — labelled *zero predictive power*. |
| `wait` | Expected / median time until a single ticket matches 3, per game. |
| `variance <game>` | Monte-Carlo the net-result distribution of a one-ticket-per-draw player. |
| `fetch [game ...]` | Downloads fresh official history as new snapshots (all games by default). |

### The three real ideas

- **You can't predict the draw.** The `prove` command demonstrates it: `random`,
  `hot`, `cold`, `last_echo`, and `order_stat_mean` (the constant ticket a
  mean-squared-error regressor settles on) all land within noise of the
  hypergeometric baseline. The uniformity test is corrected for drawing without
  replacement — the textbook chi-square is too lenient on lottery data (about 1%
  false alarms on fair 6/40 draws instead of 5%).
- **Unpopular numbers raise expected payout, not win probability.** Prizes paid
  from a shared pool are split among that draw's winners, so avoiding the numbers
  everyone plays means a bigger slice *if* you win — at every shared rank, not just
  the jackpot. Measured from winner counts and verified on held-out payouts (see
  above). (Ziemba; Henze & Riedwyl.)
- **Wheeling guarantees coverage, not profit.** A covering design `C(K, k, 3)`
  guarantees a 3-match *if* ≥3 of your K chosen numbers are drawn. It never changes
  the draw odds or per-ticket EV — it buys determinism at a fixed, honest cost
  (which exceeds what a 3-match pays).

## Layout

```
lotterylab/
  games.py          GameSpec registry — pools, prices, prize ranks and tables (one source of truth)
  combinatorics.py  exact hypergeometric odds + order-statistic means
  validate.py       ticket validity (range / uniqueness / count) — enforced everywhere
  schema.py         canonical Draw + tidy DataFrame
  adapters.py       per-game CSV -> Draw
  store.py          immutable raw snapshots; fetching; current-matrix filtering; payout frames
  payouts.py        observed prize tables, tickets per draw, raffle value, sales vs jackpot
  popularity.py     number popularity fitted to winner counts; payout elasticities
  ev.py             expected value: shared pools, jackpot splitting, rollovers, break-even
  sharing.py        walk-forward test of the sharing effect on actual payouts
  strategy.py       random / hot / cold / last_echo / order_stat_mean / biased_high
  baseline.py       the exact chance line
  backtest.py       walk-forward harness with z-vs-baseline  ← the heart
  wheeling.py       covering designs + brute-forced guarantee
  analytics.py      frequency dashboard + corrected chi-square uniformity
  simulate.py       variance simulator + time-to-3-match
  synth.py          provably-fair synthetic draws (offline tests / demos)
  cli.py            python -m lotterylab <command>
app/
  main.py           streamlit run app/main.py — the web UI entry point
  shared.py         cached data access + formatting shared by every page
  views/            one file per page (overview, prove, wheel, ev, …)
tests/              pytest suite: exact odds, calibration on fair data, no-lookahead,
                    snapshot safety, model recovery on an invented crowd, every CLI
                    command and app page
data/raw/<game>/    immutable source snapshots (never overwritten)
```

## Data

Source CSVs live under `data/raw/<game>/` as immutable, timestamped snapshots and
are never overwritten. The loader keeps
draws since each game's current matrix took effect (`GameSpec.matrix_since`), so
odds and backtests are always computed against one consistent set of rules (e.g.
old-matrix Mega Millions rows are dropped automatically). It tries snapshots
newest-first and skips any that fail to parse, so one bad file never breaks the
pipeline.

```bash
python -m lotterylab fetch            # every game
python -m lotterylab fetch loto eurodreams
```

`fetch` writes a NEW snapshot only when the data changed. It writes nothing if the
download doesn't parse to draws, or has fewer draws than the newest snapshot
already on disk (a truncated or retired source can't displace good data). A weekly
GitHub Actions job (`.github/workflows/refresh-data.yml`) runs it and commits new
snapshots to `main`.

| Game | Source | Per-rank winners & payouts |
|---|---|---|
| Loto | FDJ official history (draw-info API), 2008-10 → | ✅ |
| EuroMillions | FDJ official history (draw-info API), 2016-09 → | ✅ |
| EuroDreams | FDJ official history (draw-info API), 2023-11 → | ✅ |
| Powerball | NY Open Data (`data.ny.gov`) | — |
| Mega Millions | NY Open Data (`data.ny.gov`) | — |

The three European games come from **FDJ** (the French operator). FDJ publishes
history in semicolon-delimited, era-split files; `fetch` downloads the relevant
files, parses the French format, de-dupes by draw date, and writes one combined
snapshot in the adapter's layout with `r<rank>_winners`, `r<rank>_winners_fr` and
`r<rank>_payout` columns appended. The US games have no such data, so their prize
tables stay static and their popularity model is a labelled heuristic.

Prize structures change. Winner counts stay comparable as long as the ranks mean
the same thing (`GameSpec.ranks_since`), but payout statistics only use draws under
the present prize rules (`GameSpec.payout_since`) — EuroDreams, for instance,
re-weighted its shared ranks on 2025-10-02 (five numbers went from about €100 to
about €480).

Mega Millions changed its complete matrix on 2025-04-08: the Mega Ball pool fell
from 25 to 24 and the $5 ticket gained an embedded random multiplier. Its canonical
history therefore starts on that date; mixing older 1/25 special-ball draws into a
1/24 probability baseline would bias the analysis. Approximate non-jackpot payouts
use the multiplier's 3× expected value.

By default the data directory is `data/` in a source checkout. Set
`LOTTERYLAB_DATA` to use another location; an installed copy with neither falls
back to `~/.lotterylab/data`.

---

*Lottery Lab is for fun and learning. It does not improve anyone's odds of winning,
because nothing can. Please gamble responsibly — the expected value of every ticket
here is below what you pay for it in all but the rarest draws.*
