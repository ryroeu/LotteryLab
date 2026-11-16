"""Command-line interface: ``python -m lotterylab <command>``.

Commands
  odds                 exact match-3 and jackpot odds for every game
  prove   [game]       run all strategies through the backtest; show they hug chance
  backtest <game> <strategy>
  wheel   <game> [-n N]   build a covering design over N numbers; show the guarantee
  payouts <game>       what each prize rank really pays, from the operator's records
  popular <game>       which numbers other players over- and under-pick
  ev      <game>       price a ticket: least- vs most-popular numbers, any jackpot
  sharing <game>       walk-forward test of the sharing effect on real payouts
  freq    <game>       frequency dashboard + chi-square uniformity test
  wait                 expected/median time to a 3-match per game
  variance <game>      Monte-Carlo net-result distribution for a single-ticket player
  fetch   [game ...]   download fresh official history as new snapshots
"""

from __future__ import annotations

import argparse
import sys

import requests

from . import analytics, ev, games, payouts, popularity, simulate, store
from .backtest import backtest
from .combinatorics import odds_table
from .sharing import sharing_backtest
from .strategy import BUILTIN_STRATEGIES, get_strategy
from .synth import synth_history
from .wheeling import cycled_specials, spread_numbers, wheel_report

MAX_WHEEL = 14  # covering designs grow fast beyond this


def _load(game: str, use_synth: bool, n: int = 1500):
    """Load either real canonical history or deterministic synthetic draws."""
    if use_synth:
        return synth_history(games.get(game), n, seed=0)
    return store.load_canonical(game)


def _millions(amount: float) -> str:
    return f"{amount / 1e6:,.1f}M"


def cmd_odds(_args):
    """Print exact match-3 and jackpot odds for every game."""
    print("Exact odds (a fair draw — every combination equally likely):\n")
    print(f"  {'Game':22s} {'Matrix':16s} {'Match 3':>12s} {'Jackpot':>18s}")
    for r in odds_table():
        print(
            f"  {r['game']:22s} {r['matrix']:16s} "
            f"{'1 in ' + str(r['match3_one_in']):>12s} "
            f"{'1 in ' + format(r['jackpot_one_in'], ','):>18s}"
        )


def cmd_prove(args):
    """Run every built-in strategy against one game and print z-scores."""
    game = args.game or "eurodreams"
    spec = payouts.calibrated_spec(game)
    hist = _load(game, args.synth, n=args.synth_n)
    print(
        f"Running every strategy through the backtest on {spec.name} "
        f"({'synthetic fair data' if args.synth else 'real history'}, "
        f"{len(hist)} draws).\n"
        "If the games are fair, all of them sit within ~2 standard errors of the "
        "chance line:\n"
    )
    print(f"  {'strategy':16s} {'>=3 hits':>9s} {'rate':>9s} {'z':>7s}   verdict")
    n_strats = len(BUILTIN_STRATEGIES)
    res = None
    for name in BUILTIN_STRATEGIES:
        res = backtest(get_strategy(name), hist, spec, n_tickets=args.tickets, seed=1)
        z = res.z_vs_baseline
        # An "edge" would be ABOVE baseline. With this many strategies, |z|~2 is
        # expected by chance, so only flag a large, positive surprise.
        if z > 3:
            verdict = "surprise (check)"
        elif z < -2:
            verdict = "below chance"
        else:
            verdict = "no edge"
        print(
            f"  {name:16s} {res.three_plus_hits:>9d} {res.three_plus_rate:>9.5f} "
            f"{z:>+7.2f}   {verdict}"
        )
    p3 = res.baseline_three_plus_rate
    print(
        f"\nBaseline (exact): >=3 main = {p3:.5f}. "
        "No strategy beats it — that's the whole point."
    )
    print(
        f"  (With {n_strats} strategies, a |z|~2 reading is expected by chance; "
        "what would matter is a large positive z that PERSISTS across seeds and "
        "games. It never does.)"
    )
    expected = res.n_draws * args.tickets * p3
    if expected < 10:
        hint = (
            "use --synth --synth-n 50000"
            if not args.synth
            else "raise --synth-n or try eurodreams (match-3 is 1 in 32)"
        )
        print(
            f"\n  Low-power note: only ~{res.n_draws} draws evaluated and match-3 is "
            f"rare here (~{expected:.1f} hits expected per strategy), so the "
            f"individual z-scores are noisy by nature. For a high-power demo, {hint}."
        )


def cmd_backtest(args):
    """Run one strategy through the backtest harness."""
    spec = payouts.calibrated_spec(args.game)
    hist = _load(args.game, args.synth, n=args.synth_n)
    res = backtest(
        get_strategy(args.strategy), hist, spec, n_tickets=args.tickets, seed=1
    )
    print(res)


def cmd_wheel(args):
    """Build and print a covering-design wheel for one game."""
    spec = payouts.calibrated_spec(args.game)
    if args.numbers:
        chosen = sorted(set(args.numbers))
        bad = [x for x in chosen if x < 1 or x > spec.main_max]
        if bad:
            sys.exit(
                f"Out of range for {spec.name} (main pool is 1-{spec.main_max}): {bad}"
            )
        if not spec.main_count <= len(chosen) <= MAX_WHEEL:
            sys.exit(
                f"Give between {spec.main_count} and {MAX_WHEEL} distinct numbers to "
                f"wheel {spec.name}; got {len(chosen)}. (Covering designs grow fast.)"
            )
        source = "your numbers"
    else:
        n = args.n
        if n > MAX_WHEEL:
            print(
                f"Heads up: covering designs grow fast; using {MAX_WHEEL} numbers "
                f"instead of {n}.",
                file=sys.stderr,
            )
            n = MAX_WHEEL
        if n < spec.main_count:
            sys.exit(f"Need at least {spec.main_count} numbers to wheel {spec.name}.")
        chosen = spread_numbers(n, 1, spec.main_max)
        source = (
            f"an even spread across 1-{spec.main_max} (pass your own with --numbers)"
        )

    report = wheel_report(spec, chosen)
    print(report)
    print(f"\n  wheeling {source}: {report.chosen}")
    if spec.special_count:
        print(
            f"  tickets ({report.n_tickets}) — each also needs "
            f"{spec.special_count} {spec.special_name}(s) from 1-{spec.special_max}; "
            f"cycled below so the block spreads that pick too:"
        )
        for i, tk in enumerate(report.tickets):
            print(f"    {tk}  +  {spec.special_name}: {cycled_specials(i, spec)}")
    else:
        print(f"  tickets ({report.n_tickets}):")
        for tk in report.tickets:
            print(f"    {tk}")
    three = spec.prize_table.get((3, 0), 0.0)
    print(
        f"\n  Honest note: the 3-match guarantee covers the {spec.main_count} main "
        f"numbers only, and holds only when >=3 of your chosen numbers are drawn. "
        f"The block costs {report.cost:,.2f} {spec.currency} per draw against a "
        f"3-match prize of about {three:,.2f}: it buys determinism, not profit."
    )


def cmd_payouts(args):
    """Print what each prize rank really pays, from operator records."""
    spec = payouts.calibrated_spec(args.game)
    frame = payouts.try_load(args.game)
    if frame is None:
        print(
            f"{spec.name} has no per-rank payout data on disk (only the FDJ games "
            "publish it; run `lotterylab fetch` if this is one). Static table:\n"
        )
        for (m, s), amount in sorted(spec.prize_table.items(), reverse=True):
            print(f"  {spec.matches_label(m, s):34s} {amount:>14,.2f} {spec.currency}")
        return
    summary = payouts.rank_summary(spec, frame)
    print(
        f"{spec.name}: what each rank paid over {len(frame)} draws "
        f"({frame['date'].iloc[0]} to {frame['date'].iloc[-1]}), per winner.\n"
    )
    print(
        f"  {'rank':>4s}  {'match':30s} {'odds':>16s} {'mean':>12s} "
        f"{'low':>10s} {'high':>12s}  kind"
    )
    for row in summary.itertuples():
        print(
            f"  {row.rank:>4d}  {row.label:30s} "
            f"{'1 in ' + format(round(1 / row.probability), ','):>16s} "
            f"{row.mean_payout:>12,.2f} {row.min_payout:>10,.2f} "
            f"{row.max_payout:>12,.2f}  {row.kind}"
        )
    tickets = payouts.typical_tickets(spec, frame)
    print(
        f"\n  Tickets per draw (recent median): ~{_millions(tickets)}"
        "  — winners of the most-won rank ÷ its exact odds."
    )
    sales = payouts.sales_model(spec, frame)
    if sales:
        print(
            f"  Sales grow with the jackpot: about +{sales[1] * 1e6:,.0f} tickets "
            f"per extra million on offer."
        )
    raffle = payouts.raffle_ev(spec, frame)
    if raffle:
        print(
            f"  Raffle: {spec.raffle_per_draw:,.0f} {spec.currency} per draw across "
            f"French tickets is worth ~{raffle:.3f} per ticket."
        )
    print(
        "\n  'shared' ranks split a pool among that draw's winners, so their payout "
        "depends on how many\n  others matched — see `lotterylab ev`. 'top' ranks "
        "(rollover jackpots, annuities) use the static figure."
    )


def cmd_popular(args):
    """Print which numbers other players over- and under-pick."""
    spec = games.get(args.game)
    model = popularity.load_model(args.game)
    ranked = popularity.ranked_numbers(model)
    print(f"Number popularity among {spec.name} players — {model.describe()}.\n")
    print("  1.00 = picked as often as average. Popularity says nothing about which")
    print("  numbers get drawn; it says how many people you'd share a pool with.\n")
    n = args.top
    print("  least picked: " + "  ".join(f"{num}({w:.2f})" for num, w in ranked[:n]))
    print(
        "  most picked : "
        + "  ".join(f"{num}({w:.2f})" for num, w in reversed(ranked[-n:]))
    )
    if model.elasticity:
        print("\n  How hard each shared rank's payout reacts (elasticity):")
        for rank, value in model.elasticity:
            print(
                f"    rank {rank:>2d}  {payouts.rank_label(spec, rank):30s} {value:.2f}"
            )


def cmd_ev(args):
    """Price tickets under the measured market: popularity and jackpot size."""
    mkt = ev.market(args.game)
    spec = mkt.spec
    least, most = ev.best_and_worst(mkt)
    tickets = {"least-popular numbers": least, "most-popular numbers": most}
    if args.numbers:
        mine = tuple(sorted(set(args.numbers)))
        if len(mine) != spec.main_count or not all(
            1 <= x <= spec.main_max for x in mine
        ):
            sys.exit(
                f"--numbers needs {spec.main_count} distinct numbers from "
                f"1-{spec.main_max} for {spec.name}."
            )
        tickets = {"your numbers": mine, **tickets}
    jackpot = args.jackpot if args.jackpot else spec.jackpot_estimate

    print(f"Same odds of winning, different expected payout ({spec.name}):\n")
    for label, main in tickets.items():
        print(f"[{label}]")
        print(ev.evaluate(main, mkt, jackpot=jackpot))
        print()

    if spec.jackpot_rolls:
        print("Return per unit staked as the jackpot grows (sales grow with it):\n")
        seen = (
            payouts.jackpot_history(spec, mkt.frame)["jackpot"].max()
            if mkt.measured
            else None
        )
        top = spec.jackpot_cap or seen or 20 * spec.jackpot_estimate
        top = max(top, jackpot)
        steps = [
            spec.jackpot_estimate + (top - spec.jackpot_estimate) * i / 4
            for i in range(5)
        ]
        header = (
            "  " + f"{'jackpot':>10s}" + "".join(f"{label:>24s}" for label in tickets)
        )
        print(header)
        for amount in steps:
            cells = "".join(
                f"{ev.evaluate(main, mkt, jackpot=amount).return_rate:>24.1%}"
                for main in tickets.values()
            )
            print("  " + f"{_millions(amount):>10s}" + cells)
        print()
        for label, main in tickets.items():
            even = ev.breakeven_jackpot(main, mkt)
            if even is None:
                note = "never reaches break-even"
            elif spec.jackpot_cap and even > spec.jackpot_cap:
                note = (
                    f"would need {_millions(even)} — above the "
                    f"{_millions(spec.jackpot_cap)} cap, so never"
                )
            else:
                note = f"breaks even at about {_millions(even)}"
            print(f"  {label}: {note}")
        print()
    print(
        "Every ticket above has identical odds; only who you'd share with differs.\n"
        "Read 'break-even' as an average over millions of draws: almost all of that "
        "value sits\nin a jackpot you will almost certainly never hit, so the typical "
        "result is still a loss."
    )
    if not mkt.measured:
        print(
            "\n(No operator payout data for this game: prizes are the static table "
            "and popularity is a heuristic.)"
        )


def cmd_sharing(args):
    """Walk-forward test: did neglected numbers really collect more money?"""
    spec = games.get(args.game)
    try:
        result = sharing_backtest(args.game)
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"Cannot run the sharing test for {spec.name}: {error}")
    print(result)
    print(
        "\n  Every group wins equally often; this compares what a win was worth. "
        "'realised' uses the\n  operator's actual payouts for draws the fit never "
        "saw; 'predicted' is the model's forecast."
    )


def cmd_freq(args):
    """Print frequency bars and a chi-square uniformity test."""
    spec = games.get(args.game)
    hist = _load(args.game, args.synth, n=args.synth_n)
    print(f"Frequency dashboard — {spec.name} ({len(hist)} draws)\n")
    print(analytics.ascii_bars(hist, spec))


def cmd_wait(_args):
    """Print expected and median wait for a match-3 hit by game."""
    print("Expected time until a single ticket finally matches 3 main numbers:\n")
    for spec in games.all_games():
        print(simulate.time_to_match3(spec))
        print()


def cmd_variance(args):
    """Print Monte-Carlo season variance for one game."""
    spec = payouts.calibrated_spec(args.game)
    print(simulate.simulate_variance(spec, draws=args.draws, n_seasons=args.seasons))


def cmd_fetch(args):
    """Download fresh official history for the given games (default: all).

    Every game is attempted; the command then exits non-zero if any of them
    failed, so a scheduled run can't look healthy while one source is down.
    """
    targets = args.games or list(store.FETCHABLE)
    failures = 0
    for game in targets:
        before = store.newest_snapshot(game)
        try:
            path = store.fetch_raw(game)
        except (OSError, ValueError, requests.RequestException) as error:
            failures += 1
            print(f"  {game:13s} FAILED  {error}")
            continue
        draws = store.load_canonical(game)
        state = "unchanged" if path == before else "new snapshot"
        print(
            f"  {game:13s} {state:13s} {len(draws):>5d} draws through "
            f"{draws['date'].iloc[-1]}  ({path.rsplit('/', 1)[-1]})"
        )
    if failures:
        sys.exit(f"{failures} of {len(targets)} sources could not be fetched.")


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse command tree."""
    p = argparse.ArgumentParser(
        prog="lotterylab",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)
    game_keys = list(games.GAMES)

    def add_common(sp, synth_default=False):
        """Add common real/synthetic data flags to a subcommand."""
        sp.add_argument(
            "--synth",
            action="store_true",
            default=synth_default,
            help="use provably-fair synthetic data instead of real history",
        )
        sp.add_argument(
            "--synth-n",
            type=int,
            default=1500,
            help="number of synthetic draws (with --synth)",
        )

    sp = sub.add_parser("odds")
    sp.set_defaults(func=cmd_odds)

    sp = sub.add_parser("prove")
    sp.add_argument("game", nargs="?", choices=game_keys)
    sp.add_argument("--tickets", type=int, default=1)
    add_common(sp)
    sp.set_defaults(func=cmd_prove)

    sp = sub.add_parser("backtest")
    sp.add_argument("game", choices=game_keys)
    sp.add_argument("strategy", choices=list(BUILTIN_STRATEGIES))
    sp.add_argument("--tickets", type=int, default=1)
    add_common(sp)
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("wheel")
    sp.add_argument("game", choices=game_keys)
    sp.add_argument(
        "-n",
        type=int,
        default=9,
        help="how many numbers to wheel (default 9, spread across the pool)",
    )
    sp.add_argument(
        "--numbers",
        type=int,
        nargs="+",
        help="explicit numbers to wheel, e.g. --numbers 4 11 19 27 35 38",
    )
    sp.set_defaults(func=cmd_wheel)

    sp = sub.add_parser("payouts")
    sp.add_argument("game", choices=game_keys)
    sp.set_defaults(func=cmd_payouts)

    sp = sub.add_parser("popular")
    sp.add_argument("game", choices=game_keys)
    sp.add_argument("--top", type=int, default=10, help="how many numbers to list")
    sp.set_defaults(func=cmd_popular)

    sp = sub.add_parser("ev")
    sp.add_argument("game", choices=game_keys)
    sp.add_argument("--numbers", type=int, nargs="+", help="your main numbers")
    sp.add_argument("--jackpot", type=float, help="advertised jackpot to price at")
    sp.set_defaults(func=cmd_ev)

    sp = sub.add_parser("sharing")
    sp.add_argument("game", choices=list(store.FDJ_SOURCES))
    sp.set_defaults(func=cmd_sharing)

    sp = sub.add_parser("freq")
    sp.add_argument("game", choices=game_keys)
    add_common(sp)
    sp.set_defaults(func=cmd_freq)

    sp = sub.add_parser("wait")
    sp.set_defaults(func=cmd_wait)

    sp = sub.add_parser("variance")
    sp.add_argument("game", choices=game_keys)
    sp.add_argument("--draws", type=int, default=104)
    sp.add_argument("--seasons", type=int, default=2000)
    sp.set_defaults(func=cmd_variance)

    sp = sub.add_parser("fetch")
    sp.add_argument("games", nargs="*", choices=[*store.FETCHABLE, []])
    sp.set_defaults(func=cmd_fetch)

    return p


def main(argv=None):
    """Parse command-line arguments and dispatch the selected command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
