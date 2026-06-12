"""Command-line interface: ``python -m lotterylab <command>``.

Commands
  odds                 exact match-3 and jackpot odds for every game
  prove   [game]       run all strategies through the backtest; show they hug chance
  backtest <game> <strategy>
  wheel   <game> [-n N]   build a covering design over N numbers; show the guarantee
  ev      <game>       compare a 'birthday' ticket vs a high-number ticket by EV
  freq    <game>       frequency dashboard + chi-square uniformity test
  wait                 expected/median time to a 3-match per game
  variance <game>      Monte-Carlo net-result distribution for a single-ticket player
"""

from __future__ import annotations

import argparse
import sys

from . import analytics, ev, games, simulate
from .backtest import backtest
from .combinatorics import odds_table
from .store import load_canonical
from .strategy import BUILTIN_STRATEGIES, get_strategy
from .synth import synth_history
from .wheeling import cycled_specials, spread_numbers, wheel_report


def _load(game: str, use_synth: bool, n: int = 1500):
    """Load either real canonical history or deterministic synthetic draws."""
    spec = games.get(game)
    if use_synth:
        return synth_history(spec, n, seed=0)
    return load_canonical(game)


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
    spec = games.get(game)
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
    p3 = backtest(get_strategy("random"), hist, spec).baseline_three_plus_rate
    print(
        f"\nBaseline (exact): >=3 main = {p3:.5f}. "
        "No strategy beats it — that's the whole point."
    )
    print(
        f"  (With {n_strats} strategies, a |z|~2 reading is expected by chance; "
        "what would matter is a large positive z that PERSISTS across seeds and "
        "games. It never does.)"
    )
    evaluated = max(0, len(hist) - 50)
    if evaluated * p3 < 10:
        hint = (
            "use --synth --synth-n 50000"
            if not args.synth
            else "raise --synth-n or try eurodreams (match-3 is 1 in 32)"
        )
        print(
            f"\n  Low-power note: only ~{evaluated} draws evaluated and match-3 is "
            f"rare here (~{evaluated * p3:.1f} hits expected per strategy), so the "
            f"individual z-scores are noisy by nature. For a high-power demo, {hint}."
        )


def cmd_backtest(args):
    """Run one strategy through the backtest harness."""
    spec = games.get(args.game)
    hist = _load(args.game, args.synth, n=args.synth_n)
    res = backtest(
        get_strategy(args.strategy), hist, spec, n_tickets=args.tickets, seed=1
    )
    print(res)


def cmd_wheel(args):
    """Build and print a covering-design wheel for one game."""
    spec = games.get(args.game)
    if args.numbers:
        chosen = sorted(set(args.numbers))
        bad = [x for x in chosen if x < 1 or x > spec.main_max]
        if bad:
            print(
                f"Out of range for {spec.name} (main pool is 1-{spec.main_max}): {bad}",
                file=sys.stderr,
            )
            return
        if len(chosen) < spec.main_count:
            print(
                f"Need at least {spec.main_count} numbers to wheel {spec.name}; "
                f"got {len(chosen)}.",
                file=sys.stderr,
            )
            return
        source = "your numbers"
    else:
        chosen = spread_numbers(args.n, 1, spec.main_max)
        source = f"an even spread across 1-{spec.main_max} (pass your own with --numbers)"

    if len(chosen) > 14:
        print(
            "Heads up: covering designs grow fast; >14 numbers can be slow. "
            "Trimming to first 14.",
            file=sys.stderr,
        )
        chosen = chosen[:14]

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
        print(
            f"\n  Honest note: the 3-match guarantee covers the {spec.main_count} main "
            f"numbers only — the {spec.special_name} (1-{spec.special_max}) is an "
            f"independent pick the wheel can't help with. And it holds only when >=3 "
            f"of your chosen numbers are drawn; the ticket cost exceeds a 3-match "
            f"prize, so it buys determinism, not profit."
        )
    else:
        print(f"  tickets ({report.n_tickets}):")
        for tk in report.tickets:
            print(f"    {tk}")
        print(
            "\n  Honest note: this guarantees a 3-match ONLY when >=3 of your chosen "
            "numbers are drawn, and the ticket cost exceeds a 3-match prize. It buys "
            "determinism, not profit."
        )


def cmd_ev(args):
    """Compare EV for birthday-heavy and high-number reference tickets."""
    spec = games.get(args.game)
    k = spec.main_count
    birthday = tuple(range(1, k + 1))  # 1,2,3,... all <= 31
    high = tuple(range(spec.main_max - k + 1, spec.main_max + 1))  # top numbers
    print(f"Same win probability, different expected payout ({spec.name}):\n")
    print(ev.ticket_ev(birthday, spec))
    print()
    print(ev.ticket_ev(high, spec))
    print(
        "\nProbability of winning is identical for both. Only the expected *share* "
        "of a jackpot differs — unpopular numbers get split fewer ways."
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
    spec = games.get(args.game)
    print(simulate.simulate_variance(spec, draws=args.draws, n_seasons=args.seasons))


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse command tree."""
    p = argparse.ArgumentParser(prog="lotterylab", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

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
    sp.add_argument("game", nargs="?", choices=list(games.GAMES))
    sp.add_argument("--tickets", type=int, default=1)
    add_common(sp)
    sp.set_defaults(func=cmd_prove)

    sp = sub.add_parser("backtest")
    sp.add_argument("game", choices=list(games.GAMES))
    sp.add_argument("strategy", choices=list(BUILTIN_STRATEGIES))
    sp.add_argument("--tickets", type=int, default=1)
    add_common(sp)
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("wheel")
    sp.add_argument("game", choices=list(games.GAMES))
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

    sp = sub.add_parser("ev")
    sp.add_argument("game", choices=list(games.GAMES))
    sp.set_defaults(func=cmd_ev)

    sp = sub.add_parser("freq")
    sp.add_argument("game", choices=list(games.GAMES))
    add_common(sp)
    sp.set_defaults(func=cmd_freq)

    sp = sub.add_parser("wait")
    sp.set_defaults(func=cmd_wait)

    sp = sub.add_parser("variance")
    sp.add_argument("game", choices=list(games.GAMES))
    sp.add_argument("--draws", type=int, default=104)
    sp.add_argument("--seasons", type=int, default=2000)
    sp.set_defaults(func=cmd_variance)

    return p


def main(argv=None):
    """Parse command-line arguments and dispatch the selected command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
