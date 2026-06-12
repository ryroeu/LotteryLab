"""Strategies — functions of the form ``history -> ticket(s)``.

Every built-in here exists to be run through the backtest harness and shown to be
statistically indistinguishable from random. That includes ``OrderStatMean``, the
fixed vector the deleted LSTM converged to, and ``LSTMGhost``, an alias documenting
that the original deep-learning approach was mathematically equivalent to it.

A ticket is ``(main_tuple, special_tuple)``. ``generate`` may only look at draws
that occurred before the draw being predicted — the harness enforces this by
slicing history; strategies must not peek further.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .combinatorics import order_statistic_means
from .games import GameSpec
from .schema import main_columns, special_columns
from .validate import Ticket


def _rand_pool(
    rng: np.random.Generator, lo: int, hi: int, count: int
) -> tuple[int, ...]:
    return tuple(
        int(x) for x in rng.choice(np.arange(lo, hi + 1), count, replace=False)
    )


def _frequencies(history: pd.DataFrame, cols: list[str], pool_max: int) -> np.ndarray:
    counts = np.zeros(pool_max + 1, dtype=float)  # index 0 unused
    if not history.empty:
        vals = history[cols].to_numpy().ravel()
        for v in vals:
            counts[int(v)] += 1
    return counts


class Strategy:
    """Base contract for a no-lookahead lottery ticket generator."""

    name = "strategy"

    def generate(
        self,
        history: pd.DataFrame,
        spec: GameSpec,
        n_tickets: int,
        rng: np.random.Generator,
    ) -> list[Ticket]:
        """Return valid tickets using only the supplied historical draws."""
        raise NotImplementedError

    def special_pick(self, spec: GameSpec, rng: np.random.Generator) -> tuple[int, ...]:
        """Return a valid special-ball pick for ``spec``."""
        if not spec.special_count:
            return ()
        return _rand_pool(rng, 1, spec.special_max, spec.special_count)


class RandomPlayer(Strategy):
    """Pick every ticket uniformly at random."""

    name = "random"

    def generate(self, history, spec, n_tickets, rng):
        out = []
        for _ in range(n_tickets):
            main = _rand_pool(rng, 1, spec.main_max, spec.main_count)
            out.append((main, self.special_pick(spec, rng)))
        return out


class HotNumbers(Strategy):
    """Pick the most historically frequent numbers — the gambler's 'hot' fallacy."""

    name = "hot"

    def generate(self, history, spec, n_tickets, rng):
        freq = _frequencies(history, main_columns(spec), spec.main_max)
        if freq[1:].sum() == 0:
            return RandomPlayer().generate(history, spec, n_tickets, rng)
        order = np.argsort(freq[1:])[::-1] + 1  # hottest first
        main = tuple(sorted(int(x) for x in order[: spec.main_count]))
        return [(main, self.special_pick(spec, rng)) for _ in range(n_tickets)]


class ColdNumbers(Strategy):
    """Pick the least frequent / 'overdue' numbers — the other half of the fallacy."""

    name = "cold"

    def generate(self, history, spec, n_tickets, rng):
        freq = _frequencies(history, main_columns(spec), spec.main_max)
        if freq[1:].sum() == 0:
            return RandomPlayer().generate(history, spec, n_tickets, rng)
        order = np.argsort(freq[1:]) + 1  # coldest first
        main = tuple(sorted(int(x) for x in order[: spec.main_count]))
        return [(main, self.special_pick(spec, rng)) for _ in range(n_tickets)]


class LastDrawEcho(Strategy):
    """Replay the previous draw's numbers (the 'it just came up' fallacy)."""

    name = "last_echo"

    def generate(self, history, spec, n_tickets, rng):
        if history.empty:
            return RandomPlayer().generate(history, spec, n_tickets, rng)
        last = history.iloc[-1]
        main = tuple(sorted(int(last[c]) for c in main_columns(spec)))
        scols = special_columns(spec)
        special = tuple(sorted(int(last[c]) for c in scols)) if scols else ()
        return [(main, special) for _ in range(n_tickets)]


class OrderStatMean(Strategy):
    """The fixed order-statistic-mean vector an MSE regressor converges to.

    This is, mathematically, what the deleted LSTM was learning: a constant ticket
    of per-position averages. Included precisely to show it is no better than random.
    """

    name = "order_stat_mean"

    def generate(self, history, spec, n_tickets, rng):
        means = order_statistic_means(spec.main_max, spec.main_count)
        main = self._dedupe_round(means, spec.main_max, spec.main_count)
        smeans = (
            order_statistic_means(spec.special_max, spec.special_count)
            if spec.special_count
            else []
        )
        special = self._dedupe_round(smeans, spec.special_max, spec.special_count)
        return [(main, special) for _ in range(n_tickets)]

    @staticmethod
    def _dedupe_round(means, pool_max, count) -> tuple[int, ...]:
        # Round, then repair collisions/out-of-range so the ticket is *valid*
        # (the old predictors skipped this and could emit impossible tickets).
        chosen: list[int] = []
        for m in means:
            v = int(round(m))
            v = max(1, min(pool_max, v))
            while v in chosen:
                v += 1
                if v > pool_max:
                    v = 1
            chosen.append(v)
        return tuple(sorted(chosen[:count]))


class LSTMGhost(OrderStatMean):
    """Documentation alias: the repo's original LSTM == OrderStatMean, no better."""

    name = "lstm_ghost"


class BiasedHigh(Strategy):
    """Avoid 'birthday' numbers (<=31): same win odds, but see ev.py for why this
    can raise expected *payout* in pari-mutuel games by reducing jackpot sharing."""

    name = "biased_high"

    def generate(self, history, spec, n_tickets, rng):
        lo = min(32, spec.main_max - spec.main_count + 1)
        out = []
        for _ in range(n_tickets):
            main = _rand_pool(rng, lo, spec.main_max, spec.main_count)
            out.append((main, self.special_pick(spec, rng)))
        return out


BUILTIN_STRATEGIES: dict[str, type[Strategy]] = {
    s.name: s
    for s in [
        RandomPlayer,
        HotNumbers,
        ColdNumbers,
        LastDrawEcho,
        OrderStatMean,
        LSTMGhost,
        BiasedHigh,
    ]
}


def get_strategy(name: str) -> Strategy:
    """Instantiate a built-in strategy by name."""
    try:
        return BUILTIN_STRATEGIES[name]()
    except KeyError:
        raise KeyError(
            f"Unknown strategy {name!r}. Known: {', '.join(BUILTIN_STRATEGIES)}"
        ) from None
