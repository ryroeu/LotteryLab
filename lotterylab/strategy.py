"""Strategies — functions of the form ``past -> ticket(s)``.

Every built-in here exists to be run through the backtest harness and shown to be
statistically indistinguishable from random. That includes ``OrderStatMean``, the
fixed vector the deleted LSTM converged to, and ``LSTMGhost``, an alias documenting
that the original deep-learning approach was mathematically equivalent to it.

A ticket is ``(main_tuple, special_tuple)``. ``generate`` receives a ``Past``: what
was knowable before the draw being predicted. The harness builds it incrementally
(running number counts, the previous draw), so a strategy reading those costs
nothing per draw; ``past.frame`` is still there for anything else and is sliced to
the same no-lookahead boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .combinatorics import order_statistic_means
from .games import GameSpec
from .schema import main_columns, special_columns
from .validate import Ticket


@dataclass
class Past:
    """Everything a strategy may know before a draw — and nothing after it."""

    n_draws: int
    main_counts: np.ndarray  # main_counts[n] = times number n was drawn so far
    last_main: tuple[int, ...] | None = None
    last_special: tuple[int, ...] | None = None
    history: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def empty(self) -> bool:
        """True before the first draw."""
        return self.n_draws == 0

    @property
    def frame(self) -> pd.DataFrame:
        """The past draws as a tidy frame (sliced; never includes the future)."""
        if self.history is None:
            raise ValueError("this Past was built without a history frame")
        return self.history.iloc[: self.n_draws]

    @classmethod
    def from_frame(cls, history: pd.DataFrame, spec: GameSpec) -> "Past":
        """Summarise a whole history frame as the past of the *next* draw."""
        mains = history[main_columns(spec)].to_numpy(dtype=int)
        specials = history[special_columns(spec)].to_numpy(dtype=int)
        counts = np.bincount(mains.ravel(), minlength=spec.main_max + 1)
        has = len(history) > 0
        return cls(
            n_draws=len(history),
            main_counts=counts,
            last_main=tuple(sorted(mains[-1].tolist())) if has else None,
            last_special=tuple(sorted(specials[-1].tolist())) if has else None,
            history=history,
        )


def as_past(history, spec: GameSpec) -> Past:
    """Accept either a ``Past`` or a plain history frame."""
    return history if isinstance(history, Past) else Past.from_frame(history, spec)


def _rand_pool(
    rng: np.random.Generator, lo: int, hi: int, count: int
) -> tuple[int, ...]:
    return tuple(
        int(x) for x in rng.choice(np.arange(lo, hi + 1), count, replace=False)
    )


class Strategy:
    """Base contract for a no-lookahead lottery ticket generator."""

    name = "strategy"

    def generate(
        self,
        past: Past | pd.DataFrame,
        spec: GameSpec,
        n_tickets: int,
        rng: np.random.Generator,
    ) -> list[Ticket]:
        """Return valid tickets using only draws that have already happened."""
        raise NotImplementedError

    def special_pick(self, spec: GameSpec, rng: np.random.Generator) -> tuple[int, ...]:
        """Return a valid special-ball pick for ``spec``."""
        if not spec.special_count:
            return ()
        return _rand_pool(rng, 1, spec.special_max, spec.special_count)


class RandomPlayer(Strategy):
    """Pick every ticket uniformly at random."""

    name = "random"

    def generate(self, past, spec, n_tickets, rng):
        out = []
        for _ in range(n_tickets):
            main = _rand_pool(rng, 1, spec.main_max, spec.main_count)
            out.append((main, self.special_pick(spec, rng)))
        return out


class _Ranked(Strategy):
    """Shared machinery for 'hot' and 'cold': rank numbers by how often they have
    come up, then deal tickets down the ranking (ticket 0 takes the first
    ``main_count`` numbers, ticket 1 the next, wrapping round the pool)."""

    hottest_first = True

    def generate(self, past, spec, n_tickets, rng):
        past = as_past(past, spec)
        counts = past.main_counts[1:]
        if counts.sum() == 0:
            return RandomPlayer().generate(past, spec, n_tickets, rng)
        order = np.argsort(-counts if self.hottest_first else counts, kind="stable") + 1
        k, pool = spec.main_count, spec.main_max
        tickets = []
        for j in range(n_tickets):
            picks = [int(order[(j * k + i) % pool]) for i in range(k)]
            tickets.append((tuple(sorted(picks)), self.special_pick(spec, rng)))
        return tickets


class HotNumbers(_Ranked):
    """Pick the most historically frequent numbers — the gambler's 'hot' fallacy."""

    name = "hot"
    hottest_first = True


class ColdNumbers(_Ranked):
    """Pick the least frequent / 'overdue' numbers — the other half of the fallacy."""

    name = "cold"
    hottest_first = False


class LastDrawEcho(Strategy):
    """Replay the previous draw's numbers (the 'it just came up' fallacy)."""

    name = "last_echo"

    def generate(self, past, spec, n_tickets, rng):
        past = as_past(past, spec)
        if past.empty or past.last_main is None:
            return RandomPlayer().generate(past, spec, n_tickets, rng)
        special = past.last_special if spec.special_count else ()
        return [(past.last_main, special) for _ in range(n_tickets)]


class OrderStatMean(Strategy):
    """The fixed order-statistic-mean vector an MSE regressor converges to.

    This is, mathematically, what the deleted LSTM was learning: a constant ticket
    of per-position averages. Included precisely to show it is no better than random.
    """

    name = "order_stat_mean"

    def generate(self, past, spec, n_tickets, rng):
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
    can raise expected *payout* in shared-pool games by reducing prize sharing."""

    name = "biased_high"

    def generate(self, past, spec, n_tickets, rng):
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
