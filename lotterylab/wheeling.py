"""Wheeling / covering designs — the only mechanism that *guarantees* a 3-match.

A lottery wheel is a covering design. From a chosen pool of K numbers, generate a
set of k-number tickets so that EVERY t-subset of your K appears on at least one
ticket. The guarantee: *if* t of your K numbers are among the drawn balls, at least
one ticket is guaranteed to match t of them.

Crucial honesty:
  * It does NOT change the draw, the per-ticket win probability, or per-ticket EV.
  * The guarantee is conditional on >= t of your K actually being drawn.
  * Guaranteeing a 3-match every single draw means covering the *whole* pool, which
    costs far more in tickets than a 3-match pays. Wheeling buys determinism /
    lower variance at a known cost, not profit.

EuroDreams (6/40, match-3 = 1 in 32) is the friendliest target by an order of
magnitude, so it is the showcase game.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .combinatorics import hypergeom_pmf
from .games import GameSpec


def spread_numbers(n: int, lo: int, hi: int) -> list[int]:
    """n distinct numbers evenly spread across [lo, hi] — a neutral default pool."""
    if n >= hi - lo + 1:
        return list(range(lo, hi + 1))
    out: list[int] = []
    for i in range(n):
        v = round(lo + (hi - lo) * i / (n - 1))
        while v in out:  # nudge off a rounding collision
            v += 1
        out.append(min(v, hi))
    return sorted(set(out))


def cycled_specials(i: int, spec: GameSpec) -> tuple[int, ...]:
    """A deterministic, valid set of special balls for display ticket ``i``.

    The wheel only covers main numbers; cycling the special pick across the block
    at least spreads that independent choice too.
    """
    picks: list[int] = []
    v = i
    while len(picks) < spec.special_count:
        cand = (v % spec.special_max) + 1
        if cand not in picks:
            picks.append(cand)
        v += 1
    return tuple(sorted(picks))


def covering_design(numbers: list[int], k: int, t: int = 3) -> list[tuple[int, ...]]:
    """Greedy (v, k, t) cover: pick k-subsets until every t-subset of ``numbers``
    is covered. Greedy is not minimal (minimal covers are NP-hard; see the La Jolla
    Covering Repository for optimal numbers) but is correct — the guarantee holds.
    """
    numbers = sorted(numbers)
    if len(numbers) < k:
        raise ValueError(f"Need at least k={k} numbers to wheel, got {len(numbers)}.")
    uncovered = set(combinations(numbers, t))
    candidates = list(combinations(numbers, k))
    tickets: list[tuple[int, ...]] = []

    while uncovered:
        # choose the k-subset covering the most still-uncovered t-subsets
        best = candidates[0]
        best_gain = -1
        for cand in candidates:
            gain = sum(1 for ts in combinations(cand, t) if ts in uncovered)
            if gain > best_gain:
                best, best_gain = cand, gain
                if best_gain == len(uncovered):
                    break
        tickets.append(best)
        for ts in combinations(best, t):
            uncovered.discard(ts)
    return tickets


def verify_guarantee(tickets, numbers: list[int], t: int = 3) -> bool:
    """Brute-force proof: every t-subset of ``numbers`` is covered by some ticket."""
    covered = set()
    for tk in tickets:
        for ts in combinations(tk, t):
            covered.add(ts)
    return all(ts in covered for ts in combinations(sorted(numbers), t))


@dataclass
class WheelReport:
    """A verified wheel plus its cost and conditional guarantee statistics."""

    game: str
    chosen: tuple[int, ...]
    k: int
    t: int
    tickets: list[tuple[int, ...]]
    n_tickets: int
    cost: float
    currency: str
    guarantee: str
    p_condition: float  # P(>= t of chosen numbers are drawn)
    expected_tmatch_per_draw: float
    expected_draws_between_hits: float

    def __str__(self) -> str:
        return (
            f"Wheel for {self.game}: {len(self.chosen)} numbers {self.chosen}\n"
            f"  tickets         : {self.n_tickets}  (k={self.k} per ticket)\n"
            f"  cost / draw     : {self.cost:,.2f} {self.currency}\n"
            f"  guarantee       : {self.guarantee}\n"
            f"  P(condition met): {self.p_condition:.4f}  "
            f"(>= {self.t} of your {len(self.chosen)} numbers drawn)\n"
            f"  exp. {self.t}-match hits/draw : {self.expected_tmatch_per_draw:.4f}  "
            f"(~1 every {self.expected_draws_between_hits:.1f} draws)"
        )


def wheel_report(
    spec: GameSpec, chosen: list[int], *, t: int = 3, verify: bool = True
) -> WheelReport:
    """Build a covering design report for a chosen main-number pool."""
    k = spec.main_count
    tickets = covering_design(chosen, k, t)
    if verify and not verify_guarantee(tickets, chosen, t):
        raise AssertionError("Covering design failed its own guarantee — bug.")

    # P(>= t of your chosen numbers are among the spec.main_count drawn balls):
    # hypergeometric over the chosen subset.
    p_condition = sum(
        hypergeom_pmf(spec.main_max, spec.main_count, len(chosen), h)
        for h in range(t, min(len(chosen), spec.main_count) + 1)
    )
    # When the condition is met, the wheel guarantees a t-match ticket exists.
    exp_per_draw = p_condition
    between = 1 / exp_per_draw if exp_per_draw else float("inf")

    return WheelReport(
        game=spec.key,
        chosen=tuple(sorted(chosen)),
        k=k,
        t=t,
        tickets=tickets,
        n_tickets=len(tickets),
        cost=len(tickets) * spec.price,
        currency=spec.currency,
        guarantee=(
            f"if >= {t} of your {len(chosen)} numbers are drawn, "
            f">= 1 ticket matches >= {t}"
        ),
        p_condition=p_condition,
        expected_tmatch_per_draw=exp_per_draw,
        expected_draws_between_hits=between,
    )
