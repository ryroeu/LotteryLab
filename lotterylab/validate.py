"""Ticket / draw validity — the constraint the old ``np.round`` predictors ignored.

A real ticket has the right count of balls, every ball in range, and no duplicates
within a pool. These functions are the single boundary used by adapters, strategy
outputs, and the backtest harness, so an impossible ticket can never slip through.
"""

from __future__ import annotations

from numbers import Integral

from .games import GameSpec

Ticket = tuple[tuple[int, ...], tuple[int, ...]]  # (main, special)


class InvalidTicket(ValueError):
    """Raised when a ticket cannot be played under a game's matrix."""


def check_pool(values, count: int, lo: int, hi: int, label: str) -> None:
    """Validate one pool of main or special numbers."""
    vals = list(values)
    if len(vals) != count:
        raise InvalidTicket(f"{label}: expected {count} numbers, got {len(vals)}")
    if len(set(vals)) != count:
        raise InvalidTicket(f"{label}: duplicate numbers in {vals}")
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, Integral):
            raise InvalidTicket(f"{label}: {v!r} is not an integer")
        if v < lo or v > hi:
            raise InvalidTicket(f"{label}: {v} out of range [{lo}, {hi}]")


def validate_ticket(main, special, spec: GameSpec) -> None:
    """Raise ``InvalidTicket`` if the ticket is not playable for ``spec``."""
    check_pool(main, spec.main_count, 1, spec.main_max, f"{spec.key} main")
    check_pool(
        special,
        spec.special_count,
        1,
        spec.special_max,
        f"{spec.key} {spec.special_name}",
    )


def is_valid_ticket(main, special, spec: GameSpec) -> bool:
    """Return whether a ticket is playable for ``spec``."""
    try:
        validate_ticket(main, special, spec)
        return True
    except InvalidTicket:
        return False
