"""Lottery Lab — an honest statistical sandbox.

You cannot predict a fair draw. This package proves that empirically (the backtest
harness shows every "strategy" hugging the chance line) and then implements the
only mechanisms that genuinely exist: covering designs (wheeling) that *guarantee*
a low-tier hit at a known cost, and an expected-value model for jackpot sharing.
"""

from .games import GAMES, GameSpec, get

__all__ = ["GAMES", "GameSpec", "get"]
__version__ = "0.1.0"
