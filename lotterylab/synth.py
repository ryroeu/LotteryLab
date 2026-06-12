"""Synthetic draw generator — provably-fair data for offline tests and demos.

Because these draws are uniform by construction, any strategy run against them
MUST land on the chance line (|z| small) and ROI must match the prize-table
expectation. That makes the synthesizer a self-checking ground truth for the
whole harness, and lets everything run without a network.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from .games import GameSpec
from .schema import Draw, draws_to_frame


def synth_history(spec: GameSpec, n_draws: int, seed: int = 0) -> pd.DataFrame:
    """Generate uniform synthetic draws for one game as a canonical frame."""
    rng = np.random.default_rng(seed)
    start = _dt.date(2018, 1, 1)
    draws = []
    for i in range(n_draws):
        main = rng.choice(
            np.arange(1, spec.main_max + 1), spec.main_count, replace=False
        )
        if spec.special_count:
            special = rng.choice(
                np.arange(1, spec.special_max + 1), spec.special_count, replace=False
            )
        else:
            special = np.array([], dtype=int)
        draws.append(
            Draw(
                game=spec.key,
                date=start + _dt.timedelta(days=3 * i),
                draw_id=str(i),
                main=tuple(int(x) for x in main),
                special=tuple(int(x) for x in special),
            )
        )
    return draws_to_frame(draws, spec)
