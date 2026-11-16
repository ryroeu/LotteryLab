"""Which numbers do other players pick? — measured, not assumed.

Picking unpopular numbers never changes the chance of winning; it changes how many
people you share a prize pool with. FDJ's history lets us measure that: every
draw records how many tickets matched 2, 3, 4... numbers, and those counts swing
with *which* numbers came up. Draws full of popular numbers produce more winners.

The model: players choose main numbers with relative weights ``w_i`` (mean 1). To
first order, the count of tickets matching exactly ``h`` of a drawn set ``D`` is

    log W(h, D) = log(tickets) + log(p_h) + c_h + g_h * sum(log w_i for i in D)

with ``g_h = h/k - (k-h)/(N-k)`` (matching more numbers leans harder on their
popularity). Ticket sales are unknown per draw, so each draw gets its own free
intercept and the weights are identified from how the *ratio* between ranks moves
— which is why only ranks sharing the same special-ball condition are compared.
A ridge penalty keeps rarely-drawn numbers from over-fitting.

Counts say who picks what; money is what matters. So a second step measures, for
each shared prize rank, how strongly its *payout* responds to the popularity of the
drawn numbers (its ``elasticity``). That response is often steeper than the count
model alone implies — in EuroDreams, popular numbers also trigger more fixed
refunds, shrinking the pool left for the shared ranks. The fit is scored by how
well it predicts payouts in the most recent draws, which it never saw.

This is a model of the crowd, not of the draw. It says nothing about which numbers
will come up — only about how crowded a prize tier will be if they do.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb

import numpy as np
import pandas as pd

from .games import GameSpec, get
from .payouts import rank_probability, try_load
from .schema import main_columns
from .store import newest_snapshot
from .validate import check_pool

# Heuristic over-pick multiplier for "birthday" numbers, used only when a game
# publishes no winner counts to fit against.
BIRTHDAY_WEIGHT = 1.8
BIRTHDAY_MAX = 31

RIDGE = 2.0  # penalty on log-weights, in units of one observation
MODEL_NOISE = 0.03  # log-count error floor so huge ranks don't swamp the fit
HOLDOUT = 0.2  # most recent share of draws held out to score the fit
MIN_DRAWS = 60


@dataclass(frozen=True)
class PopularityModel:
    """Relative popularity of each main number among other players (mean 1)."""

    game: str
    weights: tuple[float, ...]  # weights[n - 1] is number n
    source: str  # "fitted", "heuristic" or "uniform"
    n_draws: int = 0
    skill: float | None = None  # out-of-sample R^2 of predicted payouts
    # ((rank, elasticity), ...): d log(payout) / d log(drawn-number popularity)
    elasticity: tuple[tuple[int, float], ...] = ()

    def weight(self, number: int) -> float:
        """Relative popularity of one main number (1.0 = average)."""
        return self.weights[number - 1]

    def elasticity_for(self, rank: int | None) -> float | None:
        """Measured payout elasticity of a prize rank, if this model has one."""
        return dict(self.elasticity).get(rank)

    def describe(self) -> str:
        """One line on where these weights came from."""
        if self.source == "fitted":
            skill = (
                ""
                if self.skill is None
                else f"; predicts {self.skill:.0%} of payout variation out of sample"
            )
            return f"fitted to FDJ winner counts from {self.n_draws} draws{skill}"
        if self.source == "heuristic":
            return (
                f"heuristic: numbers ≤{BIRTHDAY_MAX} assumed {BIRTHDAY_WEIGHT}× "
                "over-picked (no winner-count data for this game)"
            )
        return "uniform (every number equally popular)"


def _normalised(weights) -> tuple[float, ...]:
    arr = np.asarray(weights, dtype=float)
    return tuple(float(x) for x in arr / arr.mean())


def uniform_model(spec: GameSpec) -> PopularityModel:
    """Every number equally popular: no sharing edge anywhere."""
    return PopularityModel(spec.key, (1.0,) * spec.main_max, "uniform")


def heuristic_model(spec: GameSpec) -> PopularityModel:
    """The birthday-number guess, for games with no winner-count data."""
    raw = [
        BIRTHDAY_WEIGHT if n <= BIRTHDAY_MAX else 1.0
        for n in range(1, spec.main_max + 1)
    ]
    return PopularityModel(spec.key, _normalised(raw), "heuristic")


# --- Fitting ---------------------------------------------------------------------


def hit_coefficient(spec: GameSpec, hits: int) -> float:
    """``g_h``: how strongly a rank with ``hits`` mains responds to the drawn
    numbers' popularity (see the module docstring)."""
    k, n = spec.main_count, spec.main_max
    return hits / k - (k - hits) / (n - k)


def comparable_ranks(spec: GameSpec) -> list[tuple[int, int]]:
    """``(rank, main_hits)`` for the ranks used to fit popularity: one special-ball
    condition (none matched, or 'any'), at least two mains, most ranks wins."""
    groups: dict[str, list[tuple[int, int]]] = {"none": [], "any": []}
    for rank, tiers in enumerate(spec.rank_tiers, start=1):
        mains = {m for m, _ in tiers}
        specials = {s for _, s in tiers}
        if len(mains) != 1 or min(mains) < 2:
            continue
        if specials == set(range(spec.special_count + 1)):
            groups["any"].append((rank, min(mains)))
        elif specials == {0}:
            groups["none"].append((rank, min(mains)))
    return max(groups.values(), key=len)


def _observations(spec: GameSpec, frame: pd.DataFrame):
    """Stack (draw, rank) observations: log relative count, coefficient, weight."""
    ranks = comparable_ranks(spec)
    drawn = frame[main_columns(spec)].to_numpy(dtype=int)
    draw_idx, rank_idx, y, g, w = [], [], [], [], []
    for j, (rank, hits) in enumerate(ranks):
        counts = (
            frame[f"r{rank}_winners"].fillna(frame[f"r{rank}_winners_fr"]).to_numpy()
        )
        log_p = np.log(rank_probability(spec, rank))
        for d, count in enumerate(counts):
            if not np.isfinite(count) or count <= 0:
                continue
            draw_idx.append(d)
            rank_idx.append(j)
            y.append(np.log(count) - log_p)
            g.append(hit_coefficient(spec, hits))
            w.append(1.0 / (1.0 / count + MODEL_NOISE**2))
    return (
        drawn,
        np.array(draw_idx),
        np.array(rank_idx),
        np.array(y),
        np.array(g),
        np.array(w),
        len(ranks),
    )


def _solve(spec, drawn, draw_idx, rank_idx, y, g, w, n_ranks, *, ridge):
    """Weighted ridge regression. Returns (log_weights, rank_offsets).

    ``ridge=None`` fits the null model (no popularity effect) for comparison.
    """
    draws = np.unique(draw_idx)
    position = {d: i for i, d in enumerate(draws)}
    n_obs, n_draw, n_num = len(y), len(draws), spec.main_max
    use_beta = ridge is not None
    width = n_draw + (n_ranks - 1) + (n_num if use_beta else 0)
    design = np.zeros((n_obs + (n_num if use_beta else 0), width))
    target = np.zeros(design.shape[0])
    root_w = np.sqrt(w)
    for row in range(n_obs):
        design[row, position[draw_idx[row]]] = 1.0
        if rank_idx[row] > 0:
            design[row, n_draw + rank_idx[row] - 1] = 1.0
        if use_beta:
            design[row, n_draw + n_ranks - 1 + drawn[draw_idx[row]] - 1] = g[row]
    design[:n_obs] *= root_w[:, None]
    target[:n_obs] = y * root_w
    if use_beta:
        # Scale the penalty to the typical observation weight so RIDGE reads as
        # "this many observations' worth of pull toward equal popularity".
        penalty = np.sqrt(ridge * np.median(w))
        for i in range(n_num):
            design[n_obs + i, n_draw + n_ranks - 1 + i] = penalty
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    offsets = np.concatenate([[0.0], solution[n_draw : n_draw + n_ranks - 1]])
    beta = solution[n_draw + n_ranks - 1 :] if use_beta else np.zeros(n_num)
    return beta - beta.mean(), offsets


def _within_draw_sse(drawn, draw_idx, rank_idx, y, g, w, beta, offsets) -> float:
    """Weighted squared error after giving each draw its own best intercept."""
    predicted = offsets[rank_idx] + g * beta[drawn[draw_idx] - 1].sum(axis=1)
    residual = y - predicted
    total = 0.0
    for d in np.unique(draw_idx):
        mask = draw_idx == d
        if mask.sum() < 2:
            continue
        centre = np.average(residual[mask], weights=w[mask])
        total += float((w[mask] * (residual[mask] - centre) ** 2).sum())
    return total


def _log_weights(spec: GameSpec, frame: pd.DataFrame, ridge: float) -> np.ndarray:
    """Stage 1: log popularity of each number, from winner-count ratios."""
    drawn, draw_idx, rank_idx, y, g, w, n_ranks = _observations(spec, frame)
    beta, _ = _solve(spec, drawn, draw_idx, rank_idx, y, g, w, n_ranks, ridge=ridge)
    return beta


def _drawn_popularity(spec: GameSpec, frame: pd.DataFrame, beta) -> np.ndarray:
    """Per draw: summed log popularity of the numbers that came up."""
    drawn = frame[main_columns(spec)].to_numpy(dtype=int)
    return beta[drawn - 1].sum(axis=1)


def _rank_hits(spec: GameSpec, rank: int) -> int:
    return min(m for m, _ in spec.rank_tiers[rank - 1])


def _payout_lines(spec: GameSpec, frame: pd.DataFrame, beta) -> dict:
    """Stage 2: per shared rank, regress log payout on drawn-number popularity.
    Returns ``{rank: (intercept, elasticity, reliable)}``."""
    popularity = _drawn_popularity(spec, frame, beta)
    lines = {}
    for rank in sorted(spec.pari_ranks - spec.static_ranks):
        payout = frame[f"r{rank}_payout"].to_numpy(dtype=float)
        ok = np.isfinite(payout) & (payout > 0)
        x, n = popularity[ok], int(ok.sum())
        if n < 30 or np.ptp(x) == 0:
            continue
        y = np.log(payout[ok])
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (intercept + slope * x)
        stderr = np.sqrt(residual.var(ddof=2) / ((x - x.mean()) ** 2).sum())
        elasticity = -slope
        reliable = elasticity > 0 and stderr < 0.35 * elasticity
        lines[rank] = (float(intercept), float(elasticity), bool(reliable))
    return lines


def _elasticities(spec: GameSpec, lines: dict) -> tuple[tuple[int, float], ...]:
    """Measured elasticity for well-determined ranks; for the rest (few winners,
    noisy payouts) the count model's coefficient scaled by how much steeper the
    well-determined ranks turned out to be."""
    ratios = [
        elasticity / hit_coefficient(spec, _rank_hits(spec, rank))
        for rank, (_, elasticity, reliable) in lines.items()
        if reliable and hit_coefficient(spec, _rank_hits(spec, rank)) > 0
    ]
    scale = float(np.median(ratios)) if ratios else 1.0
    out = []
    for rank in sorted(spec.pari_ranks - spec.static_ranks):
        line = lines.get(rank)
        if line and line[2]:
            out.append((rank, line[1]))
        else:
            theory = max(hit_coefficient(spec, _rank_hits(spec, rank)), 0.0)
            out.append((rank, theory * scale))
    return tuple(out)


def _current(spec: GameSpec, frame: pd.DataFrame) -> pd.DataFrame:
    """Rows paid under the present prize structure."""
    if spec.payout_since is None:
        return frame
    return frame[frame["date"] >= spec.payout_since]


def _payout_skill(spec: GameSpec, frame: pd.DataFrame, ridge: float) -> float | None:
    """Out-of-sample skill at predicting payouts.

    Everything is fitted on draws before a cut-off, then asked to predict the log
    payout of each shared rank in the draws after it. The score is R^2 against
    'every draw pays the earlier average', averaged over ranks in proportion to
    how much expected value each contributes — so it measures what matters to a
    ticket's worth, not the noisy top ranks.
    """
    current = _current(spec, frame)
    if len(current) < 50:
        return None
    cut = current["date"].iloc[int(len(current) * (1 - HOLDOUT))]
    earlier, later = frame[frame["date"] < cut], current[current["date"] >= cut]
    if len(earlier) < MIN_DRAWS or len(later) < 10:
        return None
    beta = _log_weights(spec, earlier, ridge)
    known = _current(spec, earlier)
    lines = _payout_lines(spec, known, beta)
    popularity = _drawn_popularity(spec, later, beta)
    total = weight_sum = 0.0
    for rank, (intercept, elasticity, _) in lines.items():
        payout = later[f"r{rank}_payout"].to_numpy(dtype=float)
        ok = np.isfinite(payout) & (payout > 0)
        past = known[f"r{rank}_payout"].to_numpy(dtype=float)
        past = past[np.isfinite(past) & (past > 0)]
        if ok.sum() < 5 or len(past) < 5:
            continue
        actual = np.log(payout[ok])
        sst = float(((actual - np.log(past).mean()) ** 2).sum())
        if sst <= 0:
            continue
        sse = float(((actual - (intercept - elasticity * popularity[ok])) ** 2).sum())
        weight = rank_probability(spec, rank) * past.mean()
        total += weight * (1.0 - sse / sst)
        weight_sum += weight
    return total / weight_sum if weight_sum else None


def fit(
    spec: GameSpec, frame: pd.DataFrame, *, ridge: float = RIDGE
) -> PopularityModel:
    """Fit number popularity to a payout frame.

    Pass the long frame (``load_payouts(game, current_rules_only=False)``): winner
    counts from every comparable draw fix the number weights, while payout
    elasticities use only draws under the present prize structure.
    """
    if len(comparable_ranks(spec)) < 2:
        raise ValueError(f"{spec.name} has too few comparable ranks to fit.")
    frame = frame.sort_values("date").reset_index(drop=True)
    n_draws = len(np.unique(_observations(spec, frame)[1]))
    if n_draws < MIN_DRAWS:
        raise ValueError(f"only {n_draws} usable draws; need {MIN_DRAWS} to fit.")

    weights = _normalised(np.exp(_log_weights(spec, frame, ridge)))
    lines = _payout_lines(spec, _current(spec, frame), np.log(weights))
    return PopularityModel(
        game=spec.key,
        weights=weights,
        source="fitted",
        n_draws=n_draws,
        skill=_payout_skill(spec, frame, ridge),
        elasticity=_elasticities(spec, lines),
    )


@lru_cache(maxsize=None)
def _load_model(game: str, _snapshot: str | None) -> PopularityModel:
    spec = get(game)
    frame = try_load(game, current_rules_only=False)
    if frame is not None:
        try:
            return fit(spec, frame)
        except ValueError:
            pass
    return heuristic_model(spec)


def load_model(game: str) -> PopularityModel:
    """The best available popularity model for a game: fitted to its winner counts
    when a snapshot has them, the birthday heuristic otherwise."""
    return _load_model(game, newest_snapshot(game))


# --- Using the weights -----------------------------------------------------------


def _mean_product(values, size: int) -> float:
    """Mean over all ``size``-subsets of ``values`` of the product of the subset
    (a normalised elementary symmetric polynomial). Exact, O(len * size)."""
    values = list(values)
    if size == 0:
        return 1.0
    if size > len(values):
        raise ValueError("subset larger than the pool")
    poly = [0.0] * (size + 1)
    poly[0] = 1.0
    for value in values:
        for j in range(size, 0, -1):
            poly[j] += value * poly[j - 1]
    return poly[size] / comb(len(values), size)


def combo_popularity(model: PopularityModel, main, spec: GameSpec) -> float:
    """How often other players pick this exact main combination, relative to a
    uniformly random pick (1.0 = average; below 1 = shared less).

    Normalised by the exact mean product over all combinations, because the
    weights on one ticket are drawn without replacement and so are not independent.
    """
    check_pool(main, spec.main_count, 1, spec.main_max, f"{spec.key} main")
    product = 1.0
    for number in main:
        product *= model.weight(number)
    return product / _mean_product(model.weights, spec.main_count)


def share_multiplier(
    model: PopularityModel,
    main,
    spec: GameSpec,
    hits: int,
    rank: int | None = None,
) -> float:
    """Expected payout of a shared-pool rank for THIS ticket, relative to the
    average winner of that rank, given the ticket matches ``hits`` mains.

    When the ticket wins with ``hits`` matches, the draw contains ``hits`` of its
    numbers and ``k - hits`` others. The payout scales as ``exp(-e * popularity of
    the drawn numbers)``, where ``e`` is the rank's measured payout elasticity
    (or the count model's ``g_h`` when ``rank`` is not given or not measured);
    this averages that over every draw consistent with the win, exactly.
    Averaged over all possible tickets it is 1: the model only moves money between
    tickets, it never creates any.
    """
    check_pool(main, spec.main_count, 1, spec.main_max, f"{spec.key} main")
    k = spec.main_count
    if not 0 <= hits <= k:
        raise ValueError(f"hits must be in [0, {k}]")
    g = model.elasticity_for(rank)
    if g is None:
        g = hit_coefficient(spec, hits)
    scaled = [w**-g for w in model.weights]
    mine = set(main)
    inside = [scaled[n - 1] for n in mine]
    outside = [scaled[n - 1] for n in range(1, spec.main_max + 1) if n not in mine]
    conditional = _mean_product(inside, hits) * _mean_product(outside, k - hits)
    return conditional / _mean_product(scaled, k)


def ranked_numbers(model: PopularityModel) -> list[tuple[int, float]]:
    """``(number, weight)`` from least to most popular."""
    pairs = [(n, model.weight(n)) for n in range(1, len(model.weights) + 1)]
    return sorted(pairs, key=lambda pair: pair[1])
