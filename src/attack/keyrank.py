"""Multi-run key recovery + derived metrics. See CLAUDE.md §5.2 stage 5,
§5.5.2-3 — a single attack run's rank curve is sampling noise, not a
conclusion (pitfall #6); everything here is built on repeated independent runs.
"""

from __future__ import annotations

import numpy as np


def evaluate(scores: np.ndarray, correct_key: int, n_runs: int = 100,
             max_traces: int = 1000, seed: int = 0) -> np.ndarray:
    """Each run independently reshuffles the evaluation set, takes the first
    `max_traces`, cumsums scores, and ranks the correct key among all 256
    hypotheses at every trace count. Returns ranks (n_runs, max_traces) int16.
    """
    n_samples = scores.shape[0]
    if max_traces > n_samples:
        raise ValueError(f"max_traces ({max_traces}) exceeds available samples ({n_samples})")
    ranks = np.empty((n_runs, max_traces), dtype=np.int16)
    for run in range(n_runs):
        # cfg.attack.seed + run_index, see CLAUDE.md §8.1
        rng = np.random.default_rng(seed + run)
        idx = rng.permutation(n_samples)[:max_traces]
        cum = np.cumsum(scores[idx], axis=0)  # (max_traces, 256)
        correct_cum = cum[:, correct_key][:, None]
        ranks[run] = (cum > correct_cum).sum(axis=1).astype(np.int16)
    return ranks


def ge(ranks: np.ndarray) -> np.ndarray:
    """Guessing Entropy: mean rank over runs, per trace count. Shape (max_traces,)."""
    return ranks.mean(axis=0)


def sr1(ranks: np.ndarray) -> np.ndarray:
    """Success rate (rank == 0 fraction) per trace count. Shape (max_traces,)."""
    return (ranks == 0).mean(axis=0)


def _first_sustained(condition: np.ndarray) -> int | None:
    """1-indexed trace count N such that `condition` holds for every index >= N-1."""
    if condition.all():
        return 1
    last_false = int(np.nonzero(~condition)[0][-1])
    candidate = last_false + 1
    if candidate >= len(condition):
        return None
    return candidate + 1


def n_tge(ge_curve: np.ndarray, threshold: float = 1.0) -> int | None:
    """Smallest N such that GE stays below `threshold` for ALL N' >= N — not just
    the first dip below it (pitfall #7: noise can touch the threshold early and
    bounce back). Returns None if never sustained.
    """
    return _first_sustained(ge_curve < threshold)


def n_sr90(sr1_curve: np.ndarray, threshold: float = 0.9) -> int | None:
    """Smallest N such that SR1 stays >= `threshold` for all N' >= N."""
    return _first_sustained(sr1_curve >= threshold)


def percentiles(ranks: np.ndarray, q: tuple[float, ...] = (25, 50, 75)) -> np.ndarray:
    """Rank distribution percentiles per trace count, for shaded-band plots."""
    return np.percentile(ranks, q, axis=0)
