"""Four-way index split. See CLAUDE.md §4.2, §5.2 stage 1 — A/V/D are a single
permutation of the profiling set and must be pairwise disjoint; E is drawn
independently from the attack set and never mixed with profiling.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class SplitIndices(NamedTuple):
    a: np.ndarray  # attacker model training set
    v: np.ndarray  # model-selection (GE-based) set
    d: np.ndarray  # reserved for defender training (unused this stage, must still be cut)
    e: np.ndarray  # final evaluation set, drawn from the independent attack set


def four_way(n_profiling: int, n_attack: int, n_attacker: int, n_val: int, n_defender: int, seed: int) -> SplitIndices:
    """n_attacker + n_val + n_defender must be <= n_profiling. E = arange(n_attack)."""
    total = n_attacker + n_val + n_defender
    if total > n_profiling:
        raise ValueError(
            f"n_attacker + n_val + n_defender ({total}) exceeds n_profiling ({n_profiling})"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_profiling)
    a = np.sort(perm[:n_attacker])
    v = np.sort(perm[n_attacker:n_attacker + n_val])
    d = np.sort(perm[n_attacker + n_val:total])
    e = np.arange(n_attack)
    return SplitIndices(a=a, v=v, d=d, e=e)


def save(indices: SplitIndices, path: str) -> None:
    """Write to `{run_dir}/split_indices.npz`."""
    np.savez(path, a=indices.a, v=indices.v, d=indices.d, e=indices.e)


def load(path: str) -> SplitIndices:
    with np.load(path) as data:
        return SplitIndices(a=data["a"], v=data["v"], d=data["d"], e=data["e"])
