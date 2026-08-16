"""Global and derived seed control. See CLAUDE.md §8.1 — layered seeds:

    global (numpy/random/tensorflow) <- cfg.seed
    split                            <- cfg.split.seed
    augmentation (per epoch)         <- cfg.seed + epoch
    attack run reshuffle (per run)   <- cfg.attack.seed + run_index
"""

from __future__ import annotations

import random

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int) -> None:
    """Seed python `random`, `numpy`, and `tensorflow` together. Call once at startup."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def derive_seed(base_seed: int, *parts: int) -> int:
    """Deterministically derive a child seed from a base seed and integer parts."""
    seed = base_seed
    for part in parts:
        seed = (seed * 1_000_003 + part) % (2**31 - 1)
    return seed
