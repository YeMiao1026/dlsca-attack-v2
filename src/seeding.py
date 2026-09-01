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


def set_global_seed(seed: int, deterministic_ops: bool = True) -> None:
    """Seed python `random`, `numpy`, and `tensorflow` together. Call once at startup.

    Seeding the RNGs is NOT sufficient for reproducibility on GPU. cuDNN/cuBLAS
    pick algorithms at runtime and reduce with atomics, so the same config with
    the same seed drifts run to run: two E01 runs on this server diverged at
    epoch 1 (loss 5.567685 vs 5.567882) and ended with visibly different GE
    previews, violating CLAUDE.md §11's requirement that two executions produce
    identical metrics.json. `enable_op_determinism` pins those choices. It must
    run before any op is created, and it does cost some speed — hence the
    `deterministic_ops` switch (config: `deterministic: false`) for anyone who
    would rather have the throughput. See CLAUDE.md 附錄 B.63.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()


def derive_seed(base_seed: int, *parts: int) -> int:
    """Deterministically derive a child seed from a base seed and integer parts."""
    seed = base_seed
    for part in parts:
        seed = (seed * 1_000_003 + part) % (2**31 - 1)
    return seed
