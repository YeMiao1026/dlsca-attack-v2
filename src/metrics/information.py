"""Information-theoretic metrics. See CLAUDE.md §5.5.3 — PI is N-independent,
unlike GE/SR, so it is useful for comparing leakage across configs at a glance.
"""

from __future__ import annotations

import numpy as np


def pi(probs: np.ndarray, labels: np.ndarray, n_classes: int, eps: float = 1e-40) -> float:
    """Perceived Information: H[Z] + mean(log2 p(z|t)) over the evaluation set.

    H[Z] is estimated empirically from `labels` rather than assumed to be
    log2(n_classes) — exactly uniform for ID/ID_MASKED given random plaintexts,
    but not for HW (Hamming weight follows a binomial, not a uniform, over
    0..8), so the empirical estimate is correct for either leakage model.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    class_probs = counts / counts.sum()
    nonzero = class_probs > 0
    h_z = -np.sum(class_probs[nonzero] * np.log2(class_probs[nonzero]))

    true_class_probs = probs[np.arange(len(labels)), labels]
    mean_log2_p = np.mean(np.log2(true_class_probs + eps))
    return float(h_z + mean_log2_p)


def mi(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Mutual Information estimate between the true label and predicted distribution."""
    raise NotImplementedError
