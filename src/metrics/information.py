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
    """Mutual Information estimate: H[Z] - E_t[H[Z|T=t]], using the model's own
    predicted per-trace distribution as the empirical conditional p(z|t)
    (a standard "predictive entropy" MI estimator).

    Caveat: unlike PI, this does not check whether high-probability mass sits
    on the *correct* class — a confidently-wrong-but-calibrated model reads
    as high MI here despite leaking nothing useful. PI is the metric to trust
    for "is the model actually right"; this MI is a cheaper, label-light
    estimate of how peaked/decisive the model's outputs are.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    class_probs = counts / counts.sum()
    nonzero = class_probs > 0
    h_z = -np.sum(class_probs[nonzero] * np.log2(class_probs[nonzero]))

    p = np.asarray(probs, dtype=np.float64)
    nonzero_p = p > 0
    terms = np.zeros_like(p)
    terms[nonzero_p] = p[nonzero_p] * np.log2(p[nonzero_p])
    h_z_given_t = -terms.sum(axis=1).mean()

    return float(h_z - h_z_given_t)
