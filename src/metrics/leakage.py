"""Leakage assessment metrics, independent of any trained model. Used by
scripts/00_inspect_data.py (see CLAUDE.md §5.2 stage 0) and metrics.leakage_assessment.
"""

from __future__ import annotations

import numpy as np


def snr(traces: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Signal-to-noise ratio per time point: Var(mean-per-class) / mean(Var-per-class)."""
    traces = np.asarray(traces, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    n_points = traces.shape[1]
    class_means = np.empty((len(classes), n_points))
    class_vars = np.empty((len(classes), n_points))
    for i, c in enumerate(classes):
        class_traces = traces[labels == c]
        class_means[i] = class_traces.mean(axis=0)
        class_vars[i] = class_traces.var(axis=0)
    signal = class_means.var(axis=0)
    noise = class_vars.mean(axis=0)
    return np.divide(signal, noise, out=np.zeros_like(signal), where=noise > 0)


def nicv(traces: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Normalized Inter-Class Variance per time point."""
    raise NotImplementedError


def t_test(traces_a: np.ndarray, traces_b: np.ndarray) -> np.ndarray:
    """Welch's t-test statistic per time point (fixed-vs-random style TVLA)."""
    raise NotImplementedError
