"""Guards for src/metrics/information.py::mi, mirroring the two-extreme-case
philosophy used for keyrank (perfect prediction / uniform prediction)."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.information import mi


def test_mi_is_near_max_entropy_for_confident_correct_predictions():
    n, n_classes = 500, 8
    rng = np.random.default_rng(0)
    labels = rng.integers(0, n_classes, size=n)
    probs = np.full((n, n_classes), 1e-6)
    probs[np.arange(n), labels] = 1.0 - 1e-6 * (n_classes - 1)
    value = mi(probs, labels, n_classes)
    assert value == pytest.approx(np.log2(n_classes), abs=0.05)


def test_mi_is_near_zero_for_uniform_predictions():
    # H(Z|T) is exactly log2(n_classes) here (probs are exactly uniform), but
    # H(Z) is only an empirical estimate from a finite label sample, so it
    # carries a little sampling noise even when the true class distribution
    # is uniform — hence a small tolerance rather than an exact-zero check.
    n, n_classes = 500, 8
    rng = np.random.default_rng(0)
    labels = rng.integers(0, n_classes, size=n)
    probs = np.full((n, n_classes), 1.0 / n_classes)
    value = mi(probs, labels, n_classes)
    assert abs(value) < 0.05


def test_mi_flags_confident_but_wrong_predictions_as_nonzero():
    # documents the known caveat in mi()'s docstring: a confidently WRONG
    # model still reads as high MI, since MI here doesn't check correctness.
    n, n_classes = 500, 8
    rng = np.random.default_rng(0)
    labels = rng.integers(0, n_classes, size=n)
    wrong_class = (labels + 1) % n_classes
    probs = np.full((n, n_classes), 1e-6)
    probs[np.arange(n), wrong_class] = 1.0 - 1e-6 * (n_classes - 1)
    value = mi(probs, labels, n_classes)
    assert value == pytest.approx(np.log2(n_classes), abs=0.05)
