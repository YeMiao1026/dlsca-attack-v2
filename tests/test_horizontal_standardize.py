"""Guardrails for src/data/preprocess.py::horizontal_standardize (Wouters et al.)."""

import numpy as np

from src.data.preprocess import horizontal_standardize


def test_every_trace_has_zero_mean_and_unit_std_over_time():
    x = np.random.default_rng(0).normal(3.0, 7.0, size=(20, 50)).astype(np.float32)
    out = horizontal_standardize(x)
    assert np.allclose(out.mean(axis=1), 0.0, atol=1e-5)
    assert np.allclose(out.std(axis=1), 1.0, atol=1e-5)


def test_each_trace_is_normalised_independently_nothing_is_fit():
    """Per-trace statistics: a row's output must not depend on the other rows."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(10, 30)).astype(np.float32)
    alone = horizontal_standardize(x[3:4])
    together = horizontal_standardize(np.vstack([x, 1000.0 * x]))[3:4]
    assert np.allclose(alone, together)


def test_constant_trace_does_not_divide_by_zero():
    out = horizontal_standardize(np.full((2, 8), 5.0, dtype=np.float32))
    assert np.isfinite(out).all() and np.allclose(out, 0.0)
