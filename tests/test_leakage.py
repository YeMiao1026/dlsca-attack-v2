"""Guards for src/metrics/leakage.py's nicv/t_test (snr already has coverage
implicitly via 00_inspect_data.py's real-data pass/fail check)."""

from __future__ import annotations

import numpy as np

from src.metrics.leakage import nicv, t_test


def test_nicv_is_near_zero_when_traces_are_pure_noise():
    rng = np.random.default_rng(0)
    traces = rng.normal(size=(2000, 20))
    labels = rng.integers(0, 4, size=2000)
    values = nicv(traces, labels)
    assert np.all(np.abs(values) < 0.05)


def test_nicv_is_high_when_class_mean_dominates_the_variance():
    rng = np.random.default_rng(0)
    n_points = 20
    labels = rng.integers(0, 4, size=4000)
    class_offset = labels[:, None] * 10.0
    traces = class_offset + rng.normal(scale=0.1, size=(4000, n_points))
    values = nicv(traces, labels)
    assert np.all(values > 0.95)


def test_nicv_bounded_in_zero_one_range():
    rng = np.random.default_rng(1)
    traces = rng.normal(size=(500, 10))
    labels = rng.integers(0, 3, size=500)
    values = nicv(traces, labels)
    assert np.all(values >= -1e-9)
    assert np.all(values <= 1.0 + 1e-9)


def test_t_test_near_zero_when_groups_are_identically_distributed():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3000, 15))
    b = rng.normal(size=(3000, 15))
    values = t_test(a, b)
    assert np.all(np.abs(values) < 4.5)  # below the conventional TVLA threshold


def test_t_test_exceeds_tvla_threshold_when_groups_differ():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, size=(3000, 15))
    b = rng.normal(loc=1.0, size=(3000, 15))
    values = t_test(a, b)
    assert np.all(np.abs(values) > 4.5)


def test_t_test_sign_reflects_direction_of_difference():
    a = np.full((100, 1), 5.0) + np.random.default_rng(0).normal(scale=0.01, size=(100, 1))
    b = np.full((100, 1), 1.0) + np.random.default_rng(1).normal(scale=0.01, size=(100, 1))
    assert t_test(a, b)[0] > 0
    assert t_test(b, a)[0] < 0
