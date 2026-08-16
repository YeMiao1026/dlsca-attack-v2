"""Guards for blind cross-correlation realignment (CLAUDE.md 附錄 B.30)."""

from __future__ import annotations

import numpy as np

from src.data.resync import resync, resync_iterative


def _synthetic_desynced_traces(n=200, length=200, true_shift_range=10, seed=0):
    """`true_shift_range` bounds each trace's absolute jitter; pairwise
    relative shifts (what `resync` actually has to search for, since it
    aligns to one trace used as reference) can be up to 2x that.
    """
    rng = np.random.default_rng(seed)
    poi = length // 2
    template = np.zeros(length)
    template[poi - 3 : poi + 3] = 10.0  # a single sharp feature to recover
    shifts = rng.integers(-true_shift_range, true_shift_range + 1, size=n)
    traces = np.empty((n, length))
    for i, s in enumerate(shifts):
        traces[i] = np.roll(template, s) + rng.normal(0, 0.5, size=length)
    return traces, shifts


def test_resync_recovers_known_shift():
    traces, true_shifts = _synthetic_desynced_traces(true_shift_range=10)
    reference = traces[0]
    # search window must cover the worst-case relative shift (2x true_shift_range)
    _, estimated_shifts = resync(traces, reference, max_shift=20)
    relative_true = true_shifts - true_shifts[0]
    np.testing.assert_array_equal(estimated_shifts, relative_true)


def test_resync_aligns_the_feature_to_a_single_column():
    traces, _ = _synthetic_desynced_traces(true_shift_range=10)
    aligned, _, _ = resync_iterative(traces, max_shift=20, rounds=2)
    # before alignment, averaging spreads the spike's energy across many
    # columns and flattens the mean trace's peak; after alignment the spike
    # reinforces at a single column, so the mean trace's peak amplitude
    # should recover close to the template's true amplitude (10.0)
    peak_before = traces.mean(axis=0).max()
    peak_after = aligned.mean(axis=0).max()
    assert peak_after > peak_before
    assert peak_after > 8.0  # close to the template's true amplitude


def test_resync_iterative_returns_reference_usable_for_other_sets():
    traces, _ = _synthetic_desynced_traces(seed=0)
    other, _ = _synthetic_desynced_traces(seed=1)
    _, _, reference = resync_iterative(traces, max_shift=20, rounds=2)
    aligned_other, shifts_other = resync(other, reference, max_shift=20)
    assert aligned_other.shape == other.shape
    assert shifts_other.shape == (len(other),)
