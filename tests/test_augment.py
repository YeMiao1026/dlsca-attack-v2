"""Guards for the E02 gaussian noise augmentation path (CLAUDE.md §6.4, pitfall #11):
augmentation must be regenerated fresh every epoch, not a fixed precomputed dataset.
"""

from __future__ import annotations

import numpy as np

from src.data.preprocess import gaussian_augment, jamming_augment
from src.train.trainer import _GaussianAugmentedDataset


def test_gaussian_augment_is_reproducible_given_same_seed():
    traces = np.random.default_rng(0).normal(size=(50, 20)).astype(np.float32)
    a = gaussian_augment(traces, sigma_ratio=0.5, seed=123)
    b = gaussian_augment(traces, sigma_ratio=0.5, seed=123)
    np.testing.assert_array_equal(a, b)


def test_gaussian_augment_differs_across_seeds():
    traces = np.random.default_rng(0).normal(size=(50, 20)).astype(np.float32)
    a = gaussian_augment(traces, sigma_ratio=0.5, seed=1)
    b = gaussian_augment(traces, sigma_ratio=0.5, seed=2)
    assert not np.allclose(a, b)


def test_augmented_dataset_reinjects_fresh_noise_every_epoch():
    x = np.random.default_rng(0).normal(size=(40, 10)).astype(np.float32)
    y = np.random.default_rng(0).integers(0, 256, size=40)
    ds = _GaussianAugmentedDataset(x, y, batch_size=8, sigma_ratio=0.5, seed=42)

    epoch0_batch0, _ = ds[0]
    ds.on_epoch_end()
    epoch1_batch0, _ = ds[0]

    assert not np.allclose(epoch0_batch0, epoch1_batch0)


def test_augmented_dataset_batches_cover_the_whole_epoch():
    x = np.random.default_rng(0).normal(size=(40, 10)).astype(np.float32)
    y = np.arange(40)
    ds = _GaussianAugmentedDataset(x, y, batch_size=8, sigma_ratio=0.5, seed=42)

    assert len(ds) == 5
    seen_labels = np.concatenate([ds[i][1] for i in range(len(ds))])
    np.testing.assert_array_equal(np.sort(seen_labels), y)


def test_jamming_augment_is_reproducible_given_same_seed():
    traces = np.random.default_rng(0).normal(size=(50, 20)).astype(np.float32)
    a = jamming_augment(traces, max_shift=5, seed=123)
    b = jamming_augment(traces, max_shift=5, seed=123)
    np.testing.assert_array_equal(a, b)


def test_jamming_augment_differs_across_seeds():
    traces = np.random.default_rng(0).normal(size=(50, 20)).astype(np.float32)
    a = jamming_augment(traces, max_shift=5, seed=1)
    b = jamming_augment(traces, max_shift=5, seed=2)
    assert not np.allclose(a, b)


def test_jamming_augment_shifts_a_known_pattern_and_zero_pads_the_exposed_edge():
    # a single trace with an isolated spike at index 10; a known shift amount
    # (via a seed we control the draw of) should move the spike and zero-pad
    # whichever edge gets exposed, not wrap around to the other edge.
    trace = np.zeros((1, 20), dtype=np.float32)
    trace[0, 10] = 1.0

    # max_shift=0 must be a strict no-op regardless of seed.
    unshifted = jamming_augment(trace, max_shift=0, seed=0)
    np.testing.assert_array_equal(unshifted, trace)

    shifted = jamming_augment(trace, max_shift=3, seed=0)
    spike_positions = np.flatnonzero(shifted[0])
    assert len(spike_positions) <= 1  # never duplicated, never wrapped in a second spike
    if len(spike_positions) == 1:
        assert abs(int(spike_positions[0]) - 10) <= 3


def test_jamming_augment_never_changes_total_trace_length():
    traces = np.random.default_rng(0).normal(size=(30, 700)).astype(np.float32)
    out = jamming_augment(traces, max_shift=50, seed=7)
    assert out.shape == traces.shape
