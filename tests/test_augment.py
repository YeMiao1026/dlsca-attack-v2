"""Guards for the E02 gaussian noise augmentation path (CLAUDE.md §6.4, pitfall #11):
augmentation must be regenerated fresh every epoch, not a fixed precomputed dataset.
"""

from __future__ import annotations

import numpy as np

from src.data.preprocess import gaussian_augment
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
