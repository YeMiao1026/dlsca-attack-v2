"""Per-point standardization and augmentation. See CLAUDE.md §4.2, §6.4."""

from __future__ import annotations

import numpy as np


class Standardizer:
    """Per-timepoint z-score. Must be fit on A only (pitfall #5 — fitting on
    all data leaks profiling/attack-set statistics and inflates results).
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, traces: np.ndarray) -> "Standardizer":
        traces = np.asarray(traces, dtype=np.float64)
        self.mean_ = traces.mean(axis=0)
        self.std_ = traces.std(axis=0)
        return self

    def transform(self, traces: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must be fit before transform")
        traces = np.asarray(traces, dtype=np.float32)
        # a timepoint with zero variance (e.g. constant padding) would divide by
        # zero; substituting std=1 there just leaves it at its (constant) value
        # minus the mean, instead of producing NaN/Inf.
        std = np.where(self.std_ == 0, 1.0, self.std_)
        return ((traces - self.mean_) / std).astype(np.float32)

    def fit_transform(self, traces: np.ndarray) -> np.ndarray:
        return self.fit(traces).transform(traces)


class MinMaxScaler:
    """Per-timepoint min-max scaling to `feature_range`, applied after Standardizer.

    The Zaid et al. efficient-CNN-for-SCA reference pipeline chains
    StandardScaler -> MinMaxScaler([0,1]) before training; CLAUDE.md §5.2's
    preprocessing spec only mentions the first step. Must be fit on A only,
    same reasoning as Standardizer (pitfall #5).
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        self.feature_range = feature_range
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, traces: np.ndarray) -> "MinMaxScaler":
        traces = np.asarray(traces, dtype=np.float64)
        self.min_ = traces.min(axis=0)
        self.max_ = traces.max(axis=0)
        return self

    def transform(self, traces: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("MinMaxScaler must be fit before transform")
        traces = np.asarray(traces, dtype=np.float32)
        span = np.where(self.max_ == self.min_, 1.0, self.max_ - self.min_)
        lo, hi = self.feature_range
        unit = (traces - self.min_) / span
        return (unit * (hi - lo) + lo).astype(np.float32)

    def fit_transform(self, traces: np.ndarray) -> np.ndarray:
        return self.fit(traces).transform(traces)


def gaussian_augment(traces: np.ndarray, sigma_ratio: float, seed: int) -> np.ndarray:
    """Add N(0, (sigma_ratio * per-point std)^2) noise. Must be regenerated fresh
    every epoch by the caller (pitfall #11) — this function is stateless per call.
    """
    traces = np.asarray(traces, dtype=np.float32)
    rng = np.random.default_rng(seed)
    per_point_std = traces.std(axis=0)
    noise = rng.normal(0.0, 1.0, size=traces.shape).astype(np.float32) * (sigma_ratio * per_point_std)
    return traces + noise
