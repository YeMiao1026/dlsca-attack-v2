"""Unified-style figure generation for runs/{exp_id}_{timestamp}/figures/."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

FIGSIZE = (8, 5)
DPI = 150


def _save(fig: plt.Figure, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_ge_curve(ge_curve: np.ndarray, percentile_band: tuple[np.ndarray, np.ndarray] | None,
                   out_path: str | Path) -> None:
    """GE vs. trace count; shade the 25/75 percentile band when provided (CLAUDE.md §5.5.3)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(1, len(ge_curve) + 1)
    if percentile_band is not None:
        p25, p75 = percentile_band
        ax.fill_between(x, p25, p75, alpha=0.2, label="25th-75th percentile")
    ax.plot(x, ge_curve, label="GE (mean rank)")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="N_TGE threshold (GE=1)")
    ax.set_xlabel("Number of traces")
    ax.set_ylabel("Guessing Entropy")
    ax.set_title("Guessing Entropy vs. trace count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)


def plot_sr_curve(sr1_curve: np.ndarray, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(1, len(sr1_curve) + 1)
    ax.plot(x, sr1_curve, label="SR1 (success rate)")
    ax.axhline(0.9, color="red", linestyle="--", linewidth=1, label="N_SR90 threshold")
    ax.set_xlabel("Number of traces")
    ax.set_ylabel("Success rate (rank = 0)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Success Rate vs. trace count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)


def plot_snr(snr_curve: np.ndarray, out_path: str | Path, overlay: np.ndarray | None = None) -> None:
    """SNR-vs-time, optionally overlaid with the unmasked-label SNR for stage-0 sanity checks."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(snr_curve))
    ax.plot(x, snr_curve, label="masked-label SNR")
    if overlay is not None:
        ax.plot(x, overlay, label="unmasked-label SNR (control)", alpha=0.7)
    ax.set_xlabel("Time point")
    ax.set_ylabel("SNR")
    ax.set_title("SNR vs. time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)
