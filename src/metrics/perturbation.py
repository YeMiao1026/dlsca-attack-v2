"""Perturbation-size metrics, for comparing defended vs. clean traces.
Interface reserved for dlsca-defense-v2 (see CLAUDE.md §1.3 non-goals);
inputs here are plain trace arrays so this module has no dependency on the
defense project's internals.
"""

from __future__ import annotations

import numpy as np


def psr(clean: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    """Perturbation-to-Signal Ratio per trace."""
    raise NotImplementedError


def l2(clean: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def linf(clean: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    raise NotImplementedError
