"""Model -> probability matrix. See CLAUDE.md §4.2, §5.2 stage 4 — this is the
only interface between the attack and evaluation stages; probs.npy must be
re-derivable from a saved model + traces alone.
"""

from __future__ import annotations

import keras
import numpy as np


def run(model: keras.Model, traces: np.ndarray) -> np.ndarray:
    """Return probs (N, n_classes) float32; each row must sum to 1, no NaNs."""
    x = traces.astype(np.float32)
    if x.ndim == 2:
        x = x[..., None]  # (N, trace_len) -> (N, trace_len, 1) for Conv1D input
    probs = model.predict(x, verbose=0)
    return probs.astype(np.float32)
