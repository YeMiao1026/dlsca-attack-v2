"""Log-likelihood score matrix. See CLAUDE.md §5.5.1 — scores MUST be summed
in log space, never multiplied as raw probabilities (pitfall #1: probability
products underflow to exactly 0 past ~50 traces and GE gets stuck).
"""

from __future__ import annotations

import numpy as np

from src.data.ascad import AES_SBOX
from src.data.labels import HW_TABLE, LeakageModel


def build(probs: np.ndarray, plaintexts: np.ndarray, target_byte: int, eps: float = 1e-40,
          leakage_model: LeakageModel = "ID", mask: np.ndarray | None = None) -> np.ndarray:
    """Return scores (N, C) float64: score[i,k] = log(probs[i, class(i,k)] + eps),
    where `class(i,k)` is whatever class index the model was actually trained
    to predict for key hypothesis k on trace i — this must match
    src/data/labels.py::build's leakage_model exactly, or the hypothesis test
    silently checks the wrong thing:

    - ID:        class(i,k) = Sbox[p[i,byte]^k]                       (0..255)
    - ID_MASKED: class(i,k) = Sbox[p[i,byte]^k] ^ mask[i]              (0..255)
                 `mask` (shape (N,), one already-selected masks[:, mask_index]
                 column) is required. Omitting it silently scores against the
                 unmasked hypothesis and never converges even when the model
                 has clearly learned Z' — this bit an early E08 run (loss
                 dropped fine, GE never moved).
    - HW:        class(i,k) = HW_TABLE[Sbox[p[i,byte]^k]]              (0..8)
    """
    if mask is not None and leakage_model != "ID_MASKED":
        # otherwise mask is silently ignored — the same "scored against the
        # wrong hypothesis" footgun the ID_MASKED branch below exists to fix.
        raise ValueError(f"mask was given but leakage_model={leakage_model!r} (expected 'ID_MASKED')")

    n = probs.shape[0]
    plaintext_byte = plaintexts[:, target_byte].astype(np.uint8)
    key_hypotheses = np.arange(256, dtype=np.uint8)
    hyp = AES_SBOX[plaintext_byte[:, None] ^ key_hypotheses[None, :]]  # (N, 256), values 0..255

    if leakage_model == "ID":
        class_idx = hyp
    elif leakage_model == "ID_MASKED":
        if mask is None:
            raise ValueError("mask is required for ID_MASKED")
        class_idx = hyp ^ mask.astype(np.uint8)[:, None]
    elif leakage_model == "HW":
        class_idx = HW_TABLE[hyp]  # 0..255 Sbox value -> 0..8 Hamming weight
    else:
        raise ValueError(f"unknown leakage_model: {leakage_model!r}")

    if class_idx.max() >= probs.shape[1]:
        raise ValueError(
            f"class index {int(class_idx.max())} out of range for probs with {probs.shape[1]} "
            f"columns (leakage_model={leakage_model!r})"
        )

    p = probs[np.arange(n)[:, None], class_idx].astype(np.float64)
    return np.log(p + eps)
