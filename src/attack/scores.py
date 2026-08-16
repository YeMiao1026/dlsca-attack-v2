"""Log-likelihood score matrix. See CLAUDE.md §5.5.1 — scores MUST be summed
in log space, never multiplied as raw probabilities (pitfall #1: probability
products underflow to exactly 0 past ~50 traces and GE gets stuck).
"""

from __future__ import annotations

import numpy as np

from src.data.ascad import AES_SBOX


def build(probs: np.ndarray, plaintexts: np.ndarray, target_byte: int, eps: float = 1e-40,
          mask: np.ndarray | None = None) -> np.ndarray:
    """Return scores (N, 256) float64: score[i,k] = log(probs[i, Sbox[p[i,byte]^k]] + eps).

    `mask` (shape (N,), one already-selected masks[:, mask_index] column) is
    required for ID_MASKED: the model was trained to predict
    Z' = Sbox[p^k] ^ mask[i], not the unmasked Sbox[p^k], so the hypothesis
    must be XORed with the same per-trace mask value before indexing `probs`
    — scoring against the unmasked hypothesis silently tests the wrong thing
    and never converges even when the model has clearly learned Z' (this bit
    an early E08 run: loss dropped fine, GE never moved).

    Only 256-class leakage models (ID / ID_MASKED) are supported: the
    hypothesis index Sbox[p^k] ranges over 0..255 and is used to index
    `probs` directly. HW (9 classes) would need HW_TABLE[Sbox[p^k]] mapped
    in first — not implemented, hence the explicit guard below instead of
    letting a HW-shaped probs array fail with a confusing IndexError.
    """
    if probs.shape[1] != 256:
        raise NotImplementedError(
            f"scores.build only supports 256-class leakage models (ID/ID_MASKED); "
            f"got probs with {probs.shape[1]} classes (e.g. HW is not supported here yet)"
        )
    n = probs.shape[0]
    plaintext_byte = plaintexts[:, target_byte].astype(np.uint8)
    key_hypotheses = np.arange(256, dtype=np.uint8)
    hyp = AES_SBOX[plaintext_byte[:, None] ^ key_hypotheses[None, :]]  # (N, 256)
    if mask is not None:
        hyp = hyp ^ mask.astype(np.uint8)[:, None]
    p = probs[np.arange(n)[:, None], hyp].astype(np.float64)
    return np.log(p + eps)
