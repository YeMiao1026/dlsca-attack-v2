"""Log-likelihood score matrix. See CLAUDE.md §5.5.1 — scores MUST be summed
in log space, never multiplied as raw probabilities (pitfall #1: probability
products underflow to exactly 0 past ~50 traces and GE gets stuck).
"""

from __future__ import annotations

import numpy as np

from src.data.ascad import AES_SBOX


def build(probs: np.ndarray, plaintexts: np.ndarray, target_byte: int, eps: float = 1e-40) -> np.ndarray:
    """Return scores (N, 256) float64: score[i,k] = log(probs[i, Sbox[p[i,byte]^k]] + eps)."""
    n = probs.shape[0]
    plaintext_byte = plaintexts[:, target_byte].astype(np.uint8)
    key_hypotheses = np.arange(256, dtype=np.uint8)
    hyp = AES_SBOX[plaintext_byte[:, None] ^ key_hypotheses[None, :]]  # (N, 256)
    p = probs[np.arange(n)[:, None], hyp].astype(np.float64)
    return np.log(p + eps)
