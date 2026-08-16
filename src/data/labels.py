"""Leakage models -> labels. See CLAUDE.md §5.2: ID / ID_MASKED / HW."""

from __future__ import annotations

from typing import Literal

import numpy as np

from src.data.ascad import AES_SBOX

HW_TABLE = np.array([bin(v).count("1") for v in range(256)], dtype=np.uint8)

LeakageModel = Literal["ID", "ID_MASKED", "HW"]


def build(meta: np.ndarray, leakage_model: LeakageModel, target_byte: int, mask_index: int | None = None) -> np.ndarray:
    """Return y (N,) int labels. `mask_index` is required for ID_MASKED (see ascad.find_mask_index)."""
    plaintext_byte = meta["plaintext"][:, target_byte].astype(np.uint8)
    key_byte = meta["key"][:, target_byte].astype(np.uint8)
    z = AES_SBOX[plaintext_byte ^ key_byte]

    if leakage_model == "ID":
        return z.astype(np.int64)
    if leakage_model == "ID_MASKED":
        if mask_index is None:
            raise ValueError("mask_index is required for ID_MASKED")
        mask = meta["masks"][:, mask_index].astype(np.uint8)
        return (z ^ mask).astype(np.int64)
    if leakage_model == "HW":
        return HW_TABLE[z].astype(np.int64)
    raise ValueError(f"unknown leakage_model: {leakage_model!r}")


def n_classes(leakage_model: LeakageModel) -> int:
    """256 for ID/ID_MASKED, 9 for HW."""
    if leakage_model in ("ID", "ID_MASKED"):
        return 256
    if leakage_model == "HW":
        return 9
    raise ValueError(f"unknown leakage_model: {leakage_model!r}")
