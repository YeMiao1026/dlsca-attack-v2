"""See CLAUDE.md §9 — hand-computed (p, k) -> S-box output comparison; HW range 0-8."""

import numpy as np
import pytest

from src.data.ascad import AES_SBOX
from src.data.labels import HW_TABLE, build, n_classes

META_DTYPE = np.dtype([
    ("plaintext", np.uint8, (16,)),
    ("key", np.uint8, (16,)),
    ("masks", np.uint8, (16,)),
    ("desync", np.uint32, (1,)),
])


def make_meta(plaintext_byte: np.ndarray, key_byte: np.ndarray, mask_byte: np.ndarray | None = None,
              target_byte: int = 2) -> np.ndarray:
    n = len(plaintext_byte)
    meta = np.zeros(n, dtype=META_DTYPE)
    meta["plaintext"][:, target_byte] = plaintext_byte
    meta["key"][:, target_byte] = key_byte
    if mask_byte is not None:
        meta["masks"][:, 0] = mask_byte
    return meta


def test_id_matches_hand_computed_sbox():
    # AES_Sbox[0x00] = 0x63, AES_Sbox[0x53] = 0xED (standard, well-known AES S-box values)
    plaintext_byte = np.array([0x00, 0x53], dtype=np.uint8)
    key_byte = np.array([0x00, 0x00], dtype=np.uint8)
    meta = make_meta(plaintext_byte, key_byte)

    y = build(meta, "ID", target_byte=2)

    assert y.tolist() == [0x63, 0xED]


def test_id_xors_plaintext_and_key_before_sbox():
    # Sbox[0x53 ^ 0x53] = Sbox[0x00] = 0x63
    plaintext_byte = np.array([0x53], dtype=np.uint8)
    key_byte = np.array([0x53], dtype=np.uint8)
    meta = make_meta(plaintext_byte, key_byte)

    y = build(meta, "ID", target_byte=2)

    assert y.tolist() == [0x63]


def test_id_masked_xors_in_the_mask():
    plaintext_byte = np.array([0x00, 0x53], dtype=np.uint8)
    key_byte = np.array([0x00, 0x00], dtype=np.uint8)
    mask_byte = np.array([0x01, 0xFF], dtype=np.uint8)
    meta = make_meta(plaintext_byte, key_byte, mask_byte)

    y = build(meta, "ID_MASKED", target_byte=2, mask_index=0)

    expected = AES_SBOX[plaintext_byte ^ key_byte] ^ mask_byte
    assert np.array_equal(y, expected.astype(np.int64))


def test_id_masked_requires_mask_index():
    meta = make_meta(np.array([0], dtype=np.uint8), np.array([0], dtype=np.uint8))
    with pytest.raises(ValueError):
        build(meta, "ID_MASKED", target_byte=2)


def test_hw_table_known_values():
    # 0x00 = 0b00000000 -> 0 ones, 0x63 = 0b01100011 -> 4 ones, 0xFF -> 8 ones
    assert HW_TABLE[0x00] == 0
    assert HW_TABLE[0x63] == 4
    assert HW_TABLE[0xFF] == 8


def test_hw_labels_are_in_range_0_to_8():
    plaintext_byte = np.arange(256, dtype=np.uint8)
    key_byte = np.zeros(256, dtype=np.uint8)
    meta = make_meta(plaintext_byte, key_byte)

    y = build(meta, "HW", target_byte=2)

    assert y.min() >= 0
    assert y.max() <= 8
    assert np.array_equal(y, HW_TABLE[AES_SBOX[plaintext_byte]].astype(np.int64))


def test_n_classes():
    assert n_classes("ID") == 256
    assert n_classes("ID_MASKED") == 256
    assert n_classes("HW") == 9


def test_unknown_leakage_model_raises():
    meta = make_meta(np.array([0], dtype=np.uint8), np.array([0], dtype=np.uint8))
    with pytest.raises(ValueError):
        build(meta, "BOGUS", target_byte=2)
    with pytest.raises(ValueError):
        n_classes("BOGUS")
