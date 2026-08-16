"""See CLAUDE.md §9 — the correct key's cumulative score should outrank a random
key's; no NaN/inf (pitfall #1: log-sum, not probability product, must stay
finite even when a hypothesis's probability mass is exactly 0).
"""

import numpy as np

from src.attack.scores import build
from src.data.ascad import AES_SBOX
from src.data.labels import HW_TABLE


def test_scores_shape_and_dtype():
    rng = np.random.default_rng(0)
    n = 20
    plaintexts = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    probs = rng.dirichlet(np.ones(256), size=n).astype(np.float32)

    sc = build(probs, plaintexts, target_byte=2)

    assert sc.shape == (n, 256)
    assert sc.dtype == np.float64


def test_no_nan_or_inf_even_with_exact_zero_probabilities():
    rng = np.random.default_rng(1)
    n = 20
    plaintexts = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    probs = np.zeros((n, 256), dtype=np.float32)
    probs[np.arange(n), 0] = 1.0  # every trace: all mass on class 0, rest exactly 0

    sc = build(probs, plaintexts, target_byte=2)

    assert np.all(np.isfinite(sc))


def test_correct_key_cumulative_score_beats_random_keys():
    rng = np.random.default_rng(2)
    n, target_byte, correct_key = 200, 2, 91
    plaintexts = rng.integers(0, 256, (n, 16), dtype=np.uint8)

    true_class = AES_SBOX[plaintexts[:, target_byte] ^ np.uint8(correct_key)]
    probs = np.full((n, 256), 0.3 / 255, dtype=np.float64)
    probs[np.arange(n), true_class] = 0.7
    probs = probs.astype(np.float32)

    sc = build(probs, plaintexts, target_byte)
    cumulative = sc.sum(axis=0)  # (256,)

    assert cumulative.argmax() == correct_key
    other_keys = [k for k in range(256) if k != correct_key]
    assert np.all(cumulative[correct_key] > cumulative[other_keys])


def test_masked_scores_require_the_mask_to_recover_the_key():
    """Regression test for a real bug: an ID_MASKED model predicts
    Z' = Sbox[p^k] ^ mask[i] (mask varies per trace), so scoring must XOR the
    same mask into the hypothesis before indexing probs — an early E08 run
    trained fine (loss dropped) but never converged because scores.build
    ignored the mask entirely and scored against the unmasked hypothesis.
    """
    rng = np.random.default_rng(4)
    n, target_byte, correct_key = 300, 2, 77
    plaintexts = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    mask = rng.integers(0, 256, n, dtype=np.uint8)

    true_class = AES_SBOX[plaintexts[:, target_byte] ^ np.uint8(correct_key)] ^ mask
    probs = np.full((n, 256), 0.3 / 255, dtype=np.float64)
    probs[np.arange(n), true_class] = 0.7
    probs = probs.astype(np.float32)

    sc_with_mask = build(probs, plaintexts, target_byte, leakage_model="ID_MASKED", mask=mask)
    cumulative_with_mask = sc_with_mask.sum(axis=0)
    assert cumulative_with_mask.argmax() == correct_key

    # leakage_model left at the "ID" default: mask is silently not applied,
    # scoring against the wrong (unmasked) hypothesis — must not recover the key.
    sc_without_mask = build(probs, plaintexts, target_byte)
    cumulative_without_mask = sc_without_mask.sum(axis=0)
    assert cumulative_without_mask.argmax() != correct_key


def test_hw_leakage_scoring_recovers_the_key():
    """HW leakage: the model predicts HW(Sbox[p^k]) (9 classes, 0..8), so the
    hypothesis must be mapped through HW_TABLE before indexing probs — same
    "score against what the model actually predicts" principle as the mask
    fix above, just a different, coarser many-to-one mapping.
    """
    rng = np.random.default_rng(5)
    n, target_byte, correct_key = 500, 2, 133
    plaintexts = rng.integers(0, 256, (n, 16), dtype=np.uint8)

    true_hw = HW_TABLE[AES_SBOX[plaintexts[:, target_byte] ^ np.uint8(correct_key)]]
    probs = np.full((n, 9), 0.3 / 8, dtype=np.float64)
    probs[np.arange(n), true_hw] = 0.7
    probs = probs.astype(np.float32)

    sc = build(probs, plaintexts, target_byte, leakage_model="HW")
    assert sc.shape == (n, 256)
    cumulative = sc.sum(axis=0)
    assert cumulative.argmax() == correct_key


def test_class_index_out_of_range_raises():
    # leakage_model="ID" needs 256 columns; a 9-column (HW-shaped) probs array
    # must fail loudly, not silently index garbage or raise a confusing IndexError.
    rng = np.random.default_rng(6)
    n = 10
    plaintexts = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    probs = rng.dirichlet(np.ones(9), size=n).astype(np.float32)

    try:
        build(probs, plaintexts, target_byte=2, leakage_model="ID")
        raise AssertionError("should have raised ValueError")
    except ValueError:
        pass


def test_log_summation_does_not_underflow_past_fifty_traces():
    # Pitfall #1: naive probability *multiplication* underflows to exactly 0
    # well before 100 traces. Summing logs must stay informative past that.
    rng = np.random.default_rng(3)
    n, target_byte, correct_key = 120, 2, 5
    plaintexts = rng.integers(0, 256, (n, 16), dtype=np.uint8)

    true_class = AES_SBOX[plaintexts[:, target_byte] ^ np.uint8(correct_key)]
    probs = np.full((n, 256), 0.3 / 255, dtype=np.float64)
    probs[np.arange(n), true_class] = 0.7
    probs = probs.astype(np.float32)

    sc = build(probs, plaintexts, target_byte)
    cumulative_after_80 = sc[:80].sum(axis=0)

    assert np.all(np.isfinite(cumulative_after_80))
    assert cumulative_after_80.argmax() == correct_key
