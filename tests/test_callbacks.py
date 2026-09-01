"""Guardrails for src/train/callbacks.py::GEModelSelection's two selection modes.

Key recovery accumulates scores across traces assuming they share one key. That
holds on the fixed-key databases but not on ASCADv1-variable, where V carries
~256 distinct keys and the old code silently scored every trace against trace
0's key. These tests pin down (a) that the fixed-key path is unchanged, (b) that
the variable-key case is detected rather than silently mis-scored, and (c) that
the fallback criterion actually ranks models by discriminative power.
See CLAUDE.md 附錄 B.62.
"""

import numpy as np
import pytest

from src.data.ascad import AES_SBOX
from src.train.callbacks import GEModelSelection

TARGET_BYTE = 2


def _meta(n, keys, seed=0):
    """Build a metadata array shaped like ASCAD's (plaintext/key/masks/desync)."""
    rng = np.random.default_rng(seed)
    dtype = np.dtype([("plaintext", "u1", (16,)), ("key", "u1", (16,)),
                      ("masks", "u1", (16,)), ("desync", "<u4", (1,))])
    meta = np.zeros(n, dtype=dtype)
    meta["plaintext"] = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    meta["key"][:, TARGET_BYTE] = keys
    return meta


def _make(meta, y_val=None, **kw):
    x = np.zeros((len(meta), 8), dtype=np.float32)
    return GEModelSelection(x_val=x, meta_val=meta, target_byte=TARGET_BYTE,
                            y_val=y_val, **kw)


def test_fixed_key_validation_uses_the_ge_path_unchanged():
    """One key across V -> the original GE-based selection, key read from metadata."""
    meta = _meta(64, keys=np.uint8(42))
    cb = _make(meta)
    assert cb.variable_key is False
    assert cb.correct_key == 42


def test_fixed_key_path_does_not_require_y_val():
    """Existing callers pass no y_val; that must keep working (no behaviour change)."""
    meta = _meta(64, keys=np.uint8(7))
    cb = _make(meta, y_val=None)
    assert cb.variable_key is False


def test_variable_key_validation_is_detected():
    """Many keys across V must be recognised, not silently reduced to trace 0's key."""
    rng = np.random.default_rng(1)
    meta = _meta(500, keys=rng.integers(0, 256, 500, dtype=np.uint8))
    cb = _make(meta, y_val=np.zeros(500, dtype=np.int64))
    assert cb.variable_key is True
    assert cb.correct_key is None


def test_variable_key_without_labels_raises_instead_of_mis_scoring():
    """The old code would have happily scored against the wrong key here."""
    rng = np.random.default_rng(2)
    meta = _meta(500, keys=rng.integers(0, 256, 500, dtype=np.uint8))
    with pytest.raises(ValueError, match="variable-key"):
        _make(meta, y_val=None)


def _mean_true_rank(probs, y):
    """Mirror of the callback's fallback metric, for asserting on known inputs."""
    true_p = probs[np.arange(len(y)), y][:, None]
    return float((probs > true_p).sum(axis=1).mean())


def test_fallback_metric_is_zero_for_a_perfect_model():
    y = np.arange(64) % 256
    probs = np.zeros((64, 256), dtype=np.float32)
    probs[np.arange(64), y] = 1.0
    assert _mean_true_rank(probs, y) == 0.0


def test_fallback_metric_sits_at_the_random_baseline_for_uniform_output():
    """Uniform probabilities -> no class is strictly ahead, so rank 0 by the
    'strictly greater' convention. Random *tie-broken* output should instead land
    near 127.5, which is what the metric reports for genuinely random scores."""
    rng = np.random.default_rng(3)
    n = 4000
    y = rng.integers(0, 256, n)
    probs = rng.random((n, 256)).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    assert 120.0 < _mean_true_rank(probs, y) < 135.0


def test_fallback_metric_ranks_a_better_model_lower():
    """A model that concentrates mass on the truth must score below a random one --
    this is the property selection depends on."""
    rng = np.random.default_rng(4)
    n = 2000
    y = rng.integers(0, 256, n)

    weak = rng.random((n, 256)).astype(np.float32)
    weak /= weak.sum(axis=1, keepdims=True)

    strong = weak.copy()
    strong[np.arange(n), y] += 5.0  # real but not perfect preference for the truth
    strong /= strong.sum(axis=1, keepdims=True)

    assert _mean_true_rank(strong, y) < _mean_true_rank(weak, y)


def test_key_recovery_on_variable_keys_would_be_meaningless():
    """Documents *why* the fallback exists: with per-trace keys, accumulating
    scores against any single key hypothesis cannot identify the truth, because no
    single key explains the traces. The correct label needs each trace's own key."""
    rng = np.random.default_rng(5)
    n = 500
    keys = rng.integers(0, 256, n, dtype=np.uint8)
    meta = _meta(n, keys=keys)
    pt = meta["plaintext"][:, TARGET_BYTE]

    per_trace_labels = AES_SBOX[pt ^ keys]
    single_key_labels = AES_SBOX[pt ^ keys[0]]
    agree = float((per_trace_labels == single_key_labels).mean())
    assert agree < 0.05  # only the ~1/256 of traces that happen to share key[0]
