"""Guardrail tests for src/attack/keyrank.py. See CLAUDE.md §9 — these two
extreme cases are meant to be written before anything else built on top of
keyrank.py; they pin down the two failure modes (never-converges,
always-converges) that a scoring/ranking bug would otherwise hide.
"""

import numpy as np

from src.attack import keyrank, scores
from src.data.ascad import AES_SBOX


def test_keyrank_perfect_prediction_converges_within_one_trace():
    """A model that puts probability 1.0 on the correct class must recover the
    key immediately: GE == 0 and SR1 == 1 from the very first trace.
    """
    rng = np.random.default_rng(7)
    n_samples, target_byte, correct_key = 50, 2, 42
    plaintexts = rng.integers(0, 256, (n_samples, 16), dtype=np.uint8)

    true_class = AES_SBOX[plaintexts[:, target_byte] ^ np.uint8(correct_key)]
    probs = np.zeros((n_samples, 256), dtype=np.float32)
    probs[np.arange(n_samples), true_class] = 1.0

    sc = scores.build(probs, plaintexts, target_byte)
    ranks = keyrank.evaluate(sc, correct_key, n_runs=20, max_traces=5, seed=1)

    ge_curve = keyrank.ge(ranks)
    sr1_curve = keyrank.sr1(ranks)

    assert np.all(ranks == 0)
    assert np.array_equal(ge_curve, np.zeros(5))
    assert np.array_equal(sr1_curve, np.ones(5))
    assert keyrank.n_tge(ge_curve) == 1
    assert keyrank.n_sr90(sr1_curve) == 1


def test_keyrank_uniform_prediction_stays_near_midpoint():
    """A model with no information about the key (probabilities drawn
    independently of the true label) should rank the correct key like a
    random guess among 256 candidates: GE hovers around 127.5, never converges.
    """
    rng = np.random.default_rng(2024)
    n_samples, target_byte, correct_key = 5000, 2, 17
    plaintexts = rng.integers(0, 256, (n_samples, 16), dtype=np.uint8)
    probs = rng.dirichlet(np.ones(256), size=n_samples).astype(np.float32)

    sc = scores.build(probs, plaintexts, target_byte)
    ranks = keyrank.evaluate(sc, correct_key, n_runs=100, max_traces=300, seed=999)
    ge_curve = keyrank.ge(ranks)

    assert abs(ge_curve.mean() - 127.5) < 20
    assert np.all(np.abs(ge_curve - 127.5) < 50)
    assert keyrank.n_tge(ge_curve) is None


def test_n_tge_partial_convergence_returns_plain_int():
    """Regression test: `_first_sustained` used to return `np.int64` (not
    JSON-serializable, breaking `json.dumps(metrics)` in 03_evaluate.py)
    whenever the threshold was crossed partway through the curve. The two
    tests above only exercise the all-converged and never-converged branches
    — neither would have caught this, which is exactly how it shipped.
    """
    ge_curve = np.array([50.0, 20.0, 5.0, 0.5, 0.5, 0.5, 0.5])
    n_tge = keyrank.n_tge(ge_curve, threshold=1.0)
    assert n_tge == 4
    assert type(n_tge) is int

    sr1_curve = np.array([0.1, 0.2, 0.5, 0.95, 0.95, 0.95])
    n_sr90 = keyrank.n_sr90(sr1_curve, threshold=0.9)
    assert n_sr90 == 4
    assert type(n_sr90) is int
