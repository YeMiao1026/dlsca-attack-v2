"""See CLAUDE.md §9 — four sets pairwise disjoint; correct totals; same-seed reproducibility."""

import numpy as np
import pytest

from src.data.split import SplitIndices, four_way, load, save

N_PROFILING = 50_000
N_ATTACK = 10_000
N_ATTACKER = 30_000
N_VAL = 5_000
N_DEFENDER = 15_000


def make_split(seed: int = 42) -> SplitIndices:
    return four_way(N_PROFILING, N_ATTACK, N_ATTACKER, N_VAL, N_DEFENDER, seed)


def test_set_sizes_are_correct():
    s = make_split()
    assert len(s.a) == N_ATTACKER
    assert len(s.v) == N_VAL
    assert len(s.d) == N_DEFENDER
    assert len(s.e) == N_ATTACK


def test_a_v_d_are_pairwise_disjoint():
    s = make_split()
    a, v, d = set(s.a.tolist()), set(s.v.tolist()), set(s.d.tolist())
    assert a & v == set()
    assert a & d == set()
    assert v & d == set()


def test_a_v_d_are_within_profiling_range_and_e_within_attack_range():
    s = make_split()
    assert s.a.min() >= 0 and s.a.max() < N_PROFILING
    assert s.v.min() >= 0 and s.v.max() < N_PROFILING
    assert s.d.min() >= 0 and s.d.max() < N_PROFILING
    assert s.e.min() >= 0 and s.e.max() < N_ATTACK


def test_e_is_never_mixed_with_profiling_indices():
    # E is drawn from the independent attack set, so it is not required to be
    # disjoint from A/V/D's *index values* (different index space entirely) —
    # this test instead pins down that E is exactly the full attack set.
    s = make_split()
    assert np.array_equal(s.e, np.arange(N_ATTACK))


def test_same_seed_reproduces_identical_split():
    s1 = make_split(seed=7)
    s2 = make_split(seed=7)
    assert np.array_equal(s1.a, s2.a)
    assert np.array_equal(s1.v, s2.v)
    assert np.array_equal(s1.d, s2.d)
    assert np.array_equal(s1.e, s2.e)


def test_different_seed_gives_different_split():
    s1 = make_split(seed=7)
    s2 = make_split(seed=8)
    assert not np.array_equal(s1.a, s2.a)


def test_oversized_request_raises():
    with pytest.raises(ValueError):
        four_way(n_profiling=1000, n_attack=10, n_attacker=800, n_val=800, n_defender=800, seed=0)


def test_save_and_load_round_trip(tmp_path):
    s = make_split()
    path = tmp_path / "split_indices.npz"
    save(s, str(path))
    loaded = load(str(path))
    assert np.array_equal(s.a, loaded.a)
    assert np.array_equal(s.v, loaded.v)
    assert np.array_equal(s.d, loaded.d)
    assert np.array_equal(s.e, loaded.e)
