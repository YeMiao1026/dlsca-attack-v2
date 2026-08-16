"""Cross-correlation-based trace realignment for desynchronized databases.

Motivation (CLAUDE.md 附錄 B.28-B.30): per-point SNR on ASCAD_desync50/100
collapses to the noise floor (peak ~0.01, indistinguishable from the
unmasked control) even for the easiest diagnostic target (masked label),
and 700/700 points sit above 50% of that already-tiny peak — i.e. there is
no localized point of interest left for a fixed-position classifier to find.
This module blindly re-aligns each trace to a reference via windowed
cross-correlation, without touching the ground-truth `desync` metadata field
(a real attacker on unknown hardware would not have per-trace jitter values;
that field exists only for evaluating whether blind alignment worked).
"""

from __future__ import annotations

import numpy as np


def resync(traces: np.ndarray, reference: np.ndarray, max_shift: int) -> tuple[np.ndarray, np.ndarray]:
    """Align every row of `traces` to `reference` by the integer shift (within
    +-max_shift) that maximizes cross-correlation. Returns (aligned, shifts);
    aligned traces are rolled into `reference`'s time frame and re-wrap at the
    edges (acceptable here since informative content lives away from the
    boundary once the desync window's max offset is respected).
    """
    traces = np.asarray(traces, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    n, length = traces.shape
    ref_centered = reference - reference.mean()
    tr_centered = traces - traces.mean(axis=1, keepdims=True)

    shifts = np.arange(-max_shift, max_shift + 1)
    scores = np.empty((n, len(shifts)))
    for i, s in enumerate(shifts):
        if s >= 0:
            a = tr_centered[:, s:]
            b = ref_centered[: length - s]
        else:
            a = tr_centered[:, : length + s]
            b = ref_centered[-s:]
        scores[:, i] = a @ b

    best_shifts = shifts[np.argmax(scores, axis=1)]
    aligned = np.empty_like(traces)
    for i in range(n):
        aligned[i] = np.roll(traces[i], -int(best_shifts[i]))
    return aligned.astype(np.float32), best_shifts


def resync_iterative(
    traces: np.ndarray, max_shift: int, rounds: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align `traces` to each other with no external reference: round 0 uses
    traces[0] as a single-trace template, then each subsequent round
    re-aligns against the mean of the previous round's output (a sharper
    template once traces are roughly on-grid).

    Returns (aligned, shifts, reference) — `reference` is the final template,
    needed to align a *different* set of traces (e.g. V/D/E) into the same
    time frame via `resync(other_traces, reference, max_shift)`.
    """
    reference = traces[0]
    aligned, shifts = resync(traces, reference, max_shift)
    for _ in range(rounds - 1):
        reference = aligned.mean(axis=0)
        aligned, shifts = resync(traces, reference, max_shift)
    return aligned, shifts, reference
