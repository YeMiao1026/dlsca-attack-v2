#!/usr/bin/env python3
"""Stage 0 data sanity check. See CLAUDE.md §5.2 stage 0 — run before writing
any model. A failure here means mask-index selection or metadata parsing is
wrong (pitfall #2), and everything built afterwards would be built on sand.

Usage (run from repo root):
    python scripts/00_inspect_data.py --h5 data/ASCAD.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data.ascad import AES_SBOX, get_correct_key, load
from src.data.split import four_way
from src.metrics.leakage import snr

SNR_SIGNIFICANCE_RATIO = 10.0  # masked-label SNR peak must exceed this multiple of the unmasked control


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASCAD stage-0 data sanity check (CLAUDE.md §5.2 stage 0)")
    p.add_argument("--h5", default="data/ASCAD.h5", help="path to an ASCAD .h5 database")
    p.add_argument("--target-byte", type=int, default=2)
    p.add_argument("--n-attacker", type=int, default=30000, help="size of A, see CLAUDE.md §5.2 stage 1")
    p.add_argument("--n-val", type=int, default=5000, help="size of V")
    p.add_argument("--n-defender", type=int, default=15000, help="size of D")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--mask-index", type=int, default=None,
                    help="skip auto-detection and use this masks[] column directly. Needed for "
                         "desynchronized databases, where per-point SNR is too smeared to reliably "
                         "find the mask index on its own — determine it once on ASCAD.h5 (desync0) "
                         "and pass it here, since the mask assignment is a property of the physical "
                         "campaign, not of the desync applied afterward.")
    return p.parse_args()


def describe_traces(name: str, traces: np.ndarray) -> None:
    print(f"  {name:<3s} shape={str(traces.shape):<16} dtype={str(traces.dtype):<8} "
          f"range=[{traces.min()}, {traces.max()}] std={traces.astype(np.float64).std():.4f}")


def describe_desync(name: str, meta: np.ndarray) -> None:
    d = meta["desync"].reshape(-1)
    print(f"  {name:<10s} min={d.min()} max={d.max()} mean={d.mean():.2f} unique_values={len(np.unique(d))}")


def main() -> None:
    args = parse_args()

    print(f"=== loading {args.h5} ===")
    data = load(args.h5)
    print(f"  profiling: traces={data.profiling_traces.shape} meta_fields={data.profiling_meta.dtype.names}")
    print(f"  attack:    traces={data.attack_traces.shape} meta_fields={data.attack_meta.dtype.names}")

    print()
    print("=== step 1: four-way split preview (shape / dtype / range / std) ===")
    split = four_way(
        n_profiling=len(data.profiling_traces),
        n_attack=len(data.attack_traces),
        n_attacker=args.n_attacker,
        n_val=args.n_val,
        n_defender=args.n_defender,
        seed=args.split_seed,
    )
    describe_traces("A", data.profiling_traces[split.a])
    describe_traces("V", data.profiling_traces[split.v])
    describe_traces("D", data.profiling_traces[split.d])
    describe_traces("E", data.attack_traces[split.e])

    print()
    print("=== step 2: correct key (read from metadata, never hardcoded) ===")
    correct_key = get_correct_key(data.attack_meta, args.target_byte)
    print(f"  attack_meta['key'][0][{args.target_byte}] = {correct_key} (0x{correct_key:02x})")

    print()
    print("=== steps 3-4: mask index scan + unmasked-label control (computed on A) ===")
    a_traces = data.profiling_traces[split.a].astype(np.float64)
    a_meta = data.profiling_meta[split.a]

    plaintext_byte = a_meta["plaintext"][:, args.target_byte].astype(np.uint8)
    key_byte = a_meta["key"][:, args.target_byte].astype(np.uint8)
    unmasked = AES_SBOX[plaintext_byte ^ key_byte]

    if args.mask_index is not None:
        mask_index = args.mask_index
        print(f"  mask index: {mask_index} (given via --mask-index, auto-detection skipped)")
    else:
        n_mask_cols = a_meta["masks"].shape[1]
        col_peaks = [snr(a_traces, unmasked ^ a_meta["masks"][:, i].astype(np.uint8)).max()
                     for i in range(n_mask_cols)]
        ranked = sorted(col_peaks, reverse=True)
        mask_index = int(np.argmax(col_peaks))
        print(f"  mask index selected: {mask_index}")
        if len(ranked) > 1 and ranked[0] < 2 * ranked[1]:
            print(f"  WARNING: winner ({ranked[0]:.4f}) is not clearly separated from the runner-up "
                  f"({ranked[1]:.4f}) — likely desync smearing per-point SNR into noise, not a real "
                  f"peak (pointwise SNR degrades badly under desync; large-kernel CNNs exist for this "
                  f"reason, see CLAUDE.md §6.1). Re-run against ASCAD.h5 (desync0) to find the true "
                  f"mask index, then pass it here via --mask-index instead of trusting this pick.")

    masked = unmasked ^ a_meta["masks"][:, mask_index].astype(np.uint8)

    snr_masked = snr(a_traces, masked)
    snr_unmasked = snr(a_traces, unmasked)
    peak_masked, poi_masked = float(snr_masked.max()), int(snr_masked.argmax())
    peak_unmasked, poi_unmasked = float(snr_unmasked.max()), int(snr_unmasked.argmax())
    concentrated = int((snr_masked >= 0.5 * peak_masked).sum())

    print(f"  masked-label   SNR peak: {peak_masked:.4f} at point {poi_masked}")
    print(f"  unmasked-label SNR peak: {peak_unmasked:.4f} at point {poi_unmasked}  (control, should be ~0)")
    print(f"  points >= 50% of masked-label peak: {concentrated} / {a_traces.shape[1]} (should be a handful, not spread out)")

    print()
    print("=== step 5: desync field distribution ===")
    describe_desync("profiling", data.profiling_meta)
    describe_desync("attack", data.attack_meta)

    print()
    print("=== pass/fail check ===")
    passed = peak_masked > SNR_SIGNIFICANCE_RATIO * max(peak_unmasked, 1e-6)
    print(f"  masked-label SNR peak >= {SNR_SIGNIFICANCE_RATIO:g}x unmasked-label control: "
          f"{'PASS' if passed else 'FAIL'} ({peak_masked:.4f} vs {peak_unmasked:.4f})")
    if not passed:
        print("  -> either mask index is wrong / metadata parsing is broken (CLAUDE.md pitfall #2),")
        print("     or this is a desynchronized database where per-point SNR is expected to be this")
        print("     weak — re-check with ASCAD.h5 (desync0) and, if that passes cleanly, pass its")
        print("     mask index here via --mask-index instead of auto-detecting on desynced data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
