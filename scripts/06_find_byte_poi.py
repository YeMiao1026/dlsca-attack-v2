#!/usr/bin/env python3
"""Locate the leakage window for a target byte in the ASCAD *raw* traces.

Why this is needed: ASCAD_generate.py extracts a fixed `target_points` window,
and every published window (and this project's whole history) is byte 2's
[45400, 46100). To evaluate any other byte you first have to find where that
byte leaks in the 100,000-point raw trace — the S-box bytes are processed
sequentially, so each one leaks somewhere else entirely.

Two things are searched for, and neither is assumed:

  1. Which `masks` column holds the target byte's mask. CLAUDE.md 附錄 B.61
     established that the fixed-key file stores bytes 2..15 at indices 0..13
     (bytes 0-1 are unmasked), so byte b "should" be at column b-2 — but that
     is inferred from one byte, so this script verifies it by scanning every
     column and reporting the winner against the noise floor.
  2. Where the masked value Z' = Sbox[p[b] ^ k[b]] ^ mask leaks. Only the
     masked label is usable: the unmasked label's SNR is ~0 by construction
     (that is the point of masking, and 00_inspect_data.py uses exactly this
     as its control).

Traces are subsampled and the scan is chunked over the time axis, because the
raw file is 60,000 x 100,000 int8 (6 GB) and does not need to be held whole.

Usage (run from repo root):
    python scripts/06_find_byte_poi.py --raw <raw>.h5 --target-byte 3
    python scripts/06_find_byte_poi.py --raw <raw>.h5 --target-byte 3 \
        --n-traces 10000 --window 700
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np

from src.data.ascad import AES_SBOX
from src.metrics.leakage import snr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find a target byte's leakage window in ASCAD raw traces")
    p.add_argument("--raw", required=True, help="path to ATMega8515_raw_traces.h5")
    p.add_argument("--target-byte", type=int, required=True)
    p.add_argument("--n-traces", type=int, default=8000,
                    help="how many traces to subsample for the SNR scan (SNR converges long "
                         "before the full 60k; the scan cost is linear in this)")
    p.add_argument("--window", type=int, default=700,
                    help="width of the extraction window to propose, matching byte 2's 700")
    p.add_argument("--chunk", type=int, default=10000, help="time-axis chunk size for the scan")
    p.add_argument("--mask-column", type=int, default=None,
                    help="skip the mask-column search and use this column directly")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    b = args.target_byte

    with h5py.File(args.raw, "r") as f:
        meta = f["metadata"]
        n_total = len(meta)
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(n_total, size=min(args.n_traces, n_total), replace=False))

        print(f"=== {args.raw} ===")
        print(f"  traces: {f['traces'].shape}  metadata fields: {meta.dtype.names}")
        print(f"  scanning {len(idx)} subsampled traces for byte {b}")

        md = meta[idx]
        pt = md["plaintext"][:, b].astype(np.uint8)
        key = md["key"][:, b].astype(np.uint8)
        masks = md["masks"]
        unmasked = AES_SBOX[pt ^ key]

        n_points = f["traces"].shape[1]

        # Mask-column search on a coarse slice first: loading all 100k points for
        # every candidate column would be 18x the work, and the column that owns
        # this byte shows up wherever it leaks, so a single chunk containing real
        # leakage is enough to rank them. Byte 2's window sits at 45400, and the
        # bytes are processed in order, so the middle of the trace is a reasonable
        # place to look for a first signal.
        if args.mask_column is not None:
            mask_col = args.mask_column
            print(f"  mask column: {mask_col} (given)")
        else:
            print(f"  searching {masks.shape[1]} mask columns over the full trace (chunked)...")
            best = np.zeros(masks.shape[1])
            for start in range(0, n_points, args.chunk):
                stop = min(start + args.chunk, n_points)
                block = np.array(f["traces"][idx, start:stop])
                for c in range(masks.shape[1]):
                    z = unmasked ^ masks[:, c].astype(np.uint8)
                    best[c] = max(best[c], float(snr(block, z).max()))
                print(f"    points [{start}:{stop}) done", end="\r")
            print()
            floor = float(np.median(best))
            mask_col = int(np.argmax(best))
            ranked = np.argsort(best)[::-1]
            print(f"  mask column peaks (top 4): {[(int(c), round(float(best[c]), 4)) for c in ranked[:4]]}")
            print(f"  noise floor (median): {floor:.4f}")
            if floor <= 0 or best[mask_col] < 10.0 * floor:
                print(f"  WARNING: winner {best[mask_col]:.4f} does not clear 10x the noise floor — "
                       f"the byte's mask column was not identified reliably.")
            expected = b - 2
            print(f"  selected mask column: {mask_col} "
                   f"({'matches' if mask_col == expected else 'DIFFERS FROM'} the byte-2 pattern's "
                   f"prediction of {expected})")

        # Full-resolution SNR for the chosen column, to place the window.
        masked = unmasked ^ masks[:, mask_col].astype(np.uint8)
        curve = np.empty(n_points, dtype=np.float64)
        print("  computing full-resolution SNR for the selected column...")
        for start in range(0, n_points, args.chunk):
            stop = min(start + args.chunk, n_points)
            block = np.array(f["traces"][idx, start:stop])
            curve[start:stop] = snr(block, masked)
            print(f"    points [{start}:{stop}) done", end="\r")
        print()

        control = snr(np.array(f["traces"][idx, :args.chunk]), unmasked).max()

    peak_i = int(curve.argmax())
    peak = float(curve[peak_i])
    half = int((curve >= 0.5 * peak).sum())

    # Centre the window on the peak, then clamp into range.
    start = max(0, min(peak_i - args.window // 2, n_points - args.window))
    stop = start + args.window

    print()
    print(f"  masked-label SNR peak : {peak:.4f} at raw point {peak_i}")
    print(f"  unmasked-label control: {control:.4f}  (should be ~0)")
    print(f"  points >= 50% of peak : {half} / {n_points}")
    print()
    print(f"  PROPOSED WINDOW for byte {b}: [{start}, {stop})   (byte 2's is [45400, 46100))")
    print(f'  target_points : [n for n in range({start},{stop})]')
    print(f"  mask column   : {mask_col}")
    print()
    if peak < 10.0 * control:
        print("  RESULT: FAIL — masked-label peak does not clear 10x the unmasked control; "
              "do not extract a database from this window.")
        sys.exit(1)
    print("  RESULT: PASS")


if __name__ == "__main__":
    main()
