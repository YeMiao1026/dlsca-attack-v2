#!/usr/bin/env python3
"""Import real ChipWhisperer captures into this project's ASCAD-format pipeline.

Motivation: everything in this project (and the whole defense-side sub-project)
has been evaluated on ASCAD — a simulated-in-the-sense-of-prepackaged database.
The team also holds 10,000 traces captured on real hardware (ChipWhisperer +
STM32, unprotected software AES, fixed key 2b7e1516...). Running the existing
defenses against those measurements turns "validated on a public database" into
"validated on our own physical measurements", which is the application-facing
claim worth making.

Two differences from ASCAD that this script has to handle explicitly:

  1. dtype/scale. CW stores float64 in the ADC's roughly [-0.5, 0.5] range, but
     src/data/ascad.py loads with dtype=np.int8 — casting those floats directly
     would truncate every sample to 0 and silently destroy the data. So the
     traces are linearly rescaled to the full int8 range here, which preserves
     all relative structure (the Standardizer normalises afterwards anyway) and
     keeps the file byte-compatible with every other database in the project.
     Quantisation cost is negligible at this signal strength: the step is ~2.4%
     of a trace's standard deviation against an ID-label SNR of 6-12.6.

  2. No masking. This is an unprotected implementation, so there is no `masks`
     field and Sbox[p^k] leaks directly (measured ID-SNR 12.6 on byte 0, versus
     ~0.011 for the same unmasked label on masked ASCAD). A zero-filled masks
     column is written so the file layout matches, and 00_inspect_data.py
     detects the all-constant masks and switches to the unprotected-device check.

Usage (run from repo root):
    python scripts/08_import_cw.py \
        --traces .../cw_traces.npy --plaintexts .../cw_textins.npy \
        --keys .../cw_keys.npy --out data/CW_stm32.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np

from src.data.ascad import AES_SBOX


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import ChipWhisperer captures as an ASCAD-format database")
    p.add_argument("--traces", required=True, help="cw_traces.npy (n, points) float")
    p.add_argument("--plaintexts", required=True, help="cw_textins.npy (n, 16) uint8")
    p.add_argument("--keys", required=True, help="cw_keys.npy (n, 16) uint8")
    p.add_argument("--out", required=True)
    p.add_argument("--target-byte", type=int, default=0,
                    help="byte used for the stored `labels` dataset only; the pipeline itself "
                         "derives labels from metadata, so this does not constrain later analysis")
    p.add_argument("--n-profiling", type=int, default=8000)
    p.add_argument("--n-attack", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    traces = np.load(args.traces)
    plaintexts = np.load(args.plaintexts).astype(np.uint8)
    keys = np.load(args.keys).astype(np.uint8)

    n, n_points = traces.shape
    if len(plaintexts) != n or len(keys) != n:
        raise ValueError(f"length mismatch: traces {n}, plaintexts {len(plaintexts)}, keys {len(keys)}")
    need = args.n_profiling + args.n_attack
    if need > n:
        raise ValueError(f"need {need} traces but only {n} available")

    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists — refusing to overwrite")

    print(f"=== {args.traces} ===")
    print(f"  {n} traces x {n_points} points, dtype {traces.dtype}, range [{traces.min():.4f}, {traces.max():.4f}]")

    lo, hi = float(traces.min()), float(traces.max())
    if hi <= lo:
        raise ValueError("degenerate trace range")
    scaled = (traces - lo) / (hi - lo) * 255.0 - 128.0
    quantised = np.clip(np.round(scaled), -128, 127).astype(np.int8)

    step = (hi - lo) / 255.0
    print(f"  rescaled to int8: step {step:.6g} = {step / traces.std(axis=1).mean() * 100:.2f}% of a trace's std")

    n_keys = len(np.unique(keys[:, args.target_byte]))
    print(f"  key[{args.target_byte}] distinct values: {n_keys}{'  (fixed)' if n_keys == 1 else '  (variable)'}")

    dtype = np.dtype([("plaintext", "u1", (16,)), ("key", "u1", (16,)),
                      ("masks", "u1", (16,)), ("desync", "<u4", (1,))])

    with h5py.File(out_path, "w") as f:
        for group, off, count in (("Profiling_traces", 0, args.n_profiling),
                                   ("Attack_traces", args.n_profiling, args.n_attack)):
            g = f.create_group(group)
            g.create_dataset("traces", data=quantised[off:off + count])
            md = np.zeros(count, dtype=dtype)
            md["plaintext"] = plaintexts[off:off + count]
            md["key"] = keys[off:off + count]
            md["masks"] = 0        # unprotected implementation: no masks exist
            md["desync"] = 0
            g.create_dataset("metadata", data=md)
            pt = plaintexts[off:off + count, args.target_byte]
            k = keys[off:off + count, args.target_byte]
            g.create_dataset("labels", data=AES_SBOX[pt ^ k].astype(np.int64))
            print(f"  {group}: {count} traces")

    print(f"=== wrote {out_path} ===")
    print(f"  next: python scripts/00_inspect_data.py --h5 {out_path} --target-byte {args.target_byte} "
          f"--n-attacker {int(args.n_profiling * 0.8)} --n-val {int(args.n_profiling * 0.1)} "
          f"--n-defender {int(args.n_profiling * 0.1)}")


if __name__ == "__main__":
    main()
