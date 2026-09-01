#!/usr/bin/env python3
"""Extract an ASCAD-format database for an arbitrary target byte / window.

ASCAD_generate.py (vendored upstream, treated as read-only) hardcodes its
labelisation to byte 2, and every published database is byte 2's [45400, 46100)
window. To evaluate any other byte we need the same file layout around a
different window — that is all this does.

Output matches the upstream extracted files exactly:

    Profiling_traces/{traces,labels,metadata}
    Attack_traces/{traces,labels,metadata}

with metadata dtype plaintext/ciphertext/key/masks/desync — i.e. the raw
metadata with a desync field appended (zero here; this script does not
desynchronise, the desync databases are downloaded ready-made).

`labels` is written for the requested byte so the file is self-describing and
usable by upstream tooling, but note this project's own pipeline never reads it
— src/data/labels.py derives labels from metadata so that the leakage model
(ID / ID_MASKED / HW) stays a config choice rather than being baked into the
database.

Usage (run from repo root):
    python scripts/07_extract_window.py \
        --raw ATMega8515_raw_traces.h5 --out data/ASCAD_byte3_dual.h5 \
        --target-byte 3 --windows 79352:80052 98513:99213
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
    p = argparse.ArgumentParser(description="Extract an ASCAD-format database for a given byte/window")
    p.add_argument("--raw", required=True, help="path to ATMega8515_raw_traces.h5")
    p.add_argument("--out", required=True, help="output .h5 path")
    p.add_argument("--target-byte", type=int, required=True, help="byte the window targets (for labels)")
    p.add_argument("--windows", nargs="+", required=True, metavar="START:STOP",
                    help="one or more raw-point ranges to extract and concatenate, e.g. "
                         "79352:80052 98513:99213. Multiple windows are needed for most bytes: "
                         "an ID (mask-unknown) attack has to combine the masked value with the "
                         "mask itself, and outside byte 2 those two leak far apart (byte 3: 79702 "
                         "vs 98863, 19161 points). See CLAUDE.md 附錄 B.66.")
    p.add_argument("--n-profiling", type=int, default=50000,
                    help="traces [0, n) become the profiling set, matching upstream's split")
    p.add_argument("--n-attack", type=int, default=10000,
                    help="traces [n_profiling, n_profiling+n) become the attack set")
    p.add_argument("--chunk", type=int, default=5000, help="traces per read/write chunk")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    b = args.target_byte
    windows = []
    for w in args.windows:
        lo, _, hi = w.partition(":")
        lo, hi = int(lo), int(hi)
        if hi <= lo:
            raise ValueError(f"empty window {w}")
        windows.append((lo, hi))
    windows.sort()
    width = sum(hi - lo for lo, hi in windows)

    out_path = Path(args.out)
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists — refusing to overwrite an existing database")

    with h5py.File(args.raw, "r") as fin:
        n_total, n_points = fin["traces"].shape
        for lo, hi in windows:
            if hi > n_points:
                raise ValueError(f"window [{lo}, {hi}) exceeds the raw trace length {n_points}")
        need = args.n_profiling + args.n_attack
        if need > n_total:
            raise ValueError(f"need {need} traces but the raw file has {n_total}")

        raw_meta_dtype = fin["metadata"].dtype
        out_dtype = np.dtype(raw_meta_dtype.descr + [("desync", "<u4", (1,))])

        print(f"=== {args.raw} -> {out_path} ===")
        print(f"  byte {b}, windows {[f'[{lo},{hi})' for lo, hi in windows]} = {width} points total")
        print(f"  profiling: {args.n_profiling}   attack: {args.n_attack}")

        with h5py.File(out_path, "w") as fout:
            for group, offset, count in (
                ("Profiling_traces", 0, args.n_profiling),
                ("Attack_traces", args.n_profiling, args.n_attack),
            ):
                g = fout.create_group(group)
                traces = g.create_dataset("traces", shape=(count, width), dtype=np.int8)

                md = np.array(fin["metadata"][offset:offset + count])
                out_md = np.zeros(count, dtype=out_dtype)
                for name in raw_meta_dtype.names:
                    out_md[name] = md[name]
                out_md["desync"] = 0
                g.create_dataset("metadata", data=out_md)

                pt = md["plaintext"][:, b].astype(np.uint8)
                key = md["key"][:, b].astype(np.uint8)
                g.create_dataset("labels", data=AES_SBOX[pt ^ key].astype(np.int64))

                for lo in range(0, count, args.chunk):
                    hi = min(lo + args.chunk, count)
                    parts = [fin["traces"][offset + lo:offset + hi, ws:we] for ws, we in windows]
                    traces[lo:hi] = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
                    print(f"    {group}: {hi}/{count}", end="\r")
                print(f"    {group}: {count}/{count} done      ")

                keys_seen = len(np.unique(md["key"][:, b]))
                print(f"      key[{b}] distinct values: {keys_seen}"
                      f"{'  (fixed)' if keys_seen == 1 else '  (variable)'}")

    print(f"=== wrote {out_path} ===")
    print(f"  next: add a configs/data/*.yaml with trace_len {width} and target_byte {b},")
    print(f"        then run scripts/00_inspect_data.py --h5 {out_path} --target-byte {b}")


if __name__ == "__main__":
    main()
