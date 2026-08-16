#!/usr/bin/env python3
"""Run the trained attacker model on E, save probs.npy. See CLAUDE.md §5.2
stage 4 — this is the only interface between the attack and evaluation
stages; once probs.npy is written, model.keras is no longer needed to
evaluate. Re-running this stage against a *different* set of traces (e.g. a
defended waveform) with the same model is exactly how Stage B's static A0
attacker is meant to be evaluated later.

Usage (run from repo root):
    python scripts/02_run_attack.py --run runs/E01_baseline_clean_20260815_2215
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keras
import numpy as np
import yaml

from src.attack.predict import run as predict_run
from src.data.ascad import load as ascad_load
from src.data.preprocess import MinMaxScaler, Standardizer
from src.data.split import load as load_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the attacker model on E (CLAUDE.md §5.2 stage 4)")
    p.add_argument("--run", required=True, help="run directory produced by 01_train_attacker.py")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)

    with open(run_dir / "config_snapshot.yaml") as f:
        cfg = yaml.safe_load(f)

    print(f"=== loading {cfg['data']['path']} ===")
    data = ascad_load(cfg["data"]["path"])
    split = load_split(str(run_dir / "split_indices.npz"))

    # Re-fit Standardizer (+ MinMaxScaler, if enabled) on A exactly as
    # 01_train_attacker.py did — both are deterministic (no randomness in
    # fit()), so this exactly reproduces the training-time preprocessing
    # without needing to persist it separately.
    preprocess_method = cfg["preprocess"].get("method", "standardize_per_point")
    if preprocess_method == "none":
        print("=== preprocess.method=none: using raw traces (cast to float32) ===")
        x_e = data.attack_traces[split.e].astype(np.float32)
    elif preprocess_method == "standardize_per_point":
        print("=== re-fitting Standardizer on A ===")
        standardizer = Standardizer()
        x_a = standardizer.fit_transform(data.profiling_traces[split.a])

        print(f"=== transforming E ({len(split.e)} traces) ===")
        x_e = standardizer.transform(data.attack_traces[split.e])

        if cfg["preprocess"].get("minmax"):
            print("=== re-fitting MinMaxScaler on standardized A ===")
            minmax = MinMaxScaler()
            minmax.fit(x_a)
            x_e = minmax.transform(x_e)
    else:
        raise ValueError(f"unknown preprocess.method: {preprocess_method!r}")

    print(f"=== loading {run_dir / 'model.keras'} ===")
    model = keras.models.load_model(run_dir / "model.keras")

    print("=== running inference ===")
    probs = predict_run(model, x_e)

    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        bad = int((~np.isclose(row_sums, 1.0, atol=1e-3)).sum())
        print(f"FAIL: {bad}/{len(row_sums)} rows do not sum to 1 (min={row_sums.min():.6f}, max={row_sums.max():.6f})")
        sys.exit(1)
    if not np.all(np.isfinite(probs)):
        print("FAIL: probs contains NaN or Inf")
        sys.exit(1)
    print(f"  probs shape={probs.shape} dtype={probs.dtype} row-sum range=[{row_sums.min():.6f}, {row_sums.max():.6f}]")

    out_path = run_dir / "probs.npy"
    np.save(out_path, probs)
    print(f"=== saved {out_path} ===")


if __name__ == "__main__":
    main()
