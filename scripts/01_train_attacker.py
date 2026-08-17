#!/usr/bin/env python3
"""Train an attacker model on A with GE-based model selection on V.
See CLAUDE.md §5.2 stage 3, §6, §6.5 for the produced artifacts.

Usage (run from repo root):
    python scripts/01_train_attacker.py --config configs/exp/E01_baseline_clean.yaml
    python scripts/01_train_attacker.py --config configs/exp/E01_baseline_clean.yaml \
        --override train.epochs=5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf

import src.models  # noqa: F401  registers cnn_light/cnn_best/resnet with src.models.registry
from src.config import load_config, snapshot
from src.data.ascad import load as ascad_load
from src.data.labels import build as build_labels
from src.data.preprocess import MinMaxScaler, Standardizer
from src.data.resync import resync, resync_iterative
from src.data.split import four_way
from src.data.split import save as save_split
from src.models.registry import build as build_model
from src.seeding import set_global_seed
from src.train.trainer import fit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an attacker model (CLAUDE.md §5.2 stage 3)")
    p.add_argument("--config", required=True, help="path to an exp config, e.g. configs/exp/E01_baseline_clean.yaml")
    p.add_argument("--override", nargs="*", default=[], help="dotted.key=value overrides, e.g. train.epochs=5")
    p.add_argument("--runs-dir", default="runs", help="parent directory for run outputs")
    p.add_argument("--profiling-traces", default=None,
                    help="path to a .npy of alternative Profiling_traces (same shape as "
                         "data.profiling_traces, e.g. dlsca-defense-v2's defended_profiling_traces.npy) "
                         "to train on instead of the clean ASCAD Profiling_traces -- this is how Stage "
                         "B's adaptive A1 attacker (CLAUDE.md G4) retrains against a deployed defense. "
                         "Only the trace waveform is substituted; plaintext/key/masks metadata still "
                         "comes from the .h5.")
    return p.parse_args()


def git_commit_hash() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def write_env_json(path: Path) -> None:
    env = {
        "python": sys.version,
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "gpus": [g.name for g in tf.config.list_physical_devices("GPU")],
        "git_commit": git_commit_hash(),
    }
    path.write_text(json.dumps(env, indent=2))


def write_train_history_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)
    set_global_seed(cfg["seed"])

    # PID suffix: even second-precision timestamps collided when two runs were
    # launched within the same second, racing on the same run_dir/model.keras
    # (see CLAUDE.md 附錄 B.35, B.37) — os.getpid() is unique per process
    # regardless of launch timing.
    run_dir = Path(args.runs_dir) / f"{cfg['exp_id']}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== run dir: {run_dir} ===")

    snapshot(cfg, run_dir)
    write_env_json(run_dir / "env.json")

    print(f"=== loading {cfg['data']['path']} ===")
    data = ascad_load(cfg["data"]["path"])
    target_byte = cfg["data"]["target_byte"]

    if args.profiling_traces:
        print(f"=== loading alternative profiling traces from {args.profiling_traces} (adaptive A1 retraining) ===")
        alt_traces = np.load(args.profiling_traces)
        if alt_traces.shape != data.profiling_traces.shape:
            raise ValueError(f"--profiling-traces shape {alt_traces.shape} != expected {data.profiling_traces.shape}")
        data = data._replace(profiling_traces=alt_traces)

    split = four_way(
        n_profiling=len(data.profiling_traces),
        n_attack=len(data.attack_traces),
        n_attacker=cfg["split"]["n_attacker"],
        n_val=cfg["split"]["n_val"],
        n_defender=cfg["split"]["n_defender"],
        seed=cfg["split"]["seed"],
    )
    save_split(split, str(run_dir / "split_indices.npz"))

    traces_a = data.profiling_traces[split.a]
    traces_v = data.profiling_traces[split.v]
    meta_a = data.profiling_meta[split.a]
    meta_v = data.profiling_meta[split.v]

    resync_cfg = cfg["preprocess"].get("resync", {})
    if resync_cfg.get("enabled"):
        max_shift = resync_cfg.get("max_shift", 50)
        rounds = resync_cfg.get("rounds", 2)
        print(f"=== resyncing A (max_shift={max_shift}, rounds={rounds}) ===")
        traces_a, shifts_a, reference = resync_iterative(traces_a, max_shift=max_shift, rounds=rounds)
        print(f"  A shift stats: min={shifts_a.min()} max={shifts_a.max()} std={shifts_a.std():.2f}")
        print("=== resyncing V against A's reference ===")
        traces_v, _ = resync(traces_v, reference, max_shift=max_shift)

    preprocess_method = cfg["preprocess"].get("method", "standardize_per_point")
    if preprocess_method == "none":
        # matches train_with_pure/train_cnnd.py: raw int8 traces cast straight
        # to float32, no scaling at all (see CLAUDE.md 附錄 B.11)
        print("=== preprocess.method=none: using raw traces (cast to float32) ===")
        x_a = traces_a.astype(np.float32)
        x_v = traces_v.astype(np.float32)
    elif preprocess_method == "standardize_per_point":
        print("=== fitting Standardizer on A, transforming A/V ===")
        standardizer = Standardizer()
        x_a = standardizer.fit_transform(traces_a)
        x_v = standardizer.transform(traces_v)

        if cfg["preprocess"].get("minmax"):
            print("=== fitting MinMaxScaler on standardized A, transforming A/V ===")
            minmax = MinMaxScaler()
            x_a = minmax.fit_transform(x_a)
            x_v = minmax.transform(x_v)
    else:
        raise ValueError(f"unknown preprocess.method: {preprocess_method!r}")

    leakage_model = cfg["leakage"]["model"]
    mask_index = cfg["leakage"].get("mask_index")
    y_a = build_labels(meta_a, leakage_model, target_byte, mask_index=mask_index)
    y_v = build_labels(meta_v, leakage_model, target_byte, mask_index=mask_index)

    print(f"=== building model {cfg['model']['name']!r} ===")
    model = build_model(cfg["model"]["name"], input_dim=cfg["data"]["trace_len"],
                         n_classes=cfg["leakage"]["n_classes"])
    model.summary()

    checkpoint_path = str(run_dir / "model.keras")
    print("=== training ===")
    best_model, history_rows = fit(x_a, y_a, x_v, y_v, meta_v, model, cfg, checkpoint_path=checkpoint_path)

    write_train_history_csv(history_rows, run_dir / "train_history.csv")
    print(f"=== done: {run_dir} ===")


if __name__ == "__main__":
    main()
