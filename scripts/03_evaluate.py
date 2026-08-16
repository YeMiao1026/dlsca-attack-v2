#!/usr/bin/env python3
"""Key recovery + evaluation metrics from a saved probs.npy. See CLAUDE.md
§5.2 stage 5 — this stage never re-runs the model; it only reads probs.npy
and metadata (P5: evaluation and training are fully decoupled).

Usage (run from repo root):
    python scripts/03_evaluate.py --run runs/E01_baseline_clean_20260815_2215
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

from src.attack import keyrank, scores
from src.config import apply_overrides
from src.data.ascad import get_correct_key
from src.data.ascad import load as ascad_load
from src.data.labels import build as build_labels
from src.data.split import load as load_split
from src.metrics.information import pi as compute_pi

METRIC_FUNCS = {"ge", "sr1", "n_tge", "n_sr90", "pi", "percentiles"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved probs.npy (CLAUDE.md §5.2 stage 5)")
    p.add_argument("--run", required=True, help="run directory containing probs.npy")
    p.add_argument("--override", nargs="*", default=[],
                    help="dotted.key=value overrides on top of config_snapshot.yaml, e.g. attack.max_traces=9000 "
                         "— applied in-memory only, does not touch the archived snapshot or require retraining/"
                         "re-running 02_run_attack.py (probs.npy doesn't depend on evaluation-stage params)")
    p.add_argument("--probs", default=None,
                    help="path to an alternative probs.npy (e.g. from a defended-waveform run via "
                         "02_run_attack.py --traces/--out). Requires --out.")
    p.add_argument("--out", default=None,
                    help="where to write metrics.json (default: {run}/metrics.json; required when --probs is set)")
    args = p.parse_args()
    if args.probs and not args.out:
        p.error("--probs requires --out (never silently overwrite the run's own metrics.json)")
    return args


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)

    with open(run_dir / "config_snapshot.yaml") as f:
        cfg = yaml.safe_load(f)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    probs_path = Path(args.probs) if args.probs else run_dir / "probs.npy"
    probs = np.load(probs_path)
    print(f"=== loaded {probs_path}: shape={probs.shape} dtype={probs.dtype} ===")

    target_byte = cfg["data"]["target_byte"]
    print(f"=== loading {cfg['data']['path']} for E metadata ===")
    data = ascad_load(cfg["data"]["path"])
    split = load_split(str(run_dir / "split_indices.npz"))
    meta_e = data.attack_meta[split.e]

    correct_key = get_correct_key(meta_e, target_byte)
    print(f"  correct key: {correct_key} (0x{correct_key:02x})")

    leakage_model = cfg["leakage"]["model"]
    mask_index = cfg["leakage"].get("mask_index")
    mask = meta_e["masks"][:, mask_index].astype(np.uint8) if leakage_model == "ID_MASKED" else None

    print("=== building log-likelihood score matrix ===")
    sc = scores.build(probs, meta_e["plaintext"], target_byte, leakage_model=leakage_model, mask=mask)

    attack_cfg = cfg["attack"]
    n_runs = attack_cfg["n_runs"]
    max_traces = attack_cfg["max_traces"]
    print(f"=== running {n_runs} independent attacks, {max_traces} traces each ===")
    ranks = keyrank.evaluate(sc, correct_key, n_runs=n_runs, max_traces=max_traces, seed=attack_cfg["seed"])

    requested = set(cfg.get("metrics", {}).get("compute", [])) & METRIC_FUNCS
    skipped = set(cfg.get("metrics", {}).get("compute", [])) - METRIC_FUNCS
    if skipped:
        print(f"  (skipping unimplemented/unrecognized metrics: {sorted(skipped)})")

    metrics: dict = {
        "exp_id": cfg["exp_id"],
        "correct_key": correct_key,
        "n_traces_eval": max_traces,
        "n_runs": n_runs,
    }

    ge_curve = keyrank.ge(ranks) if "ge" in requested else None
    sr1_curve = keyrank.sr1(ranks) if "sr1" in requested else None
    if ge_curve is not None:
        metrics["ge"] = ge_curve.tolist()
    if sr1_curve is not None:
        metrics["sr1"] = sr1_curve.tolist()
    if "n_tge" in requested:
        if ge_curve is None:
            ge_curve = keyrank.ge(ranks)
        metrics["n_tge"] = keyrank.n_tge(ge_curve)
    if "n_sr90" in requested:
        if sr1_curve is None:
            sr1_curve = keyrank.sr1(ranks)
        metrics["n_sr90"] = keyrank.n_sr90(sr1_curve)
    if "percentiles" in requested:
        p25, p50, p75 = keyrank.percentiles(ranks, q=(25, 50, 75))
        metrics["percentiles"] = {"p25": p25.tolist(), "p50": p50.tolist(), "p75": p75.tolist()}
    if "pi" in requested:
        y_e = build_labels(meta_e, leakage_model, target_byte, mask_index=mask_index)
        metrics["pi"] = compute_pi(probs, y_e, cfg["leakage"]["n_classes"])

    out_path = Path(args.out) if args.out else run_dir / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"=== saved {out_path} ===")

    print()
    print("=== summary ===")
    if "n_tge" in metrics:
        print(f"  N_TGE:  {metrics['n_tge']}")
    if "n_sr90" in metrics:
        print(f"  N_SR90: {metrics['n_sr90']}")
    if "pi" in metrics:
        print(f"  PI:     {metrics['pi']:.4f} bits")
    if ge_curve is not None:
        print(f"  GE @ N={max_traces}: {ge_curve[-1]:.4f}")
    if sr1_curve is not None:
        print(f"  SR1 @ N={max_traces}: {sr1_curve[-1]:.4f}")


if __name__ == "__main__":
    main()
