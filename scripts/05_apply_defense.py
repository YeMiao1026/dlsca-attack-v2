#!/usr/bin/env python3
"""Apply a defense to the raw E-set traces and measure its cost (PSR/L2/Linf).
See CLAUDE.md §1.3 — defense training itself is dlsca-defense-v2's job; this
script only produces a *defended trace array* + its cost, so any already-
trained attacker model (via 02_run_attack.py --traces) can be pointed at it
for a static A0 evaluation. First defense wired in: additive Gaussian noise,
reusing the exact same pure function E02's training-time augmentation uses
(src/data/preprocess.py::gaussian_augment) — here applied once to the raw
E-set as a deployed countermeasure, not per-epoch to A as training regularization.

Usage (run from repo root):
    python scripts/05_apply_defense.py --run runs/E01_baseline_clean_20260816_1302 \
        --defense gaussian --sigma-ratio 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

from src.data.ascad import load as ascad_load
from src.data.preprocess import gaussian_augment
from src.data.split import load as load_split
from src.metrics.perturbation import l2, linf, psr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply a defense to E and measure its cost")
    p.add_argument("--run", required=True, help="attacker run dir, used for data.path + split_indices.npz")
    p.add_argument("--defense", required=True, choices=["gaussian"], help="defense method")
    p.add_argument("--sigma-ratio", type=float, default=0.5,
                    help="gaussian defense: noise std as a multiple of each raw point's std across E")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="defenses", help="parent directory for defense outputs")
    return p.parse_args()


def summarize(name: str, values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)

    with open(run_dir / "config_snapshot.yaml") as f:
        cfg = yaml.safe_load(f)

    print(f"=== loading {cfg['data']['path']} ===")
    data = ascad_load(cfg["data"]["path"])
    split = load_split(str(run_dir / "split_indices.npz"))
    clean_e = data.attack_traces[split.e].astype(np.float32)

    if args.defense == "gaussian":
        label = f"gaussian_sigma{args.sigma_ratio:g}"
        print(f"=== applying gaussian defense (sigma_ratio={args.sigma_ratio}) to {len(clean_e)} traces ===")
        defended_e = gaussian_augment(clean_e, sigma_ratio=args.sigma_ratio, seed=args.seed)
    else:
        raise ValueError(f"unknown defense: {args.defense!r}")

    print("=== computing cost metrics (PSR / L2 / Linf) vs. clean E ===")
    cost = {
        "defense": args.defense,
        "params": {"sigma_ratio": args.sigma_ratio, "seed": args.seed} if args.defense == "gaussian" else {},
        "source_run": str(run_dir),
        "n_traces": int(len(clean_e)),
        "psr": summarize("psr", psr(clean_e, defended_e)),
        "l2": summarize("l2", l2(clean_e, defended_e)),
        "linf": summarize("linf", linf(clean_e, defended_e)),
    }
    print(f"  PSR  mean={cost['psr']['mean']:.4f}  p90={cost['psr']['p90']:.4f}")
    print(f"  L2   mean={cost['l2']['mean']:.4f}  p90={cost['l2']['p90']:.4f}")
    print(f"  Linf mean={cost['linf']['mean']:.4f}  p90={cost['linf']['p90']:.4f}")

    out_dir = Path(args.out_dir) / f"{label}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "defended_traces.npy", defended_e)
    (out_dir / "cost_metrics.json").write_text(json.dumps(cost, indent=2))
    print(f"=== saved {out_dir}/defended_traces.npy + cost_metrics.json ===")
    print()
    print("=== next steps ===")
    print(f"  python3 scripts/02_run_attack.py --run {run_dir} "
          f"--traces {out_dir}/defended_traces.npy --out {out_dir}/probs.npy")
    print(f"  python3 scripts/03_evaluate.py --run {run_dir} "
          f"--probs {out_dir}/probs.npy --out {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
