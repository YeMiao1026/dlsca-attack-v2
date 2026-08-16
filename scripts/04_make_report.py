#!/usr/bin/env python3
"""Scan runs/*/metrics.json, (re)generate per-run GE/SR figures, and write a
cross-experiment comparison table. See CLAUDE.md §8.3.

Usage (run from repo root):
    python scripts/04_make_report.py
    python scripts/04_make_report.py --runs-dir runs --out-dir reports
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.report.plots import plot_ge_curve, plot_sr_curve
from src.report.tables import collect_metrics, latex_table, markdown_table

DEFAULT_COLUMNS = ["run_dir", "exp_id", "n_tge", "n_sr90", "ge_final", "sr1_final", "pi"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cross-experiment report (CLAUDE.md §8.3)")
    p.add_argument("--runs-dir", default="runs", help="parent directory containing {exp_id}_{timestamp}/ runs")
    p.add_argument("--out-dir", default="reports", help="where to write comparison.md / comparison.tex")
    p.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS, help="metrics.json keys to include as columns")
    return p.parse_args()


def make_figures(runs_dir: Path, row: dict) -> None:
    fig_dir = runs_dir / row["run_dir"] / "figures"
    if row.get("ge"):
        ge = np.array(row["ge"])
        band = None
        if row.get("percentiles"):
            band = (np.array(row["percentiles"]["p25"]), np.array(row["percentiles"]["p75"]))
        plot_ge_curve(ge, band, fig_dir / "ge_curve.png")
    if row.get("sr1"):
        plot_sr_curve(np.array(row["sr1"]), fig_dir / "sr_curve.png")


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)

    rows = collect_metrics(runs_dir)
    if not rows:
        print(f"no metrics.json found under {runs_dir}/ — run 03_evaluate.py on a run first")
        return

    print(f"=== found {len(rows)} run(s) with metrics.json ===")
    for row in rows:
        make_figures(runs_dir, row)
        print(f"  {row['run_dir']}: figures written")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = markdown_table(rows, args.columns)
    tex = latex_table(rows, args.columns)
    (out_dir / "comparison.md").write_text(md + "\n")
    (out_dir / "comparison.tex").write_text(tex + "\n")

    print()
    print(md)
    print()
    print(f"=== wrote {out_dir / 'comparison.md'} and {out_dir / 'comparison.tex'} ===")


if __name__ == "__main__":
    main()
