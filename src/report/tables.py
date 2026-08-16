"""Cross-experiment comparison tables. Consumed by scripts/04_make_report.py,
which scans runs/*/metrics.json (see CLAUDE.md §8.3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _format_cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_format_cell(row.get(col)) for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def latex_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        f"\\begin{{tabular}}{{{'l' * len(columns)}}}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_format_cell(row.get(col)) for col in columns) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def collect_metrics(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Scan `runs/*/metrics.json` and return one row per run, exp_id-tagged.

    Each row also gets `run_dir` (the run's directory name) and, when present,
    `ge_final`/`sr1_final` (last point of the ge/sr1 curves) — the raw curves
    themselves aren't useful as flat table columns, but their endpoints are.
    """
    rows = []
    for run_dir in sorted(Path(runs_dir).iterdir()):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            row = json.load(f)
        row["run_dir"] = run_dir.name
        if row.get("ge"):
            row["ge_final"] = row["ge"][-1]
        if row.get("sr1"):
            row["sr1_final"] = row["sr1"][-1]
        rows.append(row)
    return rows
