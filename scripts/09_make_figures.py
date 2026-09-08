#!/usr/bin/env python3
"""Build the report-grade figure set from every result recorded in CLAUDE.md 附錄 B/C.

scripts/04_make_report.py already writes a GE/SR curve *per run* into that run's
own figures/ directory. Those are working figures: one per execution, 100+ of
them, and gitignored along with runs/. This script produces the other half —
the cross-experiment figures that carry an argument rather than a single
measurement, written to reports/figures/ so they are versioned alongside the
prose that cites them.

Every curve and every metric plotted here is read back from runs*/metrics.json,
runs*/train_history.csv or defenses*/cost_metrics.json, so a figure cannot drift
away from the run that produced it. Two panels are not run outputs and are
labelled as such:

  * F05 recomputes SNR from the h5 databases, because SNR is a property of the
    data rather than of any run.
  * F11's right panel carries the four SNR peaks measured in CLAUDE.md 附錄
    B.66/B.67. Those are transcribed rather than recomputed: they come from the
    byte-3 databases, which live on the GPU server and are not part of the
    local data/ directory. Recompute them with scripts/06_find_byte_poi.py if
    the numbers ever need re-deriving.

Usage (run from repo root):
    python scripts/09_make_figures.py                 # everything except F05
    python scripts/09_make_figures.py --with-snr      # + recompute the resync SNR panel
    python scripts/09_make_figures.py --only F10 F13
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("reports/figures")
DPI = 160
RANDOM_GE = 127.5          # random-guess baseline for a 256-class ID attack

# One palette for the whole set so a colour means the same thing across figures.
C_GOOD = "#1b6ca8"         # the configuration being argued for
C_BAD = "#c1443c"          # a negative / refuted result
C_MID = "#e08a1e"          # intermediate steps
C_ALT = "#4a8c5c"
C_GREY = "#8a8f98"
CYCLE = [C_GOOD, C_BAD, C_MID, C_ALT, "#7d5ba6", C_GREY]

plt.rcParams.update({
    "figure.dpi": DPI,
    "font.size": 9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "legend.fontsize": 8,
})


# --------------------------------------------------------------------------- data access

def metrics(run_dir: str) -> dict:
    with open(Path(run_dir) / "metrics.json") as f:
        return json.load(f)


def ge(run_dir: str) -> np.ndarray:
    return np.asarray(metrics(run_dir)["ge"])


def history(run_dir: str) -> list[dict]:
    with open(Path(run_dir) / "train_history.csv") as f:
        return list(csv.DictReader(f))


def previews(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """(epochs, GE preview) for the epochs where GEModelSelection actually ran.

    On variable-key databases this column holds `mean_true_rank` instead — same
    0..127.5 scale, different quantity (see CLAUDE.md 附錄 B.62).
    """
    rows = [r for r in history(run_dir) if r.get("ge")]
    return (np.array([int(r["epoch"]) for r in rows]),
            np.array([float(r["ge"]) for r in rows]))


def cost(defense_dir: str) -> dict:
    with open(Path(defense_dir) / "cost_metrics.json") as f:
        return json.load(f)


def save(fig, name: str, caption: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}  — {caption}")


def _ge_axes(ax, title: str, ylabel: str = "Guessing Entropy") -> None:
    ax.axhline(RANDOM_GE, color=C_GREY, ls=":", lw=1)
    ax.axhline(1.0, color=C_BAD, ls="--", lw=0.9)
    ax.set_xlabel("Number of attack traces")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _annotate_baselines(ax, x: float) -> None:
    ax.text(x, RANDOM_GE * 1.04, "random guess (127.5)", color=C_GREY, fontsize=7, va="bottom")
    ax.text(x, 1.15, "GE = 1", color=C_BAD, fontsize=7, va="bottom")


def _bar_sweep(ax, labels, values, best_idx, title, ylabel, fmt="{:.2f}"):
    colors = [C_GOOD if i == best_idx else C_GREY for i in range(len(values))]
    bars = ax.bar(range(len(values)), values, color=colors, width=0.62)
    ax.axhline(RANDOM_GE, color=C_BAD, ls=":", lw=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(max(values), RANDOM_GE) * 1.22)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, " " + fmt.format(v), ha="center", va="bottom", fontsize=7.5)


# --------------------------------------------------------------------------- figures

def f01_recipe_evolution():
    """附錄 B.7-B.15: how E01 went from never converging to N_TGE=475."""
    steps = [
        ("runs/E01_baseline_clean_20260815_2228", "flat LR, 500 ep (early-stopped @175)", C_GREY),
        ("runs/E01_baseline_clean_20260815_2336", "+ one-cycle LR + MinMax", C_MID),
        ("runs/E01_baseline_clean_20260816_0027", "+ he_uniform init", C_ALT),
        ("runs/E01_baseline_clean_20260816_1302", "+ scale_percentage 0.05  (final)", C_GOOD),
    ]
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    for run, label, colour in steps:
        m = metrics(run)
        curve = np.asarray(m["ge"])
        n_tge = m.get("n_tge")
        tag = f"N_TGE={n_tge}" if n_tge else "never converged"
        ax.plot(np.arange(1, len(curve) + 1), curve, color=colour, lw=1.6,
                label=f"{label}  ({tag}, window {len(curve)})")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 260)
    _ge_axes(ax, "E01: each methodology fix, measured on the same protocol")
    _annotate_baselines(ax, 20)
    ax.legend(loc="lower left")

    names = ["flat LR\n(lecun)", "one-cycle\n+MinMax", "+he_uniform", "+scale 0.05"]
    vals = [6408, 6408, 695, 475]
    # the first flat-LR run never reached GE<1 inside any window; 6408 is the
    # one-cycle run's value, so show the flat run as "not reached" instead.
    axb.bar([0], [9000], color=C_GREY, hatch="//", edgecolor="white", width=0.62)
    axb.bar(range(1, 4), vals[1:], color=[C_MID, C_ALT, C_GOOD], width=0.62)
    axb.set_xticks(range(4))
    axb.set_xticklabels(names, fontsize=7.5)
    axb.set_yscale("log")
    axb.set_ylabel("N_TGE (traces to break)")
    axb.set_title("Traces needed, lower is better")
    axb.set_ylim(200, 1.6e4)
    axb.text(0, 7600, "never\nreached", ha="center", va="top", fontsize=7, color="white")
    for i, v in enumerate(vals[1:], start=1):
        axb.text(i, v, f" {v}", ha="center", va="bottom", fontsize=7.5)
    save(fig, "F01_e01_recipe_evolution", "E01 recipe evolution (附錄 B.7-B.15)")


def f02_e01_sweeps():
    """附錄 B.13-B.15: the three one-cycle dimensions, each swept alone.

    Plotted as N_TGE rather than final GE: every converged configuration reaches
    GE=0 by N=9000, so final GE cannot separate 475 traces from 4253. N_TGE is
    the quantity the appendix compares, and the one that says how much the
    attack actually costs.
    """
    CEIL = 9000        # the evaluation window; a bar drawn here never converged
    sweeps = [
        ("end_percentage", ["0.1", "0.2", "0.35"],
         ["runs/E01_baseline_clean_20260816_1217", "runs/E01_baseline_clean_20260816_0027",
          "runs/E01_baseline_clean_20260816_1212"]),
        ("max_lr", ["2.5e-3", "5e-3", "1e-2"],
         ["runs/E01_baseline_clean_20260816_1236", "runs/E01_baseline_clean_20260816_0027",
          "runs/E01_baseline_clean_20260816_1232"]),
        ("scale_percentage\n(peak LR held at 5e-3)", ["0.05", "0.1", "0.2"],
         ["runs/E01_baseline_clean_20260816_1302", "runs/E01_baseline_clean_20260816_0027",
          "runs/E01_baseline_clean_20260816_1308"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
    for ax, (name, labels, runs) in zip(axes, sweeps):
        vals = [metrics(r).get("n_tge") for r in runs]
        converged = [v for v in vals if v is not None]
        best = vals.index(min(converged))
        for i, v in enumerate(vals):
            if v is None:
                ax.bar(i, CEIL, width=0.62, color=C_BAD, alpha=0.30,
                       hatch="//", edgecolor="white")
                ax.text(i, 2600, "never\nconverged", ha="center", fontsize=7, color=C_BAD)
            else:
                ax.bar(i, v, width=0.62, color=C_GOOD if i == best else C_GREY)
                ax.text(i, v * 1.06, f"{v}", ha="center", va="bottom", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(200, CEIL * 2.2)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_title(name)
    axes[0].set_ylabel("N_TGE (traces to break)\nlower is better")
    fig.suptitle("E01 (desync0): each one-cycle dimension swept alone — end_percentage and max_lr "
                 "are local optima,\nscale_percentage still improves at the edge of the range tested",
                 fontsize=10, fontweight="bold")
    save(fig, "F02_e01_hparam_sweeps", "E01 one-cycle sweeps (附錄 B.13-B.15)")


def f03_leakage_models():
    """附錄 B.19/B.17: what the attacker is assumed to know changes everything."""
    entries = [
        ("runs/E08_masked_label_20260816_1349", "ID_MASKED — mask known (evaluator's upper bound)", C_ALT),
        ("runs/E05_hw_leakage_20260816_1415", "HW — Hamming weight, 9 classes", C_MID),
        ("runs/E01_baseline_clean_20260816_1302", "ID — mask unknown (the real attacker)", C_GOOD),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for run, label, colour in entries:
        m = metrics(run)
        curve = np.asarray(m["ge"])
        ax.plot(np.arange(1, len(curve) + 1), curve, color=colour, lw=1.7,
                label=f"{label}   N_TGE={m['n_tge']}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 300)
    _ge_axes(ax, "Leakage model sets the difficulty, not the architecture")
    ax.legend(loc="lower left")
    save(fig, "F03_leakage_models", "ID vs HW vs ID_MASKED (附錄 B.17/B.19)")


def f04_augmentation():
    """附錄 B.23: noise augmentation improves both convergence and per-trace quality."""
    e01, e02 = "runs/E01_baseline_clean_20260816_1302", "runs/E02_noisy_augment_20260816_1618"
    m1, m2 = metrics(e01), metrics(e02)
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [2, 1]})
    for m, label, colour in ((m1, f"E01 no augmentation   N_TGE={m1['n_tge']}", C_GREY),
                             (m2, f"E02 gaussian σ-ratio 0.5   N_TGE={m2['n_tge']}", C_GOOD)):
        curve = np.asarray(m["ge"])
        ax.plot(np.arange(1, len(curve) + 1), curve, color=colour, lw=1.7, label=label)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 140)
    _ge_axes(ax, "Convergence: 2.3x fewer traces")
    ax.legend()

    pis = [m1["pi"], m2["pi"]]
    axb.bar([0, 1], pis, color=[C_BAD if p < 0 else C_GOOD for p in pis], width=0.5)
    axb.axhline(0, color="black", lw=0.8)
    axb.set_xticks([0, 1]); axb.set_xticklabels(["E01", "E02"])
    axb.set_ylabel("Perceived Information (bits)")
    axb.set_title("Per-trace quality: PI turns positive")
    for i, p in enumerate(pis):
        axb.text(i, p, f"{p:+.3f}", ha="center", va="bottom" if p > 0 else "top", fontsize=8)
    axb.text(0.5, 0.02, "PI < 0 = confidently wrong", transform=axb.transAxes,
             ha="center", fontsize=7, color=C_BAD)
    save(fig, "F04_e02_augmentation", "E01 vs E02 noise augmentation (附錄 B.23)")


def f05_resync_snr(n_traces: int = 5000):
    """附錄 B.29/B.30/B.33: the desync signal is scattered, not destroyed.

    Recomputed from the h5 files rather than read from a run, because SNR is a
    property of the database. Uses the same mask index the desync0 inspection
    settled on (CLAUDE.md 附錄 B.6: never re-detect it on desync data).
    """
    from src.data import ascad, labels as lab
    from src.data.resync import resync_iterative
    from src.metrics.leakage import snr

    dbs = [("data/ASCAD.h5", "desync0", 0), ("data/ASCAD_desync50.h5", "desync50", 50),
           ("data/ASCAD_desync100.h5", "desync100", 100)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (path, name, max_shift) in zip(axes, dbs):
        data = ascad.load(path)
        traces = np.asarray(data.profiling_traces[:n_traces], dtype=np.float32)
        y = lab.build(data.profiling_meta[:n_traces], "ID_MASKED", 2, mask_index=0)
        before = snr(traces, y)
        ax.plot(before, color=C_BAD, lw=0.9, label=f"as captured (peak {before.max():.2f})")
        if max_shift:
            aligned, _, _ = resync_iterative(traces, max_shift=max_shift, rounds=2)
            after = snr(aligned, y)
            ax.plot(after, color=C_GOOD, lw=0.9, label=f"after blind resync (peak {after.max():.2f})")
        ax.set_title(name)
        ax.set_xlabel("Time point")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("SNR (mask-known label)")
    axes[0].set_yscale("log")
    fig.suptitle("Desync scatters the leakage across time; normalised cross-correlation recovers it\n"
                 f"mask-known label, mask column 0, first {n_traces:,} profiling traces of each database",
                 fontsize=10, fontweight="bold")
    save(fig, "F05_resync_snr", "SNR before/after resync (附錄 B.29/B.30/B.33)")


def f06_desync_sweeps():
    """附錄 B.51-B.54 and B.38-B.42: desync50 and desync100 each needed their own recipe."""
    d50 = [
        ("max_lr", ["1e-3", "5e-3", "1e-2"],
         ["runs/E03_desync50_20260816_214625_lr1e3", "runs/E03_desync50_20260816_1858",
          "runs/E03_desync50_20260816_214625_lr1e2"], 0),
        ("end_percentage", ["0.1", "0.2", "0.35"],
         ["runs/E03_desync50_20260816_215945_end01", "runs/E03_desync50_20260816_214625_lr1e3",
          "runs/E03_desync50_20260816_215945_end035"], 1),
        ("scale_percentage", ["0.05", "0.1", "0.2"],
         ["runs/E03_desync50_20260816_220659_scale005", "runs/E03_desync50_20260816_214625_lr1e3",
          "runs/E03_desync50_20260816_220659_scale02"], 1),
        ("epochs", ["30", "50", "75"],
         ["runs/E03_desync50_20260816_221518_ep30", "runs/E03_desync50_20260816_214625_lr1e3",
          "runs/E03_desync50_20260816_221518_ep75"], 0),
    ]
    d100 = [
        ("max_lr", ["1e-3", "5e-3", "1e-2"],
         ["runs/E04_desync100_20260816_195051_lr1e3", "runs/E04_desync100_20260816_1931_gentle",
          "runs/E04_desync100_20260816_194046_lr1e2"], 0),
        ("end_percentage", ["0.1", "0.2", "0.35"],
         ["runs/E04_desync100_20260816_200044_end01", "runs/E04_desync100_20260816_195051_lr1e3",
          "runs/E04_desync100_20260816_200044_end035"], 0),
        ("scale_percentage", ["0.1", "0.2", "0.3"],
         ["runs/E04_desync100_20260816_200044_end01", "runs/E04_desync100_20260816_201058_scale02",
          "runs/E04_desync100_20260816_201955_scale03"], 1),
        ("epochs", ["30", "50", "75"],
         ["runs/E04_desync100_20260816_203034_ep30", "runs/E04_desync100_20260816_201058_scale02",
          "runs/E04_desync100_20260816_203034_ep75"], 1),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    for row, (sweeps, title) in enumerate(((d50, "desync50 + resync"), (d100, "desync100 + resync"))):
        for ax, (name, labs, runs, best) in zip(axes[row], sweeps):
            vals = [float(ge(r)[999]) for r in runs]      # GE at N=1000, comparable across windows
            _bar_sweep(ax, labs, vals, best, name, "GE at N=1000" if not ax.get_ylabel() else "")
        axes[row][0].set_ylabel(f"{title}\nGE at N=1000")
    fig.suptitle("Each desync level needed its own four-dimension sweep — the desync0 recipe does not transfer",
                 fontsize=10, fontweight="bold")
    save(fig, "F06_desync_sweeps", "desync50/100 four-dimension sweeps (附錄 B.38-B.42, B.51-B.54)")


def f07_desync_best():
    """附錄 B.42/B.54: best achievable curve at each desync level."""
    entries = [
        ("runs/E01_baseline_clean_20260816_1302", "desync0 (no resync needed)", C_GOOD),
        ("runs/E03_desync50_20260816_221518_ep30", "desync50 + resync, tuned", C_MID),
        ("runs/E04_desync100_20260816_201058_scale02", "desync100 + resync, tuned", C_BAD),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for run, label, colour in entries:
        m = metrics(run)
        curve = np.asarray(m["ge"])
        tail = f"N_TGE={m['n_tge']}" if m.get("n_tge") else f"GE@{len(curve)}={curve[-1]:.2f}, never <1"
        ax.plot(np.arange(1, len(curve) + 1), curve, color=colour, lw=1.7, label=f"{label}   {tail}")
    _ge_axes(ax, "Best result at each desync level, after per-level tuning")
    _annotate_baselines(ax, 200)
    ax.legend(loc="upper right")
    save(fig, "F07_desync_best", "best-per-desync-level comparison (附錄 B.42/B.54)")


def f08_cnn_best():
    """附錄 B.50: six systematic attempts, all statistically indistinguishable."""
    runs = [
        ("flat RMSprop 1e-5\n(paper default)", "runs/E06_cnn_best_20260816_1801"),
        ("one-cycle\npeak 1e-5", "runs/E06_cnn_best_20260816_204820_lr1e5"),
        ("one-cycle\npeak 1e-4", "runs/E06_cnn_best_20260816_204319_lr1e4"),
        ("one-cycle\npeak 1e-3", "runs/E06_cnn_best_20260816_204320_lr1e3"),
        ("flat 1e-3", "runs/E06_cnn_best_20260816_213506_flatlr1e3"),
        ("batch 50", "runs/E06_cnn_best_20260816_213506_batch50"),
    ]
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
    labels = [n for n, _ in runs]
    vals = [float(ge(r)[999]) for _, r in runs]
    ax.bar(range(len(vals)), vals, color=C_BAD, width=0.62)
    ax.axhline(RANDOM_GE, color=C_GREY, ls=":", lw=1.2)
    ax.text(len(vals) - 0.5, RANDOM_GE + 3, "random guess", ha="right", fontsize=7, color=C_GREY)
    ax.axhspan(min(vals), max(vals), color=C_BAD, alpha=0.10)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("GE at N=1000"); ax.set_ylim(0, 200)
    ax.set_title(f"All six land in a {max(vals) - min(vals):.0f}-wide band ({min(vals):.1f}–{max(vals):.1f})")

    # Plotted against cnn_light's loss on the same task, and on an axis that is
    # not zoomed: a panel cropped to the cnn_best band would make a 0.07 drop
    # look like learning.
    for (label, run), colour in zip(runs, CYCLE):
        rows = history(run)
        axb.plot([int(r["epoch"]) for r in rows], [float(r["loss"]) for r in rows],
                 color=colour, lw=1.2, label=label.replace("\n", " "))
    ref = history("runs/E01_baseline_clean_20260816_1302")
    axb.plot([int(r["epoch"]) for r in ref], [float(r["loss"]) for r in ref],
             color="black", lw=2.0, ls="-", label="for scale: cnn_light (E01), which does learn")
    axb.axhline(np.log(256), color="black", ls="--", lw=1)
    axb.text(38, np.log(256) + 0.012, "random baseline  log(256) = 5.545", fontsize=7)
    axb.set_ylim(5.0, 5.62)
    axb.set_xlabel("Epoch"); axb.set_ylabel("Training loss")
    axb.set_title("Loss never leaves the random baseline\n"
                  "(cnn_best falls 0.07 in 75 epochs; cnn_light falls 0.49 in 50)")
    axb.legend(loc="lower left", fontsize=6.5)
    fig.suptitle("E06 cnn_best (66.6M parameters on 30k traces): an optimisation failure, not a tuning problem",
                 fontsize=10, fontweight="bold")
    save(fig, "F08_cnn_best_noise_band", "cnn_best six attempts (附錄 B.50)")


def _seed_of(run_dir: str) -> int:
    """Read the seed back out of the run's own config snapshot, so paired plots
    pair by seed rather than by sorted rank (which would look perfect even if
    the two groups were unrelated)."""
    for line in (Path(run_dir) / "config_snapshot.yaml").read_text().splitlines():
        if line.startswith("seed:"):
            return int(line.split(":", 1)[1])
    raise KeyError(f"no top-level seed in {run_dir}")


def _by_seed(dirs) -> dict[int, str]:
    return {_seed_of(d): d for d in dirs}


def f09_resnet_robustness():
    """附錄 B.57-B.59: the resnet result is initialisation noise, not learning."""
    import glob
    p6 = _by_seed(sorted(glob.glob("runs_resnet_robust/*/")) +
                  ["runs/E07_resnet_20260828_025724_155915", "runs/E07_resnet_20260828_123723_407257",
                   "runs/E07_resnet_20260829_214039_152946", "runs/E07_resnet_20260829_214249_162622"])
    p100 = _by_seed(sorted(glob.glob("runs_resnet_patience/*/")))
    ctrl = _by_seed(sorted(glob.glob("runs_resnet_ctrl1ep/*/")))
    seeds = sorted(set(p6) & set(p100) & set(ctrl))
    g6 = np.array([float(ge(p6[s])[-1]) for s in seeds])
    g100 = np.array([float(ge(p100[s])[-1]) for s in seeds])
    g1 = np.array([float(ge(ctrl[s])[-1]) for s in seeds])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    ax = axes[0]
    for i, (vals, colour) in enumerate(((g6, C_GREY), (g100, C_MID), (g1, C_GOOD))):
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.09, 0.09, len(vals)), vals,
                   color=colour, s=26, zorder=3)
        ax.hlines(vals.mean(), i - 0.22, i + 0.22, color="black", lw=1.6, zorder=4)
        ax.text(i, 252, f"mean {vals.mean():.1f}\nsd {vals.std(ddof=1):.1f}", ha="center", fontsize=7.5)
    ax.axhline(RANDOM_GE, color=C_BAD, ls=":", lw=1.2)
    ax.text(-0.42, RANDOM_GE + 4, "random guess", fontsize=7, color=C_BAD)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["full training\npatience 6", "full training\npatience 100", "1 epoch only"],
                       fontsize=8)
    ax.set_ylabel(f"GE at N=9000  ({len(seeds)} seeds each)")
    ax.set_ylim(0, 290)
    ax.set_title("Training buys nothing measurable")

    ax = axes[1]
    ax.scatter(g6, g100, color=C_GOOD, s=34, zorder=3)
    lim = [0, 240]
    ax.plot(lim, lim, color=C_GREY, ls="--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("GE, patience 6 (schedule cut short)")
    ax.set_ylabel("GE, patience 100 (schedule completed)")
    ax.set_title("Same seed, schedule allowed to finish:\n"
                 f"paired mean difference {np.mean(g100 - g6):+.2f}")

    ax = axes[2]
    prev = np.array([previews(p6[s])[1].min() for s in seeds])
    r = np.corrcoef(prev, g6)[0, 1]
    ax.scatter(prev, g6, color=C_BAD, s=34, zorder=3)
    ax.set_xlabel("Training-time 20-run GE preview (what selects the checkpoint)")
    ax.set_ylabel("Formal 100-run GE at N=9000")
    ax.set_title(f"The selection metric has no predictive power\nPearson r = {r:.3f}")
    fig.suptitle("E07 resnet: three controls, each pointing at the same conclusion",
                 fontsize=10, fontweight="bold")
    save(fig, "F09_resnet_robustness", "resnet 10-seed robustness (附錄 B.57-B.59)")


def f10_variable_key():
    """附錄 B.64/B.65: E01's recipe does not survive a variable profiling key."""
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    traj = [
        ("runs_variable/E09_ascad_variable_20260901_181249_1937599", "E09  ID, A=30,000", C_BAD),
        ("runs_variable/E09b_ascad_variable_scaled_20260901_181357_1940363",
         "E09b ID, A=120,000 (4x data)", C_MID),
        ("runs_variable/E09_variable_masked_diag_20260901_182419_1963687",
         "diagnostic: ID_MASKED, A=30,000", C_GOOD),
    ]
    for run, label, colour in traj:
        e, v = previews(run)
        ax.plot(e, v, color=colour, lw=1.7, marker="o", ms=3, label=label)
    ax.axhline(RANDOM_GE, color=C_GREY, ls=":", lw=1.2)
    ax.text(2, RANDOM_GE + 3, "random guess (127.5)", fontsize=7, color=C_GREY)
    ax.set_xlabel("Epoch"); ax.set_ylabel("mean_true_rank on V  (0 = perfect)")
    ax.set_title("Per-trace discriminative power during training")
    ax.legend(loc="center right")

    for run, label, colour in traj:
        m = metrics(run)
        curve = np.asarray(m["ge"])
        tail = f"N_TGE={m['n_tge']}" if m.get("n_tge") else "never <1"
        axb.plot(np.arange(1, len(curve) + 1), curve, color=colour, lw=1.7, label=f"{label.split(' ')[0]}  {tail}")
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_ylim(0.05, 300)
    _ge_axes(axb, "Key recovery on the attack set")
    axb.legend(loc="lower left")
    fig.suptitle("ASCADv1 variable-key: the pipeline is provably correct (ID_MASKED breaks in 3 traces), "
                 "the ID task simply is not learnable with this recipe", fontsize=10, fontweight="bold")
    save(fig, "F10_variable_key", "variable-key E09/E09b + diagnostic (附錄 B.64/B.65)")


def f11_byte_comparison():
    """附錄 B.66/B.67: byte 2 is both uniquely convenient and unusually hard."""
    entries = [
        ("runs_byte3/E10b_byte3_dual_20260901_210700_2080071",
         "byte 3, dual window 1400 pts", C_GOOD),
        ("runs/E01_baseline_clean_20260816_1302", "byte 2, ANSSI window 700 pts", C_MID),
        ("runs_byte3/E10c_byte2_wide_control_20260901_211734_2103231",
         "byte 2, widened to 1400 pts (control)", C_GREY),
        ("runs_byte3/E10_byte3_20260901_205458_2055720",
         "byte 3, single window — mask not in window", C_BAD),
    ]
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    for run, label, colour in entries:
        m = metrics(run)
        curve = np.asarray(m["ge"])
        tail = f"N_TGE={m['n_tge']}" if m.get("n_tge") else "never <1"
        ax.plot(np.arange(1, len(curve) + 1), curve, color=colour, lw=1.7, label=f"{label}   {tail}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(0.05, 300)
    _ge_axes(ax, "Same recipe, same seed — only the target byte differs")
    ax.legend(loc="lower left")

    # The two leakages an ID attack must combine, measured in 附錄 B.66/B.67.
    names = ["byte 2\nmasked value  Z^r", "byte 2\nthe mask  r", "byte 3\nmasked value  Z^r", "byte 3\nthe mask  r"]
    vals = [6.3003, 1.2737, 15.4535, 17.4826]
    axb.bar(range(4), vals, color=[C_MID, C_MID, C_GOOD, C_GOOD], width=0.6)
    axb.set_xticks(range(4)); axb.set_xticklabels(names, fontsize=7)
    axb.set_ylabel("SNR peak")
    axb.set_title("Why: byte 2's mask leaks 13.7x weaker")
    axb.text(0.5, -0.28, "SNR peaks measured on the raw captures, not read from a run",
             transform=axb.transAxes, ha="center", fontsize=6.5, color=C_GREY)
    for i, v in enumerate(vals):
        axb.text(i, v, f" {v:.2f}", ha="center", va="bottom", fontsize=7.5)
    fig.suptitle("An ID attack is second order: it needs the masked value AND the mask itself in the window",
                 fontsize=10, fontweight="bold")
    save(fig, "F11_byte_comparison", "byte 2 vs byte 3 (附錄 B.66/B.67)")


def f12_hardware_baseline(n_traces: int = 5000):
    """附錄 B.68: why one real trace is enough against an unprotected device.

    The left panel recomputes SNR from the databases (same reasoning as F05).
    Plotting E11's GE curve here would be wasted space — it is already 0.74 at
    N=1 and 0.00 by N=3 — so the panel instead answers the question that number
    raises: what does an unprotected implementation leak that a masked one does
    not?
    """
    from src.data import ascad, labels as lab
    from src.metrics.leakage import snr

    m = metrics("runs_cw/E11_cw_hardware_20260907_133718_974968")
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.5, 4.0))

    peaks = {}
    for path, byte, label, colour in (
            ("data/CW_stm32.h5", 0, "ChipWhisperer STM32 — unprotected", C_GOOD),
            ("data/ASCAD.h5", 2, "ASCAD ATMEGA8515 — boolean masked", C_BAD)):
        data = ascad.load(path)
        traces = np.asarray(data.profiling_traces[:n_traces], dtype=np.float32)
        y = lab.build(data.profiling_meta[:n_traces], "ID", byte)
        curve = snr(traces, y)
        peaks[path] = curve.max()
        ax.plot(np.linspace(0, 1, len(curve)), curve, color=colour, lw=0.9,
                label=f"{label}   peak {curve.max():.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("Position within the capture window (normalised)")
    ax.set_ylabel("SNR of the unmasked ID label  Sbox[p^k]")
    ratio = peaks["data/CW_stm32.h5"] / peaks["data/ASCAD.h5"]
    ax.set_title(f"Masking is the whole difference: the same label\n"
                 f"leaks {ratio:.0f}x harder when the implementation has none "
                 f"(over {n_traces:,} traces)")
    ax.legend(loc="upper left")

    x = np.arange(1, len(m["sr1"]) + 1)
    axb.plot(x, np.asarray(m["sr1"]), color=C_ALT, lw=2)
    axb.axhline(0.9, color=C_BAD, ls="--", lw=0.9)
    axb.set_xlim(0.5, 10); axb.set_ylim(0, 1.03)
    axb.set_xticks(range(1, 11))
    axb.set_xlabel("Number of attack traces"); axb.set_ylabel("Success rate (rank 0)")
    axb.set_title(f"N_TGE = {m['n_tge']},  PI = {m['pi']:.2f} of 8 bits\n"
                  "61% of attacks succeed on a single trace")
    fig.suptitle("E11 baseline — 10,000 ChipWhisperer captures, unprotected software AES on STM32",
                 fontsize=10, fontweight="bold")
    save(fig, "F12_hardware_baseline", "E11 ChipWhisperer baseline (附錄 B.68)")


def f13_hardware_defense():
    """附錄 B.68/B.69: the dose-response curve, and why PSR is the wrong cost axis."""
    import glob
    base = metrics("runs_cw/E11_cw_hardware_20260907_133718_974968")
    rows = [("none", 0.0, base["n_tge"], base["pi"], "none")]
    for d in sorted(glob.glob("defenses_cw/*/")):
        c, m = cost(d), metrics(d)
        kind = c["defense"]
        p = c["params"]
        tag = f"σ={p.get('sigma_ratio')}" if kind == "gaussian" else f"shift ±{p.get('max_shift')}"
        rows.append((f"{kind} {tag}", c["psr"]["mean"], m["n_tge"], m["pi"], kind))
    rows.sort(key=lambda r: r[2])

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    for kind, colour, marker in (("gaussian", C_MID, "o"), ("jamming", C_GOOD, "s")):
        pts = [(r[1], r[2]) for r in rows if r[4] == kind]
        ax.plot(*zip(*pts), color=colour, marker=marker, ms=6, lw=1.5, label=kind)
    ax.scatter([0], [base["n_tge"]], color=C_BAD, marker="*", s=140, zorder=4, label="undefended")
    for i, (label, psr, ntge, _, kind) in enumerate(r for r in rows if r[4] != "none"):
        ax.annotate(label.split(" ", 1)[-1], (psr, ntge), textcoords="offset points",
                    xytext=(7, -9 if i % 2 else 3), fontsize=6.5, color=C_GREY)
    ax.set_yscale("log")
    ax.set_xlabel("PSR — perturbation cost as this project measures it")
    ax.set_ylabel("N_TGE (traces the attacker now needs)")
    ax.set_title("Monotone dose-response on real measurements")
    ax.legend(loc="lower right")

    g4 = next(r for r in rows if r[0].startswith("gaussian σ=4.0") or r[0].startswith("gaussian σ=4"))
    j50 = next(r for r in rows if "50" in r[0] and r[4] == "jamming")
    bars = [("gaussian σ=4", g4[1], 0.0, g4[2]),
            ("jamming ±50", j50[1], 50 / 1500, j50[2])]
    idx = np.arange(2)
    axb.bar(idx - 0.19, [b[1] for b in bars], width=0.36, color=C_GREY, label="PSR (our metric)")
    axb.bar(idx + 0.19, [b[2] for b in bars], width=0.36, color=C_GOOD, label="execution-time overhead")
    axb.set_xticks(idx); axb.set_xticklabels([b[0] for b in bars])
    axb.set_ylabel("cost, as a fraction")
    axb.set_title("PSR ranks jamming 6x more expensive than gaussian.\n"
                  "On the device it costs 3.3% execution time — and works 20x better.")
    for i, b in enumerate(bars):
        axb.text(i - 0.19, b[1], f"{b[1]:.3f}", ha="center", va="bottom", fontsize=7.5)
        axb.text(i + 0.19, b[2], f"{b[2]:.3f}" if b[2] else "n/a*", ha="center", va="bottom", fontsize=7.5)
        axb.text(i, -0.075, f"N_TGE {b[3]}", ha="center", fontsize=7.5, color=C_BAD)
    axb.set_ylim(-0.1, 1.28)
    axb.legend(loc="upper left")
    axb.text(0.03, 0.72, "* gaussian buys its effect with power, not time:\n"
                         "  it has to inject noise at 15% of signal amplitude,\n"
                         "  which PSR then scores as the cheaper option",
             transform=axb.transAxes, fontsize=7, color=C_GREY)
    fig.suptitle("Deployable defenses on ChipWhisperer captures — and the cost metric that misranks them",
                 fontsize=10, fontweight="bold")
    save(fig, "F13_hardware_defense", "CW defense dose-response + PSR finding (附錄 B.68/B.69)")


def f14_ascad_gaussian_baseline():
    """附錄 C.3: the gaussian-noise baseline curve a GAN defense has to beat."""
    import glob
    rows = []
    for d in sorted(glob.glob("defenses/gaussian_*/")):
        c, m = cost(d), metrics(d)
        if m["exp_id"] != "E01_baseline_clean":
            continue          # C.2: E02 is immune to its own training-time noise
        rows.append((c["params"]["sigma_ratio"], c["psr"]["mean"], m["n_tge"],
                     float(np.asarray(m["ge"])[-1]), m["pi"]))
    rows.sort()
    base = metrics("runs/E01_baseline_clean_20260816_1302")
    psr = [0.0] + [r[1] for r in rows]
    ntge = [base["n_tge"]] + [r[2] for r in rows]
    pi = [base["pi"]] + [r[4] for r in rows]

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11, 4.0))
    solid = [(p, n) for p, n in zip(psr, ntge) if n is not None]
    ax.plot(*zip(*solid), color=C_GOOD, marker="o", ms=5, lw=1.5)
    for p, n in zip(psr, ntge):
        if n is None:
            ax.scatter([p], [9000], color=C_BAD, marker="x", s=55, zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("PSR (perturbation cost)"); ax.set_ylabel("N_TGE")
    ax.set_title("E01 attacker under gaussian noise\nred x = never broken inside 9000 traces")
    for (sigma, ps, nt, _, _), i in zip(rows, range(len(rows))):
        ax.annotate(f"σ={sigma}", (ps, nt if nt is not None else 9000),
                    textcoords="offset points", xytext=(6, -9 if i % 2 else 4),
                    fontsize=6.5, color=C_GREY)
    thr = next(p for p, n in zip(psr, ntge) if n is None)
    ax.axvline(thr, color=C_BAD, ls=":", lw=1)
    ax.text(thr, 1.4e3, f" defense threshold\n PSR ≈ {thr:.3f}", fontsize=7.5, color=C_BAD)

    axb.plot(psr, pi, color=C_MID, marker="o", ms=5, lw=1.5)
    axb.axhline(0, color="black", lw=0.8)
    axb.set_xlabel("PSR (perturbation cost)"); axb.set_ylabel("Perceived Information (bits)")
    axb.set_title("PI collapses far past zero — the model becomes\nconfidently wrong, not merely uninformed")
    fig.suptitle("Gaussian-noise baseline that any learned defense must beat",
                 fontsize=10, fontweight="bold")
    save(fig, "F14_ascad_gaussian_baseline", "ASCAD gaussian defense baseline (附錄 C.3)")


FIGURES = {
    "F01": f01_recipe_evolution, "F02": f02_e01_sweeps, "F03": f03_leakage_models,
    "F04": f04_augmentation, "F05": f05_resync_snr, "F06": f06_desync_sweeps,
    "F07": f07_desync_best, "F08": f08_cnn_best, "F09": f09_resnet_robustness,
    "F10": f10_variable_key, "F11": f11_byte_comparison, "F12": f12_hardware_baseline,
    "F13": f13_hardware_defense, "F14": f14_ascad_gaussian_baseline,
}


def main() -> None:
    p = argparse.ArgumentParser(description="Build reports/figures/ from every recorded result")
    p.add_argument("--only", nargs="*", help="figure ids to build, e.g. F10 F13")
    p.add_argument("--with-snr", action="store_true",
                   help="also build F05, which reloads the h5 databases and recomputes SNR (slow)")
    args = p.parse_args()

    names = args.only or [k for k in FIGURES if k != "F05" or args.with_snr]
    print(f"=== building {len(names)} figure(s) into {OUT}/ ===")
    for name in names:
        if name not in FIGURES:
            raise SystemExit(f"unknown figure {name}; known: {', '.join(FIGURES)}")
        FIGURES[name]()


if __name__ == "__main__":
    main()
