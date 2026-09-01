"""GE-based model selection. See CLAUDE.md §6.3 — the core methodological
contribution of this refactor: val_loss/val_accuracy barely correlate with
attack effectiveness on 256-way SCA classification (random baseline ~0.39%),
so checkpoint selection and early stopping must be driven by Guessing Entropy
computed on V instead (pitfall #4).
"""

from __future__ import annotations

import keras
import numpy as np

from src.attack import keyrank, predict, scores
from src.data.ascad import get_correct_key
from src.data.labels import LeakageModel


class GEModelSelection(keras.callbacks.Callback):
    """Every `eval_every` epochs: run inference on V, compute GE with `n_runs_val`
    independent attack runs, derive N_TGE (or GE at the final trace count if not
    converged), and checkpoint to best.keras if it beats the historical best.

    Ranking rule: both converged -> smaller N_TGE wins; only one converged ->
    it wins; neither converged -> lower final-GE wins.

    Early stops after `patience` evaluations (i.e. patience * eval_every epochs)
    with no improvement by this rule.
    """

    def __init__(self, x_val, meta_val, target_byte: int, eval_every: int = 5,
                 n_runs_val: int = 20, patience: int = 6, checkpoint_path: str = "best.keras",
                 max_traces: int = 1000, seed: int = 0, verbose: bool = True,
                 leakage_model: LeakageModel = "ID", mask: np.ndarray | None = None,
                 y_val: np.ndarray | None = None):
        super().__init__()
        self.x_val = x_val
        self.plaintexts = meta_val["plaintext"]
        self.target_byte = target_byte
        self.eval_every = eval_every
        self.n_runs_val = n_runs_val
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.max_traces = min(max_traces, len(x_val))
        self.seed = seed
        self.verbose = verbose
        self.leakage_model = leakage_model
        self.mask = mask
        self.y_val = y_val

        # Key recovery accumulates per-trace scores under the assumption that every
        # trace was produced with the SAME key. That holds on the fixed-key
        # databases, but NOT on ASCADv1-variable, whose profiling set (and therefore
        # V) carries ~256 distinct keys — there, get_correct_key would silently
        # return trace 0's key and score every other trace against the wrong
        # hypothesis (measured: only 0.54% of V actually has it), making the whole
        # GE preview meaningless and checkpoint selection effectively random.
        # See CLAUDE.md 附錄 B.62.
        keys = np.asarray(meta_val["key"][:, target_byte])
        self.variable_key = bool(np.unique(keys).size > 1)
        if self.variable_key:
            if y_val is None:
                raise ValueError(
                    "V holds multiple keys (variable-key database), so GE cannot be computed on it. "
                    "GEModelSelection needs y_val to fall back to the key-independent "
                    "mean-true-class-rank criterion."
                )
            self.correct_key = None
        else:
            self.correct_key = get_correct_key(meta_val, target_byte)
        self.best_n_tge: int | None = None
        self.best_final_ge: float | None = None
        self.evals_without_improvement = 0
        self.history: list[dict] = []

    def _is_better(self, n_tge: int | None, final_ge: float) -> bool:
        if self.best_n_tge is None and self.best_final_ge is None:
            return True
        cur_converged = n_tge is not None
        best_converged = self.best_n_tge is not None
        if cur_converged and best_converged:
            return n_tge < self.best_n_tge
        if cur_converged != best_converged:
            return cur_converged
        return final_ge < self.best_final_ge

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        epoch_number = epoch + 1  # keras epoch is 0-indexed
        if epoch_number % self.eval_every != 0:
            return

        probs = predict.run(self.model, self.x_val)

        if self.variable_key:
            # Key-independent stand-in for GE: the mean rank of each trace's own
            # true class, using that trace's own known key. Same "number of
            # candidates strictly ahead of the truth" convention as keyrank, so the
            # scale matches (0 = perfect, ~127.5 = random for 256 classes). It
            # measures the per-trace discriminative power that drives GE, without
            # needing all traces to share a key.
            true_p = probs[np.arange(len(self.y_val)), self.y_val][:, None]
            metric = float((probs > true_p).sum(axis=1).mean())
            n_tge, final_ge = None, metric
            label = "mean_true_rank"
        else:
            sc = scores.build(probs, self.plaintexts, self.target_byte,
                               leakage_model=self.leakage_model, mask=self.mask)
            ranks = keyrank.evaluate(sc, self.correct_key, n_runs=self.n_runs_val,
                                      max_traces=self.max_traces, seed=self.seed)
            ge_curve = keyrank.ge(ranks)
            n_tge = keyrank.n_tge(ge_curve)
            final_ge = float(ge_curve[-1])
            label = "final_GE"
        self.history.append({"epoch": epoch_number, "n_tge": n_tge, "ge": final_ge})

        improved = self._is_better(n_tge, final_ge)
        if self.verbose:
            tag = " (new best, saving)" if improved else ""
            print(f"\n[GEModelSelection] epoch {epoch_number}: N_TGE={n_tge} {label}={final_ge:.2f}{tag}")

        if improved:
            self.best_n_tge = n_tge
            self.best_final_ge = final_ge
            self.evals_without_improvement = 0
            self.model.save(self.checkpoint_path)
        else:
            self.evals_without_improvement += 1
            if self.evals_without_improvement >= self.patience:
                self.model.stop_training = True
                if self.verbose:
                    print(f"[GEModelSelection] no improvement for {self.patience} evaluations, stopping")
