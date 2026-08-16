"""One-Cycle learning rate policy (Smith, "Super-Convergence"), ported from the
reference implementation used by Zaid et al.'s efficient-CNN-for-SCA
methodology (https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA,
itself adapted from https://github.com/titu1994/keras-one-cycle / fastai).

Investigated as the likely root cause of E01's GE plateauing far above the
CLAUDE.md §11 target (see runs/E01_baseline_clean_*): the reference ASCAD
desync0 model converges in ~50 epochs specifically because of this per-batch
triangular LR schedule (flat LR was the untested assumption in the original
CLAUDE.md §6.2 hyperparameter table).

Per-batch LR: rises from max_lr*scale to max_lr over the first
(1-end_percentage)/2 of all training iterations, descends back to
max_lr*scale over the next (1-end_percentage)/2, then anneals down to
max_lr*scale/100 over the final end_percentage fraction. This is a direct
port of the reference `compute_lr` — deliberately not simplified, since the
"low/peak" shape it produces only reduces to something intuitive for the
specific scale_percentage=0.1 the reference uses.
"""

from __future__ import annotations

import keras


class OneCycleLR(keras.callbacks.Callback):
    def __init__(self, max_lr: float, end_percentage: float = 0.2, scale_percentage: float = 0.1,
                 verbose: bool = True):
        super().__init__()
        if not 0.0 <= end_percentage <= 1.0:
            raise ValueError("end_percentage must be between 0 and 1")
        self.max_lr = max_lr
        self.end_percentage = end_percentage
        self.scale = scale_percentage
        self.verbose = verbose

        self._iteration = 0
        self._num_iterations = 1
        self._mid_cycle_id = 1

    def _compute_lr(self) -> float:
        if self._iteration > 2 * self._mid_cycle_id:
            pct = (self._iteration - 2 * self._mid_cycle_id) / (self._num_iterations - 2 * self._mid_cycle_id)
            return self.max_lr * (1.0 + pct * (1.0 - 100.0) / 100.0) * self.scale
        elif self._iteration > self._mid_cycle_id:
            pct = 1.0 - (self._iteration - self._mid_cycle_id) / self._mid_cycle_id
            return self.max_lr * (1.0 + pct * (self.scale * 100.0 - 1.0)) * self.scale
        else:
            pct = self._iteration / self._mid_cycle_id
            return self.max_lr * (1.0 + pct * (self.scale * 100.0 - 1.0)) * self.scale

    def on_train_begin(self, logs: dict | None = None) -> None:
        epochs = self.params["epochs"]
        steps = self.params.get("steps") or 1
        self._num_iterations = max(epochs * steps, 2)
        self._mid_cycle_id = max(int(self._num_iterations * (1.0 - self.end_percentage) / 2), 1)
        self._iteration = 0
        self.model.optimizer.learning_rate = self._compute_lr()

    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        self._iteration += 1
        if self._iteration <= self._num_iterations:
            self.model.optimizer.learning_rate = self._compute_lr()

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if self.verbose:
            print(f"  [OneCycleLR] lr={float(self.model.optimizer.learning_rate):.6f}")
