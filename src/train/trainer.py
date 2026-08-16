"""Training loop. See CLAUDE.md §4.2 — `fit` must never touch E, and contains
no Key Rank computation (that belongs to src/attack + src/metrics, per P5).
"""

from __future__ import annotations

import os
from typing import Any

import keras
import numpy as np

from src.data.preprocess import gaussian_augment
from src.train.callbacks import GEModelSelection
from src.train.lr_schedule import OneCycleLR


def _build_optimizer(name: str, lr: float) -> keras.optimizers.Optimizer:
    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    raise ValueError(f"unknown optimizer: {name!r}")


def _with_channel_dim(x: np.ndarray) -> np.ndarray:
    return x[..., None] if x.ndim == 2 else x


class _GaussianAugmentedDataset(keras.utils.PyDataset):
    """Feeds model.fit() a freshly re-noised copy of `x` every epoch (pitfall
    #11 — augmentation must not be a fixed precomputed dataset). Seed derives
    from cfg.seed + epoch per CLAUDE.md §8.1. Noise generation itself is the
    pure `gaussian_augment` function in src/data/preprocess.py; this class is
    just the stateful Keras-facing glue (P2 confines side effects to trainer.py).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, batch_size: int, sigma_ratio: float, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.sigma_ratio = sigma_ratio
        self.seed = seed
        self.epoch = 0
        self._augmented = gaussian_augment(self.x, self.sigma_ratio, seed=self.seed + self.epoch)

    def __len__(self) -> int:
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx: int):
        lo = idx * self.batch_size
        hi = min(lo + self.batch_size, len(self.x))
        return _with_channel_dim(self._augmented[lo:hi]), self.y[lo:hi]

    def on_epoch_end(self) -> None:
        self.epoch += 1
        self._augmented = gaussian_augment(self.x, self.sigma_ratio, seed=self.seed + self.epoch)


def fit(x_a: np.ndarray, y_a: np.ndarray, x_v: np.ndarray, y_v: np.ndarray, meta_v: np.ndarray,
        model: keras.Model, cfg: dict[str, Any], checkpoint_path: str = "best.keras") -> tuple[keras.Model, list[dict]]:
    """Train on (A) with model selection driven by GEModelSelection on (V).
    Returns the best checkpoint and the per-epoch history rows for train_history.csv.
    `meta_v` is V's raw metadata (plaintext/key), needed by GEModelSelection to
    score attack runs — it is not used as a training label.
    """
    train_cfg = cfg["train"]
    augment_cfg = cfg.get("augment", {}).get("gaussian", {})

    selection_cfg = train_cfg.get("selection", {})
    model.compile(
        optimizer=_build_optimizer(train_cfg["optimizer"], train_cfg["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    leakage_model = cfg["leakage"]["model"]
    mask_index = cfg["leakage"].get("mask_index")
    mask = meta_v["masks"][:, mask_index].astype(np.uint8) if leakage_model == "ID_MASKED" else None

    ge_callback = GEModelSelection(
        x_val=x_v,
        meta_val=meta_v,
        target_byte=cfg["data"]["target_byte"],
        eval_every=selection_cfg.get("eval_every", 5),
        n_runs_val=selection_cfg.get("n_runs_val", 20),
        patience=selection_cfg.get("patience", 6),
        checkpoint_path=checkpoint_path,
        max_traces=selection_cfg.get("max_traces", 1000),
        seed=cfg.get("seed", 0),
        leakage_model=leakage_model,
        mask=mask,
    )

    callbacks: list[keras.callbacks.Callback] = [ge_callback]
    lr_schedule = train_cfg.get("lr_schedule")
    if lr_schedule == "one_cycle":
        one_cycle_cfg = train_cfg.get("one_cycle", {})
        callbacks.append(OneCycleLR(
            max_lr=train_cfg["lr"],
            end_percentage=one_cycle_cfg.get("end_percentage", 0.2),
            scale_percentage=one_cycle_cfg.get("scale_percentage", 0.1),
        ))
    elif lr_schedule not in (None, "flat"):
        raise ValueError(f"unknown lr_schedule: {lr_schedule!r}")

    if augment_cfg.get("enabled"):
        train_data = _GaussianAugmentedDataset(
            x_a, y_a,
            batch_size=train_cfg["batch_size"],
            sigma_ratio=augment_cfg.get("sigma_ratio", 0.5),
            seed=cfg.get("seed", 0),
        )
        history = model.fit(
            train_data,
            validation_data=(_with_channel_dim(x_v), y_v),
            epochs=train_cfg["epochs"],
            callbacks=callbacks,
            verbose=2,
        )
    else:
        history = model.fit(
            _with_channel_dim(x_a), y_a,
            validation_data=(_with_channel_dim(x_v), y_v),
            batch_size=train_cfg["batch_size"],
            epochs=train_cfg["epochs"],
            callbacks=callbacks,
            verbose=2,
        )

    if os.path.exists(checkpoint_path):
        best_model = keras.models.load_model(checkpoint_path)
    else:
        # eval_every never landed on a completed epoch (e.g. epochs < eval_every):
        # GE-based selection never ran, fall back to the final trained model.
        model.save(checkpoint_path)
        best_model = model

    n_epochs = len(history.history["loss"])
    rows = [
        {
            "epoch": i + 1,
            "loss": history.history["loss"][i],
            "accuracy": history.history.get("accuracy", [None] * n_epochs)[i],
            "val_loss": history.history["val_loss"][i],
            "val_accuracy": history.history.get("val_accuracy", [None] * n_epochs)[i],
        }
        for i in range(n_epochs)
    ]
    ge_by_epoch = {row["epoch"]: row for row in ge_callback.history}
    for row in rows:
        if row["epoch"] in ge_by_epoch:
            row["n_tge"] = ge_by_epoch[row["epoch"]]["n_tge"]
            row["ge"] = ge_by_epoch[row["epoch"]]["ge"]

    return best_model, rows
