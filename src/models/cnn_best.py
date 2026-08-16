"""ASCAD paper's CNN_best, 66,652,544 params (VGG-style, 5x Conv(k11) + 3x FC).
See CLAUDE.md §6.1, §6.2. Reference implementation: ASCAD/ASCAD_train_models.py::cnn_best.

Must be trained with RMSprop lr=1e-5 per the original paper (pitfall #10) —
Adam 1e-3 diverges.
"""

from __future__ import annotations

import keras

from src.models.registry import register


@register("cnn_best")
def build(input_dim: int = 700, n_classes: int = 256) -> keras.Model:
    raise NotImplementedError
