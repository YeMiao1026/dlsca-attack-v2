"""3 residual blocks + GAP, ~30K params. Stage B transferability attacker
(architecture mismatch vs. cnn_light). See CLAUDE.md §6.1.
"""

from __future__ import annotations

import keras

from src.models.registry import register


@register("resnet")
def build(input_dim: int = 700, n_classes: int = 256) -> keras.Model:
    raise NotImplementedError
