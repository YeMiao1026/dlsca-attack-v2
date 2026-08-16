"""3 residual blocks + GAP, 28,816 params (target was CLAUDE.md §6.1's
"~30K", tuned via _FILTERS to land close). Stage B transferability attacker
(architecture mismatch vs. cnn_light).
"""

from __future__ import annotations

import keras
from keras import layers

from src.models.registry import register

# No upstream reference for this one (unlike cnn_best/ASCAD_train_models.py) —
# CLAUDE.md §6.1 only specifies "3 residual blocks + GAP, ~30K params" as a
# target for Stage B architecture-mismatch transferability testing. Standard
# ReLU+BN+he_normal (the textbook-correct pairing, unlike cnn_light's
# empirically-better-but-nonstandard SELU+he_uniform) since there's no
# historical result here to contradict theory.
_FILTERS = (12, 24, 48)


def _residual_block(x: keras.KerasTensor, filters: int) -> keras.KerasTensor:
    in_channels = x.shape[-1]
    shortcut = x if in_channels == filters else layers.Conv1D(filters, 1, padding="same")(x)

    y = layers.Conv1D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv1D(filters, 3, padding="same", kernel_initializer="he_normal")(y)
    y = layers.BatchNormalization()(y)

    y = layers.Add()([y, shortcut])
    return layers.Activation("relu")(y)


@register("resnet")
def build(input_dim: int = 700, n_classes: int = 256) -> keras.Model:
    inputs = keras.Input(shape=(input_dim, 1))
    x = layers.Conv1D(_FILTERS[0], 3, padding="same", kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.AveragePooling1D(2)(x)

    for filters in _FILTERS:
        x = _residual_block(x, filters)
        x = layers.AveragePooling1D(2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="resnet")
