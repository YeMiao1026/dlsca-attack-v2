"""Lightweight CNN, 18,640-18,642 params (CLAUDE.md §6.1 originally described
lecun_normal init — corrected below, see CLAUDE.md 附錄 B.11):

    Conv(4,k3)-BN-AvgPool-Conv(8,k51)-BN-AvgPool-FC(10)-FC(10)-FC(n_classes)

The k=51 second-layer kernel is a deliberately large receptive field for
desync tolerance, not an accident of tuning. Activations are SELU +
he_uniform init — matches the actual "CNNd" model
(train_with_pure/train_cnnd.py) that produced this project's historical
N_TGE≈80-100 reference result; CLAUDE.md's original "lecun_normal" claim
does not match that ground-truth implementation.
"""

from __future__ import annotations

import keras
from keras import layers

from src.models.registry import register

_SELU = dict(activation="selu", kernel_initializer="he_uniform")


@register("cnn_light")
def build(input_dim: int = 700, n_classes: int = 256) -> keras.Model:
    inputs = keras.Input(shape=(input_dim, 1))
    x = layers.Conv1D(4, 3, padding="same", **_SELU)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Conv1D(8, 51, padding="same", **_SELU)(x)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10, **_SELU)(x)
    x = layers.Dense(10, **_SELU)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="cnn_light")
