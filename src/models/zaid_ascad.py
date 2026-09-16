"""Reference efficient-CNN attackers from the literature, reproduced verbatim so
this project's defenses can be evaluated against an attacker that is not its
own home-grown model (文獻查核.md §3.1 item 5).

Two models, both for ASCAD fixed-key desync0 (700 samples):

  zaid_ascad0     Zaid, Bossuet, Habrard, Venelli, TCHES 2020(1), "Methodology for
                  Efficient CNN Architectures in Profiling Attacks".
                  Source: github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA,
                  ASCAD/N0=0/cnn_architecture.py.
                  Conv1D(4, k=1, same) - BN - AvgPool(2) - FC10 - FC10 - FC256
                  SELU + he_uniform throughout.

  wouters_noconv1_ascad0
                  Wouters, Arribas, Gierlichs, Preneel, TCHES 2020(3), "Revisiting a
                  Methodology for Efficient CNN Architectures in Profiling Attacks".
                  Source: github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA, src/models.py,
                  noConv1_ascad_desync_0 — Zaid's model with the convolution removed:
                  AvgPool(2) - FC10 - FC10 - FC256, SELU, Keras default init.

Why this matters: cnn_light (E01) is NOT Zaid's model, despite the lineage in
its docstring. It descends from the team's own train_cnnd.py and has an extra
Conv(8, k=51) block and a k=3 first kernel. E01 needs 475 traces; the
literature reports efficient CNNs far below that on the same data. A defense
that holds against a weaker-than-published attacker is not yet evidence.
"""

from __future__ import annotations

import keras
from keras import layers

from src.models.registry import register

_SELU_HE = dict(activation="selu", kernel_initializer="he_uniform")


@register("zaid_ascad0")
def build_zaid(input_dim: int = 700, n_classes: int = 256) -> keras.Model:
    inputs = keras.Input(shape=(input_dim, 1))
    x = layers.Conv1D(4, 1, padding="same", **_SELU_HE)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.AveragePooling1D(2, strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10, **_SELU_HE)(x)
    x = layers.Dense(10, **_SELU_HE)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="zaid_ascad0")


@register("wouters_noconv1_ascad0")
def build_wouters(input_dim: int = 700, n_classes: int = 256) -> keras.Model:
    inputs = keras.Input(shape=(input_dim, 1))
    x = layers.AveragePooling1D(2, strides=2)(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation="selu")(x)
    x = layers.Dense(10, activation="selu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="wouters_noconv1_ascad0")
