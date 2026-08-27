"""
vgg16_classifier.py
====================

Model architecture: VGG16 backbone (ImageNet weights) with a partially
fine-tuned last block, feeding a small dense classification head.

This is the *same architecture* as the original notebook:

    Input -> VGG16(include_top=False) -> Flatten -> Dropout(0.3)
          -> Dense(128, relu) -> Dropout(0.2) -> Dense(num_classes, softmax)

with the same 3 backbone layers unfrozen for fine-tuning and the same
Adam(lr=1e-4) / sparse_categorical_crossentropy configuration. Nothing
about *what* the model is has changed -- only *how* the unfrozen layers
are selected.

Fix relative to the original notebook
---------------------------------------
The original selected fine-tunable layers via negative indexing
(``base_model.layers[-2].trainable = True``, etc.). This is fragile:
it silently unfreezes the wrong layers if Keras ever changes how
``VGG16(include_top=False)`` is structured (e.g. adds/removes a
pooling layer), with no error raised. This module instead unfreezes
layers by explicit name (``block5_conv3``, ``block5_conv2``,
``block5_conv1`` -- what those indices resolve to today), and raises a
clear error if an expected layer name is missing.
"""

from __future__ import annotations

import logging
from typing import Sequence

logger = logging.getLogger(__name__)


def build_model(
    image_size: int = 128,
    num_classes: int = 4,
    learning_rate: float = 0.0001,
    dropout_1: float = 0.3,
    dense_units: int = 128,
    dropout_2: float = 0.2,
    trainable_backbone_layers: Sequence[str] = ("block5_conv3", "block5_conv2", "block5_conv1"),
    weights: str | None = "imagenet",
):
    """Build and compile the VGG16-based classifier.

    Args:
        image_size: Square input image size (e.g. 128 -> 128x128x3 input).
        num_classes: Number of output classes.
        learning_rate: Adam optimizer learning rate.
        dropout_1: Dropout rate after the Flatten layer.
        dense_units: Number of units in the hidden Dense layer.
        dropout_2: Dropout rate after the hidden Dense layer.
        trainable_backbone_layers: Names of VGG16 backbone layers to
            unfreeze for fine-tuning. All other backbone layers remain
            frozen with ImageNet weights.
        weights: Pretrained weights to initialize backbone with, or None.

    Returns:
        A compiled ``tf.keras.Model`` ready for ``model.fit(...)``.

    Raises:
        ValueError: If any name in ``trainable_backbone_layers`` does
            not exist in the VGG16 backbone.
    """
    
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    base_model = VGG16(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights=weights,
    )

    base_layer_names = {layer.name for layer in base_model.layers}
    missing = set(trainable_backbone_layers) - base_layer_names
    if missing:
        raise ValueError(
            f"Requested trainable layers not found in VGG16 backbone: {missing}. "
            f"Available layer names: {sorted(base_layer_names)}"
        )

    for layer in base_model.layers:
        layer.trainable = layer.name in trainable_backbone_layers

    n_trainable = sum(layer.trainable for layer in base_model.layers)
    logger.info(
        "VGG16 backbone: %d/%d layers trainable (%s)",
        n_trainable,
        len(base_model.layers),
        sorted(trainable_backbone_layers),
    )

    model = Sequential(
        [
            Input(shape=(image_size, image_size, 3)),
            base_model,
            Flatten(),
            Dropout(dropout_1),
            Dense(dense_units, activation="relu"),
            Dropout(dropout_2),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    return model
