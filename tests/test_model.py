"""
Tests for brain_tumor_classifier.models.vgg16_classifier.

These require TensorFlow (and download ImageNet weights on first run),
so they're skipped automatically in environments without TensorFlow
installed via ``pytest.importorskip``. They run in CI where
``requirements.txt`` guarantees TensorFlow is present.
"""

from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow")

from brain_tumor_classifier.models.vgg16_classifier import build_model  # noqa: E402


def test_build_model_output_shape():
    model = build_model(image_size=32, num_classes=4)
    dummy_input = tf.zeros((1, 32, 32, 3))
    output = model(dummy_input, training=False)
    assert output.shape == (1, 4)


def test_build_model_output_is_a_valid_probability_distribution():
    model = build_model(image_size=32, num_classes=4)
    dummy_input = tf.random.uniform((2, 32, 32, 3))
    output = model(dummy_input, training=False).numpy()

    assert output.shape == (2, 4)
    row_sums = output.sum(axis=1)
    assert (abs(row_sums - 1.0) < 1e-4).all()


def test_build_model_unfreezes_only_requested_layers():
    trainable_layers = ("block5_conv3", "block5_conv2")
    model = build_model(image_size=32, num_classes=4, trainable_backbone_layers=trainable_layers)

    base_model = model.layers[0]
    for layer in base_model.layers:
        expected_trainable = layer.name in trainable_layers
        assert layer.trainable == expected_trainable, (
            f"Layer '{layer.name}' trainable={layer.trainable}, expected {expected_trainable}"
        )


def test_build_model_invalid_layer_name_raises():
    with pytest.raises(ValueError):
        build_model(image_size=32, num_classes=4, trainable_backbone_layers=("not_a_real_layer",))
