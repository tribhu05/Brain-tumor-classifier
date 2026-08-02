"""
Tests for brain_tumor_classifier.inference.predict.

Uses a tiny, randomly-initialized Keras model (not the full VGG16
backbone) so these tests run quickly and don't require downloading
ImageNet weights -- they're testing the *prediction pipeline
mechanics* (image loading, preprocessing, result packaging), not model
accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

tf = pytest.importorskip("tensorflow")

from brain_tumor_classifier.inference.predict import predict_image  # noqa: E402


@pytest.fixture
def tiny_model():
    """A minimal 4-class classifier with no meaningful learned weights."""
    inputs = tf.keras.Input(shape=(16, 16, 3))
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


@pytest.fixture
def sample_image_path(tmp_path: Path) -> str:
    rng = np.random.RandomState(0)
    array = rng.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    path = tmp_path / "sample.jpg"
    Image.fromarray(array).save(path)
    return str(path)


def test_predict_image_returns_valid_result(tiny_model, sample_image_path):
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]

    result = predict_image(sample_image_path, tiny_model, class_names, image_size=16)

    assert result.predicted_class in class_names
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.class_probabilities.keys()) == set(class_names)
    assert abs(sum(result.class_probabilities.values()) - 1.0) < 1e-4


def test_predict_image_is_tumor_flag(tiny_model, sample_image_path):
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    result = predict_image(sample_image_path, tiny_model, class_names, image_size=16)

    expected_is_tumor = result.predicted_class != "notumor"
    assert result.is_tumor == expected_is_tumor


def test_predict_image_missing_file_raises(tiny_model):
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    with pytest.raises(FileNotFoundError):
        predict_image("/nonexistent/path.jpg", tiny_model, class_names, image_size=16)
