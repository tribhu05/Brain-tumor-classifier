"""Tests for brain_tumor_classifier.data.augmentation."""

from __future__ import annotations

import numpy as np

from brain_tumor_classifier.data.augmentation import augment_image, rescale_only


def _dummy_image() -> np.ndarray:
    rng = np.random.RandomState(0)
    return rng.randint(0, 255, size=(16, 16, 3), dtype=np.uint8)


def test_rescale_only_scales_to_unit_range():
    image = _dummy_image()
    result = rescale_only(image)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_augment_image_scales_to_unit_range():
    image = _dummy_image()
    result = augment_image(image)
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    assert result.shape == image.shape


def test_augment_image_is_stochastic():
    """Augmentation should vary run to run (brightness/contrast jitter is random)."""
    image = _dummy_image()
    result_a = augment_image(image)
    result_b = augment_image(image)
    # Extremely unlikely to be bit-identical given independent random jitter.
    assert not np.array_equal(result_a, result_b)


def test_rescale_only_is_deterministic():
    """Eval/inference preprocessing must NOT be stochastic (fixes leakage bug)."""
    image = _dummy_image()
    result_a = rescale_only(image)
    result_b = rescale_only(image)
    assert np.array_equal(result_a, result_b)
