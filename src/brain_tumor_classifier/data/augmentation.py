"""
augmentation.py
================

Image augmentation, split into a train-only path and an eval-only path.

Why this file exists
---------------------
In the original notebook, a single ``open_images()`` helper called
``augment_image()`` (random brightness/contrast jitter) unconditionally
-- for both the *training* set and the *test* set. That means every
reported test accuracy, classification report, and confusion matrix was
computed on randomly perturbed test images rather than the real,
unmodified test set. Test metrics were therefore non-reproducible
(different every run) and did not represent true generalization
performance. This is a data leakage / evaluation-validity bug, not a
style issue.

The fix: augmentation now lives behind an explicit ``augment: bool``
flag threaded through the pipeline (see ``dataset.py``), and the
default eval/predict paths never set it to True.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageEnhance


def augment_image(image: np.ndarray) -> np.ndarray:
    """Apply random brightness/contrast jitter and rescale to [0, 1].

    This must only be used on training images. Applying it to
    validation or test images invalidates evaluation metrics (see
    module docstring).

    Args:
        image: An HxWx3 uint8-range array (values in [0, 255]).

    Returns:
        An HxWx3 float array with pixel values rescaled to [0, 1],
        after random brightness/contrast perturbation.
    """
    pil_image = Image.fromarray(np.uint8(image))
    pil_image = ImageEnhance.Brightness(pil_image).enhance(random.uniform(0.8, 1.2))
    pil_image = ImageEnhance.Contrast(pil_image).enhance(random.uniform(0.8, 1.2))
    return np.array(pil_image) / 255.0


def rescale_only(image: np.ndarray) -> np.ndarray:
    """Rescale an image to [0, 1] with no augmentation.

    Used for validation, test, and inference so that evaluation is
    always performed on the true, unperturbed data.

    Args:
        image: An HxWx3 uint8-range array (values in [0, 255]).

    Returns:
        An HxWx3 float array with pixel values rescaled to [0, 1].

    Note:
        VGG16's ImageNet weights were originally trained with
        channel-wise mean subtraction (``keras.applications.vgg16.preprocess_input``),
        not simple min-max rescaling. The original notebook used plain
        ``/255.0`` rescaling for both training and inference, and this
        refactor preserves that behavior exactly to avoid changing model
        semantics. See docs/known_limitations.md for details on this
        tradeoff and how to switch to ``preprocess_input`` if desired.
    """
    return np.asarray(image, dtype=np.float32) / 255.0
