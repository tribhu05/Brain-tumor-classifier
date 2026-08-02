"""
predict.py
==========

Single-image inference.

Fix relative to the original notebook
----------------------------------------
The original ``detect_and_display()`` function mixed prediction logic
and ``matplotlib`` display logic in one function, and hardcoded the
class list as a third, independent literal
(``['glioma', 'meningioma', 'notumor', 'pituitary']``) separate from
the two other places class names were derived elsewhere in the
notebook. That's a single-source-of-truth violation: if the dataset's
class folders ever change, this list silently goes stale.

Here, prediction is a pure function that takes ``class_names`` as a
parameter (always derived once via
``data.dataset.discover_class_names``), returns a typed result object,
and has no plotting side effects. Display is a separate, optional step
in ``visualization/plots.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of a single-image prediction."""

    predicted_class: str
    confidence: float
    is_tumor: bool
    class_probabilities: dict


def predict_image(
    image_path: str,
    model,
    class_names: Sequence[str],
    image_size: int = 128,
    no_tumor_label: str = "notumor",
) -> PredictionResult:
    """Run inference on a single MRI image.

    Args:
        image_path: Path to the image file.
        model: A trained ``tf.keras.Model``.
        class_names: Canonical, ordered class name list matching the
            model's output layer (must be the same ordering used
            during training -- see ``data.dataset.discover_class_names``).
        image_size: Square size the image is resized to before inference.
        no_tumor_label: The class name that represents "no tumor
            present", used to set ``is_tumor``.

    Returns:
        A :class:`PredictionResult`.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist or can't be read.
    """
    import tensorflow as tf

    from brain_tumor_classifier.data.augmentation import rescale_only

    try:
        image = tf.keras.utils.load_img(image_path, target_size=(image_size, image_size))
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError(f"Could not read image at {image_path}: {exc}") from exc

    image_array = tf.keras.utils.img_to_array(image)
    processed = rescale_only(image_array)
    batch = np.expand_dims(processed, axis=0)

    probabilities = model.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    result = PredictionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        is_tumor=predicted_class != no_tumor_label,
        class_probabilities={
            name: float(prob) for name, prob in zip(class_names, probabilities)
        },
    )
    logger.info(
        "Prediction for %s: %s (%.2f%% confidence)",
        image_path,
        result.predicted_class,
        result.confidence * 100,
    )
    return result
