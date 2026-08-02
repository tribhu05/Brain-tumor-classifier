"""
seed.py
=======

Reproducibility utilities.

The original notebook never set a random seed anywhere -- not for
Python's ``random`` module (used in augmentation jitter and the
train/test shuffle), not for NumPy, not for TensorFlow/Keras weight
initialization or Dropout. That means two runs of the exact same
notebook on the exact same data produce different metrics, which makes
results impossible to reproduce or compare across experiments.

``set_global_seed`` fixes that by seeding every source of randomness
the pipeline touches.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and TensorFlow RNGs for reproducible runs.

    Args:
        seed: The seed value to apply everywhere.

    Notes:
        TensorFlow is imported lazily inside this function so that
        utilities which don't need TF (e.g. pure data-splitting tests)
        can import this module without requiring TensorFlow to be
        installed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        logger.warning(
            "TensorFlow is not installed; skipped seeding tf.random. "
            "This is fine for data-only utilities but not for training."
        )

    logger.info("Global random seed set to %d", seed)
