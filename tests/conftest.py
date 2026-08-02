"""
conftest.py
===========

Shared pytest fixtures.

All tests use a small synthetically-generated image dataset rather than
the real MRI dataset. This keeps the test suite fast, deterministic,
and runnable in CI without needing gigabytes of medical imaging data or
any dataset-access credentials.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    """Create a tiny fake dataset with the same class-folder layout as the real one.

    Layout:
        tmp_path/
            glioma/      (3 images)
            meningioma/  (3 images)
            notumor/     (3 images)
            pituitary/   (3 images)

    Returns:
        Path to the root directory containing the class subfolders.
    """
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    rng = np.random.RandomState(0)

    for class_name in class_names:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(3):
            array = rng.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(array).save(class_dir / f"{class_name}_{i}.jpg")

    return tmp_path
