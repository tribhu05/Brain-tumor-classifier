"""
config.py
=========

Centralized, typed configuration for the whole pipeline.

Why this exists
----------------
The original prototype notebook scattered hyperparameters (image size,
batch size, epoch count, learning rate, dataset paths) across multiple
cells as bare magic numbers and hardcoded absolute Windows paths
(e.g. ``C:\\Users\\priya\\OneDrive\\Desktop\\project\\Training``). That
made the notebook impossible to run on any other machine and impossible
to reason about without reading every cell.

This module gives every downstream component (data loading, model
building, training, evaluation, inference) a single, testable, typed
source of truth. Values can be overridden via a YAML file
(``configs/config.yaml``) or environment variables, with defaults that
exactly match the original prototype's hyperparameters so behavior is
preserved unless the user explicitly changes the config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    """Paths and split parameters for the dataset."""

    train_dir: str = os.environ.get("BTC_TRAIN_DIR", "data/raw/Training")
    test_dir: str = os.environ.get("BTC_TEST_DIR", "data/raw/Testing")
    # Fraction of the training set held out for validation. The original
    # notebook had NO validation split at all (see docs/known_limitations.md,
    # issue #2) -- this is a genuine fix, not just refactoring.
    validation_split: float = 0.15
    image_size: int = 128
    class_names: List[str] = field(
        default_factory=lambda: ["glioma", "meningioma", "notumor", "pituitary"]
    )


@dataclass
class ModelConfig:
    """Architecture hyperparameters. Mirrors the original notebook exactly."""

    num_classes: int = 4
    dropout_1: float = 0.3
    dense_units: int = 128
    dropout_2: float = 0.2
    # Names of the VGG16 backbone layers to unfreeze for fine-tuning.
    # The original notebook selected these via negative indexing
    # (layers[-2], layers[-3], layers[-4]), which is fragile: it silently
    # breaks if Keras ever changes how include_top=False VGG16 is built.
    # These names are what those indices resolve to today, made explicit.
    trainable_backbone_layers: List[str] = field(
        default_factory=lambda: ["block5_conv3", "block5_conv2", "block5_conv1"]
    )


@dataclass
class TrainingConfig:
    """Training loop hyperparameters. Defaults match the original notebook."""

    batch_size: int = 20
    epochs: int = 10
    learning_rate: float = 0.0001
    seed: int = 42
    checkpoint_dir: str = "artifacts/checkpoints"
    log_dir: str = "artifacts/logs"
    early_stopping_patience: Optional[int] = 5


@dataclass
class Config:
    """Top-level config aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load a Config, overriding defaults with values from a YAML file.

        Args:
            path: Path to a YAML file shaped like ``configs/config.yaml``.

        Returns:
            A populated Config instance. Any key omitted from the YAML
            file falls back to the dataclass default (which matches the
            original notebook's hardcoded values).

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )


def load_config(path: Optional[str | Path] = None) -> Config:
    """Load configuration, falling back to hardcoded defaults if no file given.

    Args:
        path: Optional path to a YAML config file. If ``None``, returns
            a Config populated purely with dataclass defaults.

    Returns:
        A Config instance.
    """
    if path is None:
        return Config()
    return Config.from_yaml(path)
