"""Tests for brain_tumor_classifier.config."""

from __future__ import annotations

import pytest

from brain_tumor_classifier.config import Config, load_config


def test_default_config_matches_original_notebook_hyperparameters():
    """Defaults must match the original prototype so behavior is preserved."""
    config = Config()
    assert config.data.image_size == 128
    assert config.training.batch_size == 20
    assert config.training.epochs == 10
    assert config.training.learning_rate == 0.0001
    assert config.model.num_classes == 4
    assert config.model.dropout_1 == 0.3
    assert config.model.dense_units == 128
    assert config.model.dropout_2 == 0.2


def test_load_config_from_yaml(tmp_path):
    yaml_content = """
data:
  image_size: 224
  validation_split: 0.2
training:
  batch_size: 32
  epochs: 5
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml_content)

    config = load_config(config_path)

    assert config.data.image_size == 224
    assert config.data.validation_split == 0.2
    assert config.training.batch_size == 32
    assert config.training.epochs == 5
    # Unspecified values fall back to defaults.
    assert config.training.learning_rate == 0.0001


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_load_config_none_returns_defaults():
    config = load_config(None)
    assert config == Config()
