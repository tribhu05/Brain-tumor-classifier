"""
Tests for brain_tumor_classifier.data.dataset.

These specifically cover the bugs identified in the original notebook:
label-order consistency, validation-split correctness, and no images
being silently dropped.
"""

from __future__ import annotations

import pytest

from brain_tumor_classifier.data.dataset import (
    discover_class_names,
    encode_labels,
    load_paths_and_labels,
    train_validation_split,
)


def test_discover_class_names_is_sorted(synthetic_dataset_dir):
    class_names = discover_class_names(synthetic_dataset_dir)
    assert class_names == sorted(class_names)
    assert class_names == ["glioma", "meningioma", "notumor", "pituitary"]


def test_discover_class_names_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_class_names(tmp_path / "does_not_exist")


def test_discover_class_names_empty_directory_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        discover_class_names(empty_dir)


def test_load_paths_and_labels_finds_all_images(synthetic_dataset_dir):
    class_names = discover_class_names(synthetic_dataset_dir)
    paths, labels = load_paths_and_labels(synthetic_dataset_dir, class_names)

    assert len(paths) == 12  # 4 classes x 3 images
    assert len(labels) == 12
    assert set(labels) == set(class_names)


def test_encode_labels_matches_class_name_index(synthetic_dataset_dir):
    class_names = discover_class_names(synthetic_dataset_dir)
    labels = ["notumor", "glioma", "pituitary"]

    encoded = encode_labels(labels, class_names)

    # This is the exact bug from the original notebook: encoding must
    # always match discover_class_names' sorted order, everywhere.
    assert encoded.tolist() == [class_names.index(lbl) for lbl in labels]


def test_encode_labels_unknown_label_raises():
    with pytest.raises(ValueError):
        encode_labels(["unknown_class"], ["glioma", "meningioma"])


def test_train_validation_split_no_overlap(synthetic_dataset_dir):
    class_names = discover_class_names(synthetic_dataset_dir)
    paths, labels = load_paths_and_labels(synthetic_dataset_dir, class_names)

    train_paths, train_labels, val_paths, val_labels = train_validation_split(
        paths, labels, validation_split=0.34, seed=42
    )

    assert set(train_paths).isdisjoint(set(val_paths))
    assert len(train_paths) + len(val_paths) == len(paths)
    assert len(train_labels) == len(train_paths)
    assert len(val_labels) == len(val_paths)


def test_train_validation_split_is_deterministic(synthetic_dataset_dir):
    class_names = discover_class_names(synthetic_dataset_dir)
    paths, labels = load_paths_and_labels(synthetic_dataset_dir, class_names)

    split_a = train_validation_split(paths, labels, validation_split=0.34, seed=42)
    split_b = train_validation_split(paths, labels, validation_split=0.34, seed=42)

    assert split_a == split_b


def test_train_validation_split_invalid_fraction_raises(synthetic_dataset_dir):
    class_names = discover_class_names(synthetic_dataset_dir)
    paths, labels = load_paths_and_labels(synthetic_dataset_dir, class_names)

    with pytest.raises(ValueError):
        train_validation_split(paths, labels, validation_split=1.5, seed=42)

    with pytest.raises(ValueError):
        train_validation_split(paths, labels, validation_split=0.0, seed=42)


def test_train_validation_split_preserves_class_balance(synthetic_dataset_dir):
    class_names = discover_class_names(synthetic_dataset_dir)
    paths, labels = load_paths_and_labels(synthetic_dataset_dir, class_names)

    _, _, val_paths, val_labels = train_validation_split(
        paths, labels, validation_split=0.34, seed=42
    )

    # Stratified split: every class should have at least one validation
    # sample, not be entirely absent (a risk with naive random splits
    # on small datasets).
    assert set(val_labels) == set(class_names)
