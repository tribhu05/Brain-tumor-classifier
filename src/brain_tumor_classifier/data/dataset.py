"""
dataset.py
==========

Dataset discovery, splitting, and ``tf.data`` pipeline construction.

Fixes relative to the original notebook
----------------------------------------
1. **Label-order mismatch (bug):** the original notebook encoded labels
   using ``sorted(os.listdir(train_dir))`` but plotted the confusion
   matrix using the *unsorted* ``os.listdir(train_dir)`` as axis labels.
   On filesystems where directory listing order isn't alphabetical,
   this silently mislabels the confusion matrix. Here, class names are
   discovered exactly once via :func:`discover_class_names` (which
   sorts) and threaded through every downstream consumer, so there is
   only one source of truth.
2. **No validation split (bug):** the original ``model.fit()`` call had
   no ``validation_data``, so there was no way to detect overfitting
   during training. :func:`train_validation_split` adds a proper,
   seeded, stratification-aware split.
3. **Dropped remainder batch (bug):** ``steps = int(len(train_path) /
   batch_size)`` silently discarded up to ``batch_size - 1`` training
   images every epoch. The ``tf.data`` pipeline here uses
   ``drop_remainder=False`` by default, so no data is discarded.
4. **Slow hand-rolled Python generator (perf smell):** replaced with a
   ``tf.data.Dataset`` pipeline using parallel map + prefetch, which
   overlaps image decode/augmentation with GPU compute and reshuffles
   every epoch (the original generator iterated the same fixed list
   order every epoch).
5. **Hardcoded personal paths (portability):** paths now come from
   :class:`~brain_tumor_classifier.config.DataConfig`, not a literal
   Windows path baked into the notebook.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def discover_class_names(directory: str | Path) -> List[str]:
    """Discover class names from subdirectory names, alphabetically sorted.

    Sorting is what guarantees the same label <-> integer mapping is
    used everywhere (training, evaluation, inference, plotting), fixing
    the label-order mismatch bug described in the module docstring.

    Args:
        directory: A directory containing one subdirectory per class.

    Returns:
        Sorted list of class names.

    Raises:
        FileNotFoundError: If ``directory`` does not exist.
        ValueError: If ``directory`` contains no subdirectories.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    class_names = sorted(p.name for p in directory.iterdir() if p.is_dir())
    if not class_names:
        raise ValueError(f"No class subdirectories found under: {directory}")

    return class_names


def load_paths_and_labels(
    directory: str | Path,
    class_names: Sequence[str],
    valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> Tuple[List[str], List[str]]:
    """Walk a class-subdirectory dataset layout and collect image paths + labels.

    Args:
        directory: A directory containing one subdirectory per class
            (e.g. ``Training/glioma/*.jpg``).
        class_names: The class names to include, in the order that
            defines the label encoding (use :func:`discover_class_names`
            to get a canonical, sorted list).
        valid_extensions: File extensions to treat as images; anything
            else in the directory is skipped rather than crashing.

    Returns:
        A tuple ``(paths, labels)`` of parallel lists: file paths and
        their corresponding class-name strings (not yet integer-encoded).
    """
    directory = Path(directory)
    paths: List[str] = []
    labels: List[str] = []

    for class_name in class_names:
        class_dir = directory / class_name
        if not class_dir.is_dir():
            logger.warning("Expected class directory missing: %s", class_dir)
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in valid_extensions:
                paths.append(str(image_path))
                labels.append(class_name)

    logger.info(
        "Loaded %d images across %d classes from %s", len(paths), len(class_names), directory
    )
    return paths, labels


def encode_labels(labels: Sequence[str], class_names: Sequence[str]) -> np.ndarray:
    """Convert string labels to integer indices using a fixed class ordering.

    Args:
        labels: Class-name strings to encode.
        class_names: The canonical, ordered list of class names (the
            index of each name in this list is its integer label).

    Returns:
        A 1-D int array of encoded labels.

    Raises:
        ValueError: If a label is not present in ``class_names``.
    """
    index_lookup = {name: idx for idx, name in enumerate(class_names)}
    try:
        return np.array([index_lookup[label] for label in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Label {exc} not found in class_names={list(class_names)}") from exc


def train_validation_split(
    paths: Sequence[str],
    labels: Sequence[str],
    validation_split: float,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Stratified train/validation split, seeded for reproducibility.

    The original notebook had no validation split at all (see module
    docstring, fix #2). Stratification keeps class proportions
    consistent between the train and validation subsets.

    Args:
        paths: Image file paths.
        labels: Corresponding class-name labels (parallel to ``paths``).
        validation_split: Fraction of data to hold out for validation,
            in ``(0, 1)``.
        seed: Random seed controlling the split, for reproducibility.

    Returns:
        ``(train_paths, train_labels, val_paths, val_labels)``.

    Raises:
        ValueError: If ``validation_split`` is not in ``(0, 1)``.
    """
    if not 0.0 < validation_split < 1.0:
        raise ValueError(f"validation_split must be in (0, 1), got {validation_split}")

    rng = np.random.RandomState(seed)
    paths_arr = np.array(paths)
    labels_arr = np.array(labels)

    train_idx: List[int] = []
    val_idx: List[int] = []

    # Stratify: split each class independently so the val set mirrors
    # the overall class distribution instead of risking a class being
    # entirely absent from validation on a small dataset.
    for class_name in sorted(set(labels)):
        class_indices = np.where(labels_arr == class_name)[0]
        rng.shuffle(class_indices)
        n_val = max(1, int(round(len(class_indices) * validation_split)))
        val_idx.extend(class_indices[:n_val].tolist())
        train_idx.extend(class_indices[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return (
        paths_arr[train_idx].tolist(),
        labels_arr[train_idx].tolist(),
        paths_arr[val_idx].tolist(),
        labels_arr[val_idx].tolist(),
    )


def build_tf_dataset(
    paths: Sequence[str],
    labels: Sequence[str],
    class_names: Sequence[str],
    image_size: int,
    batch_size: int,
    augment: bool = False,
    shuffle: bool = False,
    seed: int = 42,
):
    """Build a ``tf.data.Dataset`` that yields ``(image_batch, label_batch)``.

    Replaces the original hand-rolled Python generator (``datagen``)
    with a proper ``tf.data`` pipeline: parallel image decode via
    ``num_parallel_calls=AUTOTUNE``, ``prefetch(AUTOTUNE)`` to overlap
    I/O with training, per-epoch reshuffling when ``shuffle=True``, and
    ``drop_remainder=False`` so no images are silently discarded.

    Args:
        paths: Image file paths.
        labels: Corresponding class-name labels (parallel to ``paths``).
        class_names: Canonical, ordered class name list (see
            :func:`discover_class_names`).
        image_size: Target square size images are resized to.
        batch_size: Batch size.
        augment: If True, apply train-only random brightness/contrast
            jitter (see ``augmentation.py``). Must be False for
            validation/test/inference data -- this is what fixes the
            test-leakage bug described in ``augmentation.py``.
        shuffle: If True, shuffle and reshuffle every epoch. Should be
            True for training data and False for validation/test data
            (so evaluation order is stable and reproducible).
        seed: Random seed for shuffling.

    Returns:
        A batched, prefetched ``tf.data.Dataset``.
    """
    import tensorflow as tf

    from .augmentation import augment_image, rescale_only

    encoded_labels = encode_labels(labels, class_names)

    def _load_and_preprocess(path: tf.Tensor, label: tf.Tensor):
        def _py_load(path_bytes):
            image = tf.keras.utils.load_img(
                path_bytes.numpy().decode("utf-8"),
                target_size=(image_size, image_size),
            )
            image_arr = tf.keras.utils.img_to_array(image)
            processed = augment_image(image_arr) if augment else rescale_only(image_arr)
            return processed.astype(np.float32)

        image = tf.py_function(_py_load, [path], tf.float32)
        image.set_shape((image_size, image_size, 3))
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((list(paths), encoded_labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
