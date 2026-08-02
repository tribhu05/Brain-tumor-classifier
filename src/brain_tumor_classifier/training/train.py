"""
train.py
========

Training orchestration.

Fixes relative to the original notebook
----------------------------------------
- Adds a real validation set and passes it to ``model.fit(validation_data=...)``.
  The original had none, so overfitting was invisible during training.
- Adds ``ModelCheckpoint`` (saves the best model by validation accuracy),
  ``EarlyStopping`` (configurable, stops wasting compute once validation
  stops improving), and ``CSVLogger`` (persists per-epoch metrics to
  disk instead of only living in the notebook's `history` variable).
- Saves the final model in the modern ``.keras`` format instead of the
  legacy ``.h5`` format, with a version tag in the filename.
- Seeds all randomness up front via ``utils.seed.set_global_seed``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

from brain_tumor_classifier.config import Config
from brain_tumor_classifier.data.dataset import (
    build_tf_dataset,
    discover_class_names,
    load_paths_and_labels,
    train_validation_split,
)
from brain_tumor_classifier.models.vgg16_classifier import build_model
from brain_tumor_classifier.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def train(config: Config) -> Tuple[object, object, list]:
    """Run the full training pipeline: data -> model -> fit -> save.

    Args:
        config: A populated :class:`~brain_tumor_classifier.config.Config`.

    Returns:
        A tuple ``(model, history, class_names)``:
            - ``model``: the trained ``tf.keras.Model``.
            - ``history``: the Keras ``History`` object from ``model.fit``.
            - ``class_names``: the sorted list of class names used for
              label encoding, needed by evaluation/inference to decode
              predictions consistently.
    """
    set_global_seed(config.training.seed)

    class_names = discover_class_names(config.data.train_dir)
    logger.info("Discovered classes: %s", class_names)

    all_paths, all_labels = load_paths_and_labels(config.data.train_dir, class_names)
    train_paths, train_labels, val_paths, val_labels = train_validation_split(
        all_paths,
        all_labels,
        validation_split=config.data.validation_split,
        seed=config.training.seed,
    )
    logger.info(
        "Split: %d train / %d validation images", len(train_paths), len(val_paths)
    )

    train_dataset = build_tf_dataset(
        train_paths,
        train_labels,
        class_names,
        image_size=config.data.image_size,
        batch_size=config.training.batch_size,
        augment=True,
        shuffle=True,
        seed=config.training.seed,
    )
    val_dataset = build_tf_dataset(
        val_paths,
        val_labels,
        class_names,
        image_size=config.data.image_size,
        batch_size=config.training.batch_size,
        augment=False,  # never augment validation data -- see augmentation.py
        shuffle=False,
        seed=config.training.seed,
    )

    model = build_model(
        image_size=config.data.image_size,
        num_classes=config.model.num_classes,
        learning_rate=config.training.learning_rate,
        dropout_1=config.model.dropout_1,
        dense_units=config.model.dense_units,
        dropout_2=config.model.dropout_2,
        trainable_backbone_layers=config.model.trainable_backbone_layers,
    )
    model.summary(print_fn=logger.info)

    callbacks = _build_callbacks(config)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.training.epochs,
        callbacks=callbacks,
    )

    final_model_path = _save_final_model(model, config)
    logger.info("Training complete. Final model saved to %s", final_model_path)

    return model, history, class_names


def _build_callbacks(config: Config) -> list:
    """Construct the Keras callback list: checkpointing, early stopping, CSV logging."""
    from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint

    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config.training.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model.keras"),
            monitor="val_sparse_categorical_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        CSVLogger(str(log_dir / "training_history.csv")),
    ]

    if config.training.early_stopping_patience:
        callbacks.append(
            EarlyStopping(
                monitor="val_sparse_categorical_accuracy",
                patience=config.training.early_stopping_patience,
                restore_best_weights=True,
                mode="max",
                verbose=1,
            )
        )

    return callbacks


def _save_final_model(model, config: Config) -> str:
    """Save the trained model in the modern .keras format with a timestamp tag."""
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = checkpoint_dir / f"brain_tumor_classifier_{timestamp}.keras"
    model.save(final_path)
    return str(final_path)
