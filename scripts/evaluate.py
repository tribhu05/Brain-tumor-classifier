#!/usr/bin/env python3
"""
scripts/evaluate.py
====================

CLI entrypoint for evaluating a trained model on the held-out test set.

Usage:
    python scripts/evaluate.py --config configs/config.yaml \\
        --model-path artifacts/checkpoints/best_model.keras
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_tumor_classifier.config import load_config
from brain_tumor_classifier.data.dataset import (
    build_tf_dataset,
    discover_class_names,
    load_paths_and_labels,
)
from brain_tumor_classifier.evaluation.evaluate import evaluate_model
from brain_tumor_classifier.utils.logging_config import configure_logging
from brain_tumor_classifier.visualization.plots import plot_confusion_matrix

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained brain tumor classifier.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/checkpoints/best_model.keras",
        help="Path to a saved .keras model (defaults to artifacts/checkpoints/best_model.keras or assets/best_model.keras).",
    )
    parser.add_argument(
        "--save-confusion-matrix",
        type=str,
        default="artifacts/evaluation/confusion_matrix.png",
        help="Where to save the confusion matrix plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    import tensorflow as tf

    config = load_config(args.config)

    # Check test dataset directory
    test_dir = Path(config.data.test_dir)
    if not test_dir.exists() or not any(p.is_dir() for p in test_dir.iterdir()):
        print(f"\n[ERROR] Test dataset directory not found or empty at '{config.data.test_dir}'.")
        print(
            "To evaluate the model, place your test data in data/raw/Testing/ (with subdirectories: glioma, meningioma, notumor, pituitary)"
        )
        print("or edit configs/config.yaml to point to your testing dataset location.\n")
        sys.exit(1)

    # Resolve class names
    train_dir = Path(config.data.train_dir)
    if train_dir.exists() and any(p.is_dir() for p in train_dir.iterdir()):
        class_names = discover_class_names(train_dir)
    elif test_dir.exists() and any(p.is_dir() for p in test_dir.iterdir()):
        class_names = discover_class_names(test_dir)
    else:
        class_names = getattr(
            config.data, "class_names", ["glioma", "meningioma", "notumor", "pituitary"]
        )

    # Resolve model path with fallbacks
    model_path = Path(args.model_path)
    if not model_path.exists():
        fallback_candidates = [
            Path("assets/best_model.keras"),
            Path("brain-tumor-classifier/assets/best_model.keras"),
            Path("../assets/best_model.keras"),
        ]
        for candidate in fallback_candidates:
            if candidate.exists():
                logger.info(
                    "Model not found at %s. Using bundled model at %s", model_path, candidate
                )
                model_path = candidate
                break
        else:
            print(f"\n[ERROR] Model file not found at: {args.model_path}")
            print(
                "Please provide a valid --model-path or place a model at assets/best_model.keras\n"
            )
            sys.exit(1)

    test_paths, test_labels = load_paths_and_labels(config.data.test_dir, class_names)
    test_dataset = build_tf_dataset(
        test_paths,
        test_labels,
        class_names,
        image_size=config.data.image_size,
        batch_size=config.training.batch_size,
        augment=False,  # true test-set evaluation -- never augment (fixes leakage bug)
        shuffle=False,
    )

    model = tf.keras.models.load_model(str(model_path))
    result = evaluate_model(model, test_dataset, class_names)

    print("\nClassification Report\n" + "=" * 60)
    print(result.classification_report)
    print("Per-class ROC-AUC:")
    for class_name, auc_value in result.roc_auc_per_class.items():
        print(f"  {class_name:>12}: {auc_value:.4f}")
    print(f"\nMacro-average ROC-AUC: {result.macro_auc:.4f}")

    plot_confusion_matrix(
        result.confusion_matrix, class_names, save_path=args.save_confusion_matrix
    )
    logger.info("Confusion matrix saved to %s", args.save_confusion_matrix)


if __name__ == "__main__":
    main()
