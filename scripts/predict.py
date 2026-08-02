#!/usr/bin/env python3
"""
scripts/predict.py
====================

CLI entrypoint for running inference on a single MRI image.

Usage:
    python scripts/predict.py --model-path artifacts/checkpoints/best_model.keras \\
        --image path/to/scan.jpg
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_tumor_classifier.config import load_config
from brain_tumor_classifier.data.dataset import discover_class_names
from brain_tumor_classifier.inference.predict import predict_image
from brain_tumor_classifier.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict tumor class for a single MRI image.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a saved .keras model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the MRI image to classify.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    import tensorflow as tf

    config = load_config(args.config)
    class_names = discover_class_names(config.data.train_dir)
    model = tf.keras.models.load_model(args.model_path)

    result = predict_image(args.image, model, class_names, image_size=config.data.image_size)

    label = "No Tumor" if not result.is_tumor else f"Tumor: {result.predicted_class}"
    print(f"\n{label} (confidence: {result.confidence * 100:.2f}%)")
    print("\nPer-class probabilities:")
    for class_name, prob in sorted(result.class_probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {class_name:>12}: {prob * 100:5.2f}%")


if __name__ == "__main__":
    main()
