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
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/checkpoints/best_model.keras",
        help="Path to a saved .keras model (defaults to artifacts/checkpoints/best_model.keras or assets/best_model.keras).",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the MRI image to classify.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    import tensorflow as tf

    config = load_config(args.config)

    # Resolve class names: discover from train_dir if present, otherwise use config defaults
    train_dir = Path(config.data.train_dir)
    if train_dir.exists() and any(p.is_dir() for p in train_dir.iterdir()):
        class_names = discover_class_names(train_dir)
    else:
        class_names = getattr(config.data, "class_names", ["glioma", "meningioma", "notumor", "pituitary"])

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
                logger.info("Model not found at %s. Using bundled model at %s", model_path, candidate)
                model_path = candidate
                break
        else:
            print(f"\n[ERROR] Model file not found at: {args.model_path}")
            print("Please provide a valid --model-path or place a model at assets/best_model.keras\n")
            sys.exit(1)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"\n[ERROR] Image not found at: {args.image}")
        print("Please check the image path and try again.\n")
        sys.exit(1)

    model = tf.keras.models.load_model(str(model_path))
    result = predict_image(str(image_path), model, class_names, image_size=config.data.image_size)

    label = "No Tumor" if not result.is_tumor else f"Tumor: {result.predicted_class}"
    print(f"\nPrediction Result:\n" + "=" * 40)
    print(f"Status: {label}")
    print(f"Confidence: {result.confidence * 100:.2f}%")
    print("\nPer-class probabilities:")
    for class_name, prob in sorted(result.class_probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {class_name:>12}: {prob * 100:5.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
