#!/usr/bin/env python3
"""
scripts/train.py
=================

CLI entrypoint for training the brain tumor classifier.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --epochs 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_tumor_classifier.config import load_config
from brain_tumor_classifier.training.train import train
from brain_tumor_classifier.utils.logging_config import configure_logging
from brain_tumor_classifier.visualization.plots import plot_training_history

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the brain tumor MRI classifier.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epoch count from config."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size from config."
    )
    parser.add_argument(
        "--log-file", type=str, default="artifacts/logs/train.log", help="Path to write logs."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(log_file=args.log_file)

    config = load_config(args.config)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    # Validate dataset existence
    train_dir = Path(config.data.train_dir)
    if not train_dir.exists() or not any(p.is_dir() for p in train_dir.iterdir()):
        print(f"\n[ERROR] Training dataset not found or empty at: {train_dir.resolve()}")
        print("To train a new model, please download the Brain Tumor MRI dataset and place it in:")
        print(f"  {train_dir.resolve()}\\<class_name>\\*.jpg")
        print("Expected class subfolders: glioma, meningioma, notumor, pituitary")
        print("Or edit 'configs/config.yaml' to point to your custom dataset location.\n")
        sys.exit(1)

    logger.info("Starting training with config: %s", args.config)
    _model, history, class_names = train(config)

    final_train_acc = history.history["sparse_categorical_accuracy"][-1]
    final_val_acc = history.history.get("val_sparse_categorical_accuracy", [None])[-1]
    logger.info(
        "Training finished. Final train acc: %.4f | Final val acc: %s | Classes: %s",
        final_train_acc,
        f"{final_val_acc:.4f}" if final_val_acc is not None else "N/A",
        class_names,
    )

    # Save training curves plot
    plot_save_path = "artifacts/plots/training_history.png"
    plot_training_history(history, save_path=plot_save_path)
    logger.info("Training history plot saved to %s", plot_save_path)
    if Path("assets").exists():
        plot_training_history(history, save_path="assets/training_history.png")


if __name__ == "__main__":
    main()
