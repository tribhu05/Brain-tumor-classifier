"""
plots.py
========

Plotting utilities, consolidated from four separate notebook cells.

Fix relative to the original notebook
----------------------------------------
The original sample-image cell called ``plt.tight_layout`` without
parentheses -- a no-op that never actually applied the layout fix (a
real bug, not a style choice). Every call site below calls
``plt.tight_layout()`` correctly.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_sample_images(
    paths: Sequence[str],
    labels: Sequence[str],
    n: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """Display a grid of random sample images with their labels.

    Args:
        paths: Image file paths.
        labels: Corresponding labels (parallel to ``paths``).
        n: Number of sample images to display.
        save_path: If given, save the figure to this path instead of
            (or in addition to) showing it interactively.
    """
    indices = random.sample(range(len(paths)), min(n, len(paths)))
    cols = min(5, n)
    rows = (len(indices) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).ravel()

    for ax, idx in zip(axes, indices):
        image = plt.imread(paths[idx])
        ax.imshow(image)
        ax.set_title(labels[idx])
        ax.axis("off")
    for ax in axes[len(indices) :]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_history(history, save_path: Optional[str] = None) -> None:
    """Plot training/validation accuracy and loss curves.

    Args:
        history: A Keras ``History`` object (or an object/dict exposing
            a ``.history`` dict with keys like
            ``sparse_categorical_accuracy``, ``val_sparse_categorical_accuracy``,
            ``loss``, ``val_loss``).
        save_path: Optional path to save the figure to.
    """
    hist = history.history if hasattr(history, "history") else history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(hist.get("sparse_categorical_accuracy", []), label="train", color="green")
    if "val_sparse_categorical_accuracy" in hist:
        axes[0].plot(hist["val_sparse_categorical_accuracy"], label="validation", color="orange")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(hist.get("loss", []), label="train", color="red")
    if "val_loss" in hist:
        axes[1].plot(hist["val_loss"], label="validation", color="orange")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Sequence[str],
    save_path: Optional[str] = None,
) -> None:
    """Plot a confusion matrix heatmap with correctly matched axis labels.

    Unlike the original notebook (which used mismatched sorted/unsorted
    label sources -- see ``data/dataset.py`` docstring), ``class_names``
    here must be the same ordered list used everywhere else in the
    pipeline, guaranteeing correct axis labels.

    Args:
        confusion_matrix: A ``(num_classes, num_classes)`` integer array.
        class_names: Ordered class names matching the matrix's rows/columns.
        save_path: Optional path to save the figure to.
    """
    import seaborn as sns

    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
