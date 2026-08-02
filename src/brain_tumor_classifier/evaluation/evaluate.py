"""
evaluate.py
===========

Model evaluation on held-out data.

Fixes / additions relative to the original notebook
------------------------------------------------------
- The original notebook computed metrics on **augmented** test images
  (see ``augmentation.py`` docstring for why that's a leakage bug). This
  module's :func:`evaluate_model` is designed to be called with a
  ``tf.data.Dataset`` built via ``build_tf_dataset(..., augment=False)``,
  so evaluation is always performed on the true, unmodified test set.
- The original notebook imported ``roc_curve``, ``auc``, and
  ``label_binarize`` from sklearn but never actually used them --
  dead imports suggesting unfinished work. This module implements the
  per-class ROC/AUC computation those imports were presumably intended
  for, satisfying the "more evaluation metrics" improvement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for all computed evaluation metrics."""

    class_names: List[str]
    classification_report: str
    confusion_matrix: np.ndarray
    roc_auc_per_class: Dict[str, float]
    macro_auc: float


def evaluate_model(model, dataset, class_names: Sequence[str]) -> EvaluationResult:
    """Compute classification report, confusion matrix, and per-class ROC-AUC.

    Args:
        model: A trained ``tf.keras.Model``.
        dataset: A ``tf.data.Dataset`` of ``(image_batch, label_batch)``,
            built with ``augment=False`` (validation/test data must
            never be augmented -- see module docstring).
        class_names: Canonical, ordered class name list matching the
            integer labels in ``dataset``.

    Returns:
        An :class:`EvaluationResult` with all computed metrics.
    """
    from sklearn.metrics import confusion_matrix, classification_report as sk_report
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    y_true: List[int] = []
    y_pred_probs: List[np.ndarray] = []

    for image_batch, label_batch in dataset:
        batch_probs = model.predict(image_batch, verbose=0)
        y_pred_probs.append(batch_probs)
        y_true.extend(label_batch.numpy().tolist())

    y_true_arr = np.array(y_true)
    y_pred_probs_arr = np.vstack(y_pred_probs)
    y_pred_labels = np.argmax(y_pred_probs_arr, axis=1)

    report = sk_report(
        y_true_arr, y_pred_labels, target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(y_true_arr, y_pred_labels)

    # Per-class ROC-AUC (one-vs-rest), implementing what the original
    # notebook's unused roc_curve/auc/label_binarize imports hinted at.
    roc_auc_per_class: Dict[str, float] = {}
    if len(class_names) > 2:
        y_true_binarized = label_binarize(y_true_arr, classes=list(range(len(class_names))))
        for idx, class_name in enumerate(class_names):
            try:
                roc_auc_per_class[class_name] = float(
                    roc_auc_score(y_true_binarized[:, idx], y_pred_probs_arr[:, idx])
                )
            except ValueError:
                # Raised if a class is absent from y_true in this batch/split.
                logger.warning("Could not compute ROC-AUC for class '%s' (class absent from data).", class_name)
                roc_auc_per_class[class_name] = float("nan")
        macro_auc = float(np.nanmean(list(roc_auc_per_class.values())))
    else:
        auc_value = float(roc_auc_score(y_true_arr, y_pred_probs_arr[:, 1]))
        roc_auc_per_class = {class_names[1]: auc_value}
        macro_auc = auc_value

    logger.info("Evaluation complete. Macro-average ROC-AUC: %.4f", macro_auc)

    return EvaluationResult(
        class_names=list(class_names),
        classification_report=report,
        confusion_matrix=cm,
        roc_auc_per_class=roc_auc_per_class,
        macro_auc=macro_auc,
    )
