"""
brain_tumor_classifier
=======================

A VGG16 transfer-learning pipeline for classifying brain MRI scans into
four categories: glioma, meningioma, pituitary tumor, or no tumor.

Public API
----------
- config.Config            : typed, YAML-driven configuration
- data.dataset              : dataset discovery, splitting, tf.data pipelines
- data.augmentation         : train-time-only image augmentation
- models.vgg16_classifier   : model architecture definition
- training.train             : training loop with callbacks
- evaluation.evaluate         : metrics computation (report, confusion matrix, ROC/AUC)
- inference.predict           : single-image prediction
- visualization.plots         : shared plotting utilities
"""

__version__ = "1.0.0"
