<div align="center">

# 🧠 Brain Tumor MRI Classifier

### VGG16 Transfer Learning for 4-Class Brain Tumor Detection

<!-- Replace with your own banner image at assets/banner.png -->
<!-- ![Banner](assets/banner.png) -->

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/tribhu05/brain-tumor-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/tribhu05/brain-tumor-classifier/actions/workflows/ci.yml)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub stars](https://img.shields.io/github/stars/tribhu05/brain-tumor-classifier?style=social)](https://github.com/tribhu05/brain-tumor-classifier/stargazers)

[Overview](#overview) •
[Features](#features) •
[Installation](#installation) •
[Usage](#usage) •
[Architecture](#model-architecture) •
[Results](#performance-metrics) •
[Contributing](#contributing)

</div>

---

## Overview

### Problem Statement

Brain tumors (glioma, meningioma, and pituitary tumors) require accurate,
timely identification from MRI scans. Manual review is time-consuming and
subject to inter-radiologist variability. This project explores whether a
transfer-learning approach using a pretrained VGG16 backbone can reliably
classify axial brain MRI slices into one of four categories: **glioma**,
**meningioma**, **pituitary tumor**, or **no tumor**.

### Motivation

Training a convolutional network from scratch on a relatively small medical
imaging dataset is prone to overfitting. Transfer learning — reusing
ImageNet-pretrained convolutional filters and fine-tuning only the deepest
layers — is a standard, well-justified technique for this kind of
data-constrained image classification problem. This repo implements that
approach end-to-end: data pipeline, model, training loop, evaluation, and
inference, structured as a maintainable Python package rather than a single
notebook.

> **Note:** This is a research/portfolio project, not a validated clinical
> tool. See [`docs/known_limitations.md`](docs/known_limitations.md) for an
> honest breakdown of what has and hasn't been validated.

## Features

- 🏗️ **Modular architecture** — clean separation between data loading,
  model definition, training, evaluation, and inference (see
  [`docs/architecture.md`](docs/architecture.md)).
- ⚙️ **Config-driven** — every hyperparameter lives in
  [`configs/config.yaml`](configs/config.yaml), no magic numbers in code.
- 🔁 **Reproducible** — global seeding across Python/NumPy/TensorFlow.
- 📊 **Real evaluation** — held-out validation split during training, plus
  classification report, confusion matrix, and per-class ROC-AUC on the
  test set (never computed on augmented data — see
  [`CHANGELOG.md`](CHANGELOG.md)).
- 🚦 **Training callbacks** — checkpointing (best model by validation
  accuracy), early stopping, CSV metric logging.
- 🧪 **Tested** — pytest suite using synthetic data, no dependency on the
  real dataset to run in CI.
- 🐳 **Dockerized** — reproducible training/inference environment.
- 🖥️ **CLI scripts** — `train.py`, `evaluate.py`, `predict.py`.

## Tech Stack

| Layer | Tools |
|---|---|
| Modeling | TensorFlow / Keras, VGG16 (ImageNet weights) |
| Data | `tf.data`, Pillow, NumPy |
| Evaluation | scikit-learn (classification report, confusion matrix, ROC-AUC) |
| Visualization | Matplotlib, Seaborn |
| Config | PyYAML, Python dataclasses |
| Testing | pytest, pytest-cov |
| Tooling | Black, isort, flake8, mypy |
| CI/CD | GitHub Actions |
| Packaging | Docker, `pyproject.toml` / setuptools |

## Model Architecture

```
Input (128×128×3)
      │
      ▼
VGG16 backbone (ImageNet weights, include_top=False)
  - all layers frozen EXCEPT:
      block5_conv1, block5_conv2, block5_conv3  (fine-tuned)
      │
      ▼
Flatten
      │
      ▼
Dropout(0.3)
      │
      ▼
Dense(128, activation="relu")
      │
      ▼
Dropout(0.2)
      │
      ▼
Dense(4, activation="softmax")   →  [glioma, meningioma, notumor, pituitary]
```

- **Optimizer:** Adam, learning rate `1e-4`
- **Loss:** sparse categorical crossentropy
- **Metric:** sparse categorical accuracy

Full rationale for these choices — and what was deliberately *not* changed
during the refactor — is in [`docs/architecture.md`](docs/architecture.md)
and [`docs/known_limitations.md`](docs/known_limitations.md).

## Dataset

Expects the standard Kaggle-style "Brain Tumor MRI Dataset" layout: one
subdirectory per class, containing JPEG images.

```
data/raw/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Point `configs/config.yaml`'s `data.train_dir` / `data.test_dir` (or the
`BTC_TRAIN_DIR` / `BTC_TEST_DIR` environment variables) at your local copy.
**Raw MRI images are not committed to this repository** — see `.gitignore`.

## Data Pipeline

1. `discover_class_names()` — scans `train_dir` subfolders, sorted
   alphabetically. This sorted order is the single source of truth for
   label encoding used everywhere downstream.
2. `load_paths_and_labels()` — collects image file paths per class.
3. `train_validation_split()` — seeded, stratified split (default 85/15).
4. `build_tf_dataset()` — builds a batched, prefetched `tf.data.Dataset`
   with parallel image decode. Training data is randomly
   brightness/contrast-augmented; validation and test data are not.

## Training Pipeline

```bash
python scripts/train.py --config configs/config.yaml
```

- Loads config, seeds all RNGs, discovers classes.
- Builds train/validation `tf.data` pipelines.
- Builds and compiles the VGG16-based model.
- Trains with `ModelCheckpoint` (saves best model by validation accuracy),
  `EarlyStopping` (patience configurable), and `CSVLogger`.
- Saves the final model to `artifacts/checkpoints/` in `.keras` format.

## Evaluation

```bash
python scripts/evaluate.py --config configs/config.yaml \
    --model-path artifacts/checkpoints/best_model.keras
```

Produces a classification report (precision/recall/F1 per class),
confusion matrix (saved as an image), and per-class + macro-average
ROC-AUC — computed strictly on the unaugmented test set.

## Performance Metrics

Trained for 10 epochs (best checkpoint at epoch 7) on a T4 GPU. Evaluated
on a held-out test set of 1,600 images (400 per class), never seen during
training or validation.

| Class | Precision | Recall | F1-score | ROC-AUC | Support |
|---|---|---|---|---|---|
| glioma | 0.99 | 0.70 | 0.82 | 0.9204 | 400 |
| meningioma | 0.83 | 0.96 | 0.89 | 0.9822 | 400 |
| notumor | 0.84 | 1.00 | 0.91 | 0.9962 | 400 |
| pituitary | 1.00 | 0.94 | 0.97 | 0.9982 | 400 |
| **Macro avg** | **0.91** | **0.90** | **0.90** | **0.9742** | 1600 |

**Overall test accuracy: 90%**

Best validation accuracy during training: **94.76%** (epoch 7), after which
`EarlyStopping` restored the best weights rather than the final epoch's.

![Confusion Matrix](assets/confusion_matrix.png)
![Training Curves](assets/training_history.png)

### A note on glioma recall

Glioma has the highest precision (0.99) but the lowest recall (0.70) of
the four classes — meaning that when the model predicts glioma it's
almost always correct, but it misses roughly 30% of actual glioma cases
(most likely misclassifying them as meningioma, based on the confusion
matrix). This is the model's clearest weak point and a reasonable next
thing to investigate — see [Future Improvements](#future-improvements).

## Screenshots

Sample training run (10 epochs, T4 GPU) and evaluation output are in
[`docs/known_limitations.md`](docs/known_limitations.md) and the
Performance Metrics section above. See `assets/` for the confusion
matrix and training curve plots.

## Installation

### Option 1: pip

```bash
git clone https://github.com/tribhu05/brain-tumor-classifier.git
cd brain-tumor-classifier
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Option 2: Docker

```bash
docker build -t brain-tumor-classifier .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts \
    brain-tumor-classifier --config configs/config.yaml
```

## Usage

```bash
# Train
python scripts/train.py --config configs/config.yaml

# Evaluate
python scripts/evaluate.py --config configs/config.yaml \
    --model-path artifacts/checkpoints/best_model.keras

# Predict on a single image
python scripts/predict.py --config configs/config.yaml \
    --model-path artifacts/checkpoints/best_model.keras \
    --image path/to/scan.jpg
```

### Programmatic usage

```python
from brain_tumor_classifier.config import load_config
from brain_tumor_classifier.data.dataset import discover_class_names
from brain_tumor_classifier.inference.predict import predict_image
import tensorflow as tf

config = load_config("configs/config.yaml")
class_names = discover_class_names(config.data.train_dir)
model = tf.keras.models.load_model("artifacts/checkpoints/best_model.keras")

result = predict_image("scan.jpg", model, class_names, image_size=config.data.image_size)
print(result.predicted_class, result.confidence)
```

See [`examples/quickstart.md`](examples/quickstart.md) for a fuller walkthrough.

## Folder Structure

```
brain-tumor-classifier/
├── src/brain_tumor_classifier/   # installable package
│   ├── config.py
│   ├── data/                      # dataset + augmentation
│   ├── models/                    # architecture definition
│   ├── training/                  # training orchestration
│   ├── evaluation/                # metrics
│   ├── inference/                 # single-image prediction
│   ├── visualization/             # plotting
│   └── utils/                     # seeding, logging
├── tests/                          # pytest, synthetic data
├── scripts/                        # CLI entrypoints
├── configs/config.yaml
├── notebooks/archive/               # original prototype notebook (reference)
├── docs/                            # architecture + known limitations
├── .github/                         # CI, issue/PR templates
├── Dockerfile
├── Makefile
└── requirements.txt
```

## Future Improvements

- [ ] Investigate glioma's low recall (0.70 vs. 0.83–1.00 for other
      classes) — likely candidates: class-specific augmentation,
      per-class decision threshold tuning, or examining
      glioma/meningioma misclassifications directly in the confusion matrix
- [ ] Switch to `vgg16.preprocess_input` for correct ImageNet-distribution
      preprocessing (see `docs/known_limitations.md`)
- [ ] Class-weighted loss / oversampling if class imbalance is present
- [ ] K-fold cross-validation for more robust performance estimates
- [ ] Grad-CAM visualization for model interpretability
- [ ] Experiment with EfficientNet/ConvNeXt backbones for comparison
- [ ] FastAPI inference service wrapping `inference/predict.py`
- [ ] Model card documenting intended use, limitations, and evaluation data

## Contributors

- **Tribhuwan Singh** ([@tribhu05](https://github.com/tribhu05)) — author

Contributions welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Acknowledgements

- [VGG16](https://arxiv.org/abs/1409.1556) (Simonyan & Zisserman, 2014),
  pretrained weights via `tensorflow.keras.applications`.
- Brain Tumor MRI Dataset (standard 4-class Kaggle dataset layout).

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

## Citation

If you use this codebase, please cite it as:

```bibtex
@software{singh_brain_tumor_classifier,
  author = {Singh, Tribhuwan},
  title = {Brain Tumor MRI Classifier: VGG16 Transfer Learning},
  year = {2026},
  url = {https://github.com/tribhu05/brain-tumor-classifier}
}
```

## Contact

Tribhuwan Singh — [LinkedIn](https://linkedin.com/in/tribhuwan5050) —
[GitHub](https://github.com/tribhu05)

Project Link: [https://github.com/tribhu05/brain-tumor-classifier](https://github.com/tribhu05/brain-tumor-classifier)
