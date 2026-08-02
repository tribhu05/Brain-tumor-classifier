# Known Limitations

This document is intentionally honest about tradeoffs and limitations
that were **not** changed during the refactor, so anyone reviewing this
repo (including you, later) knows exactly what's been fixed and what
hasn't.

## 1. Pixel preprocessing does not match VGG16's original training distribution

VGG16's ImageNet weights were originally trained using
`tf.keras.applications.vgg16.preprocess_input`, which converts RGB to
BGR and subtracts per-channel ImageNet means — **not** simple `/255.0`
min-max rescaling. This codebase preserves the original notebook's
`/255.0` rescaling behavior exactly (see `data/augmentation.py`,
`rescale_only()`), rather than silently changing it, per the goal of
not altering core model behavior during this refactor.

**Practical effect:** the frozen (non-fine-tuned) VGG16 layers receive
inputs on a different distribution than they were trained on, which
can reduce the effectiveness of transfer learning somewhat. Because
the last 3 conv layers are fine-tuned, the network can partially adapt
to this mismatch during training, but it's not optimal.

**If you want to fix this:** swap `rescale_only()` for
`tf.keras.applications.vgg16.preprocess_input` in
`data/augmentation.py` and re-train. This is flagged as a "Future
Improvements" item in the README rather than applied automatically,
since it changes model input semantics and would require re-training
and re-validating reported metrics.

## 2. No dataset-level class imbalance handling

If the four classes (glioma / meningioma / notumor / pituitary) are
not evenly represented in the source dataset, `training/train.py` does
not currently apply class weighting or oversampling. The stratified
train/validation split (`data/dataset.py::train_validation_split`)
ensures the *split* doesn't introduce additional imbalance, but it
doesn't correct for imbalance that already exists in the raw data.

## 3. Image size (128x128) is small relative to typical medical imaging pipelines

The original notebook used 128x128 inputs, likely for training speed.
This is preserved as the default. Many published brain-MRI transfer
learning results use 224x224 (VGG16's native ImageNet input size) or
larger; increasing `data.image_size` in `configs/config.yaml` is a
straightforward experiment but will increase training time and memory
use, and is not benchmarked in this repository.

## 4. No cross-validation

Training uses a single stratified train/validation split rather than
k-fold cross-validation. For a dataset this size, k-fold would give a
more robust estimate of generalization performance but was out of
scope for this refactor (it changes the training script's control flow
significantly).

## 5. Not clinically validated

This project is a portfolio/research artifact demonstrating a
transfer-learning pipeline. It has not been validated against a
clinical ground truth, reviewed by a radiologist, or tested for
demographic/scanner bias, and should not be used for actual diagnostic
decisions.
