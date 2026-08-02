# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] - Production Refactor

Restructured the original single-notebook prototype into a modular,
tested, documented Python package. Model architecture and core training
behavior are unchanged; this release focuses on correctness fixes,
reproducibility, and engineering quality.

### Fixed
- **Test-set evaluation leakage:** random augmentation was previously
  applied to test images during evaluation, making reported metrics
  non-reproducible and not representative of true generalization.
  Evaluation and inference now always use unaugmented preprocessing.
- **Confusion matrix label mismatch:** axis labels were sourced from
  unsorted `os.listdir()` while label encoding used a sorted list,
  risking mislabeled axes. Class names are now discovered once and
  threaded through the entire pipeline consistently.
- **Missing validation split:** training previously had no validation
  set, making overfitting invisible. Added a seeded, stratified
  train/validation split.
- **Dropped final batch every epoch:** integer division on step count
  silently discarded up to `batch_size - 1` training images per epoch.
  Replaced with a `tf.data` pipeline (`drop_remainder=False`).
- **Non-reproducible runs:** no random seed was set anywhere. Added
  centralized seeding for Python, NumPy, and TensorFlow.
- **`plt.tight_layout` no-op:** was called without parentheses in the
  original sample-image visualization.
- **Fragile backbone layer selection:** VGG16 fine-tuning layers were
  selected via negative indexing (`layers[-2]`, etc.), which silently
  breaks if Keras changes internal layer ordering. Now selected
  explicitly by layer name, with validation.

### Added
- Modular `src/brain_tumor_classifier/` package (data, models,
  training, evaluation, inference, visualization, utils).
- YAML-driven, typed configuration system (`configs/config.yaml`).
- `tf.data` pipeline replacing the hand-rolled Python generator
  (parallel decode, prefetching, proper per-epoch reshuffling).
- Training callbacks: `ModelCheckpoint` (best-by-val-accuracy),
  `EarlyStopping`, `CSVLogger`.
- Per-class ROC-AUC evaluation metric (previously imported but unused
  in the original notebook).
- pytest test suite (dataset, config, augmentation, model, inference)
  using synthetic data, decoupled from the real dataset.
- CLI scripts for train / evaluate / predict.
- Docker image, Makefile, CI workflow, and full repository
  documentation (this README, CONTRIBUTING, SECURITY, etc.).

### Changed
- Model checkpoints now save in the modern `.keras` format instead of
  legacy `.h5`.
- Hardcoded personal Windows paths replaced with configurable dataset
  paths (`configs/config.yaml` or `BTC_TRAIN_DIR`/`BTC_TEST_DIR` env vars).

### Preserved (intentionally unchanged)
- VGG16 backbone + `Flatten -> Dropout(0.3) -> Dense(128, relu) ->
  Dropout(0.2) -> Dense(4, softmax)` architecture.
- Adam optimizer, learning rate `1e-4`, `sparse_categorical_crossentropy` loss.
- Image size 128x128, batch size 20, 10 epochs (all now configurable,
  defaulting to these original values).
- Manual `/255.0` pixel rescaling (rather than
  `vgg16.preprocess_input`) — see `docs/known_limitations.md` for the
  tradeoff this implies and how to change it.

## [0.1.0] - Original Prototype

Initial single-notebook implementation (`notebooks/archive/original_prototype.ipynb`):
VGG16 transfer learning for 4-class brain MRI classification
(glioma / meningioma / notumor / pituitary), trained and evaluated
end-to-end in one notebook.
