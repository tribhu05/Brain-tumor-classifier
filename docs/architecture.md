# Architecture

## Overview

```
                    ┌─────────────────────┐
                    │   configs/config.yaml │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      config.py       │  typed Config dataclasses
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌────────────────┐ ┌─────────────┐ ┌──────────────┐
     │ data/dataset.py │ │models/vgg16_│ │ utils/seed.py │
     │ data/augment.py │ │classifier.py│ │ utils/logging │
     └────────┬────────┘ └──────┬──────┘ └──────────────┘
              │                 │
              └────────┬────────┘
                        ▼
              ┌──────────────────┐
              │ training/train.py │  fit() + callbacks + checkpointing
              └─────────┬─────────┘
                        │  saved .keras model
              ┌─────────┴─────────┐
              ▼                   ▼
   ┌───────────────────┐ ┌──────────────────┐
   │evaluation/evaluate │ │inference/predict │
   └──────────┬─────────┘ └────────┬─────────┘
              ▼                    ▼
   ┌───────────────────────────────────────┐
   │       visualization/plots.py            │
   └───────────────────────────────────────┘
```

## Why this module split

Each pipeline stage (data → model → train → evaluate → infer) is an
independently importable, independently testable unit:

- **`data/`** owns everything about turning files-on-disk into
  `tf.data.Dataset` batches: class discovery, path/label loading,
  train/validation splitting, and augmentation. Nothing outside this
  package needs to know how images are read from disk.
- **`models/`** owns architecture definition only — no data loading, no
  training loop logic. This means the model can be unit-tested with
  synthetic tensors in milliseconds, without touching the filesystem.
- **`training/`** orchestrates data + model + callbacks into a fit
  loop. It's the only module that knows about `ModelCheckpoint`,
  `EarlyStopping`, etc.
- **`evaluation/`** and **`inference/`** are deliberately separate from
  each other: evaluation runs on a labeled dataset and produces
  aggregate metrics; inference runs on a single unlabeled image and
  produces one prediction. Conflating them (as the original
  `detect_and_display()` did) makes both harder to test and harder to
  reuse — e.g. inference logic can be dropped into a future Flask/
  FastAPI service without dragging along matplotlib or sklearn.
- **`visualization/`** is presentation-only and takes plain data
  structures (arrays, `History` objects) as input — it doesn't know
  where those structures came from, so the same plotting functions
  work whether called from a script, a notebook, or a test.

## Data flow contract

The single most important invariant in this codebase, because it's
what the original notebook got wrong (see `docs/known_limitations.md`
and `CHANGELOG.md`): **class name ordering is established exactly once**,
via `data.dataset.discover_class_names()`, and that same ordered list
is passed explicitly to every function that encodes/decodes labels —
`encode_labels`, `build_tf_dataset`, `evaluate_model`,
`predict_image`, `plot_confusion_matrix`. No function re-derives class
names independently, which is what caused the original label-mismatch
bug.

## Config-driven, not hardcoded

Every hyperparameter and path flows through `config.py`'s dataclasses,
sourced from `configs/config.yaml` (with CLI flag overrides in the
`scripts/*.py` entrypoints). Defaults exactly match the original
notebook's hardcoded values, so nothing about model behavior changes
unless you explicitly edit the config.
