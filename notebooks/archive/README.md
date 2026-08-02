# Archived Prototype Notebook

`original_prototype.ipynb` is the original single-notebook implementation
this repository was refactored from. It's kept here for provenance and to
let reviewers see the "before" state referenced in `CHANGELOG.md`.

**Do not use this notebook directly** — it contains hardcoded local file
paths and the bugs documented in `CHANGELOG.md` (test-set augmentation
leakage, missing validation split, label-order mismatch, no seeding). The
production implementation lives in `src/brain_tumor_classifier/` and is
run via the scripts in `scripts/`.
