# Contributing to Brain Tumor Classifier

Thanks for your interest in contributing! This document covers how to get
set up, the coding standards used in this repo, and how to submit changes.

## Getting Started

```bash
git clone https://github.com/tribhu05/brain-tumor-classifier.git
cd brain-tumor-classifier
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
make install-dev
```

## Development Workflow

1. Create a branch off `main`: `git checkout -b feature/your-feature-name`
2. Make your changes.
3. Run the test suite: `make test`
4. Run linting/formatting: `make lint` and `make format`
5. Commit using a clear, descriptive message (see below).
6. Push and open a pull request against `main`.

## Coding Standards

- **Style:** PEP 8, enforced via `black` and `flake8` (line length 100).
- **Imports:** sorted with `isort` (black-compatible profile).
- **Type hints:** all new functions should have type hints on parameters
  and return values.
- **Docstrings:** all public functions/classes need a docstring covering
  purpose, args, returns, and any exceptions raised (Google style, as
  used throughout `src/`).
- **Tests:** new functionality needs corresponding tests in `tests/`.
  Tests should use synthetic/generated data (see `tests/conftest.py`),
  never the real MRI dataset, so the suite stays fast and doesn't
  require dataset access in CI.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) style
where practical:

```
feat: add early stopping callback to training loop
fix: correct label ordering in confusion matrix plot
docs: update README installation instructions
test: add coverage for train/validation split edge cases
```

## Reporting Bugs / Requesting Features

Please use the issue templates under `.github/ISSUE_TEMPLATE/`.

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). By
participating, you agree to uphold it.
