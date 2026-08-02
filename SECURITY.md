# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| 0.1.x (original prototype notebook) | :x: |

## Reporting a Vulnerability

If you discover a security vulnerability in this project (e.g. an
issue in how model artifacts, uploaded images, or configuration files
are handled), please **do not open a public issue**. Instead:

1. Email the maintainer directly (see `README.md` for contact info)
   with a description of the vulnerability and steps to reproduce it.
2. Allow a reasonable amount of time for a response and fix before
   any public disclosure.

## Scope Notes

This is a research/portfolio project, not a deployed clinical system.
It has **not** been validated for real medical diagnostic use and
should not be used to make actual treatment decisions. Relevant
security-adjacent considerations for anyone extending this project
toward production use:

- **Model/data provenance:** trained model files (`.keras`) are
  Python-pickle-adjacent artifacts under the hood in some formats;
  only load model files from sources you trust.
- **Input validation:** `inference/predict.py` will raise a clear
  `FileNotFoundError` for missing/corrupt images rather than silently
  failing, but does not currently sanitize file paths — if you wrap
  this in a web service, validate/sanitize any user-supplied path or
  switch to in-memory file uploads before passing to
  `predict_image()`.
- **Dependency hygiene:** `requirements.txt` pins minimum versions;
  run `pip list --outdated` periodically and update for security
  patches, especially for `tensorflow` and `pillow`.
