"""
logging_config.py
==================

Structured logging setup. Replaces the original notebook's ``print()``
calls with proper leveled logging that can be filtered, redirected to a
file, and turned off in production without touching call sites.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """Configure root logging for the application.

    Args:
        level: Logging level (e.g. ``logging.INFO``, ``logging.DEBUG``).
        log_file: Optional path to also write logs to a file. Parent
            directories are created if they don't exist.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
