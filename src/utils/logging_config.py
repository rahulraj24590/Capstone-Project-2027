"""
Logging configuration for LandslideGuard.

Sets up structured logging with both console and file handlers.
Call ``setup_logging()`` once at application startup.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_dir: str | Path = "logs",
    log_filename: str = "landslideguard.log",
) -> logging.Logger:
    """Configure root logger with console and rotating file handlers.

    Args:
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files.
        log_filename: Name of the log file.

    Returns:
        Configured root logger.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers if called multiple times
    if root_logger.handlers:
        return root_logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating file handler (10 MB × 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / log_filename,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy_lib in ("urllib3", "requests", "httpx", "matplotlib", "PIL"):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
