"""
Centralized logging configuration for audiotext.

Provides consistent logging across the application with:
- Structured log format
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Optional file and console output
- No dependency on print() statements
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from utils.path_helper import ROOT_PATH

# Default log directory
LOG_DIR = ROOT_PATH / "logs"
LOG_FILE = LOG_DIR / "audiotext.log"


def setup_logger(
    name: str = "audiotext",
    level: int = logging.INFO,
    log_file: Optional[Path] = LOG_FILE,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up and return a logger instance.

    :param name: Logger name (usually __name__)
    :param level: Logging level (default: INFO)
    :param log_file: Path to log file. If None, no file logging.
    :param console_output: Whether to log to console
    :return: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Common formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    if log_file:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as e:
            # If we can't write logs, fall back to console only
            print(f"Warning: Could not create log file {log_file}: {e} - logger.py:62", file=sys.stderr)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger for the given module name.
    Uses lazy initialization to avoid circular imports.

    :param name: Module name (typically __name__)
    :return: Logger instance
    """
    logger = logging.getLogger(name)

    # If not yet configured, set up with defaults
    if not logger.handlers:
        return setup_logger(name)

    return logger