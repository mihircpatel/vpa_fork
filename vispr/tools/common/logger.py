"""Unified logging helper for vispr flows.

Provides get_logger(name, log_file=None, level=None, console=True, rotate=True).
Creates logs directory under repo root (logs/) by default.
Uses RotatingFileHandler to avoid unbounded logs and allows LOG_LEVEL env override.
"""
import logging
import logging.handlers
import os
import os.path as osp
import sys

DEFAULT_LOG_DIR = None  # resolved relative to repo root
DEFAULT_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()


def _get_repo_root():
    # vispr/tools/common/logger.py -> repo root is three dirs up
    return osp.abspath(osp.dirname(osp.dirname(osp.dirname(__file__))))


def get_logger(name: str, log_file: str = None, level: str = None, console: bool = True, rotate: bool = True):
    """Return a configured logger.

    If log_file is None, uses <repo_root>/logs/{name}.log.
    If level is None, uses LOG_LEVEL env var or DEFAULT_LEVEL.
    Multiple calls for same name will return the same logger (idempotent).
    """
    repo_root = _get_repo_root()
    global DEFAULT_LOG_DIR
    if DEFAULT_LOG_DIR is None:
        DEFAULT_LOG_DIR = osp.join(repo_root, 'logs')

    if level is None:
        level = os.environ.get('LOG_LEVEL', DEFAULT_LEVEL)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    if log_file is None:
        log_file = osp.join(DEFAULT_LOG_DIR, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Ensure log dir exists
    log_dir = osp.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if rotate:
        handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
    else:
        handler = logging.FileHandler(log_file)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if console:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    # Do not propagate to root logger to avoid duplicate logs when root is configured
    logger.propagate = False
    return logger
