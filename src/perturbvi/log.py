from __future__ import annotations

import logging


LOG_FORMAT = "[%(asctime)s - %(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Extra per-iteration detail lives on this child logger. Its records are
# ordinary INFO lines; set_verbose() decides whether they are emitted.
CHATTER_LOGGER = "perturbvi.verbose"


def _has_real_handler(logger: logging.Logger) -> bool:
    """True if any usable handler exists on this logger or its ancestors."""
    node = logger
    while node:
        if any(not isinstance(h, logging.NullHandler) for h in node.handlers):
            return True
        if not node.propagate:
            return False
        node = node.parent
    return False


def get_logger(name: str) -> logging.Logger:
    """Return a logger, guaranteeing timestamped output when unconfigured.

    Attaches the standard console handler to the package logger only when no
    application-provided handler is found. Applications with their own logging
    configuration are left untouched.
    """
    logger = logging.getLogger(name)
    package = logging.getLogger("perturbvi")
    if not _has_real_handler(package):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
        package.addHandler(console)
        if package.level == logging.NOTSET:
            package.setLevel(logging.INFO)
        package.propagate = False
    return logger


def set_verbose(enabled: bool) -> None:
    """Route library logging identically for the CLI and Python API.

    ``False`` (default) keeps milestone progress visible as INFO. ``True``
    additionally emits the per-iteration ELBO detail — also labeled INFO.
    """
    logging.getLogger(CHATTER_LOGGER).setLevel(logging.INFO if enabled else logging.WARNING)
