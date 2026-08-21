import logging

from pathlib import Path


_LOG_FORMAT = "[%(asctime)s - %(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _formatter() -> logging.Formatter:
    return logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)


def _log_path(path: str | Path) -> Path:
    return Path(f"{path}.log").resolve()


def get_logger(name: str, path: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a configured PerturbVI logger, optionally writing to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    has_console = any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )
    if not has_console:
        console = logging.StreamHandler()
        console.setFormatter(_formatter())
        logger.addHandler(console)

    if path is not None:
        requested_path = _log_path(path)
        file_handlers = [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]
        if not any(Path(handler.baseFilename) == requested_path for handler in file_handlers):
            for handler in file_handlers:
                if getattr(handler, "_perturbvi_file_handler", False):
                    logger.removeHandler(handler)
                    handler.close()

            disk_handler = logging.FileHandler(requested_path, mode="w", encoding="utf-8")
            disk_handler._perturbvi_file_handler = True
            disk_handler.setFormatter(_formatter())
            logger.addHandler(disk_handler)

    return logger
