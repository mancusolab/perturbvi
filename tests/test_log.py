import logging

from perturbvi.log import get_logger


def _close_handlers(logger):
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_logger_honors_file_path_after_console_only_setup(tmp_path):
    logger = get_logger("perturbvi.test.late-file")
    try:
        logger = get_logger(logger.name, path=tmp_path / "run")
        logger.info("saved message")
        for handler in logger.handlers:
            handler.flush()

        assert "saved message" in (tmp_path / "run.log").read_text(encoding="utf-8")
    finally:
        _close_handlers(logger)


def test_logger_reuses_or_closes_managed_file_handlers(tmp_path):
    logger = get_logger("perturbvi.test.file-lifecycle", path=tmp_path / "first")
    try:
        first_handler = next(handler for handler in logger.handlers if isinstance(handler, logging.FileHandler))

        same_logger = get_logger(logger.name, path=tmp_path / "first")
        same_handler = next(handler for handler in same_logger.handlers if isinstance(handler, logging.FileHandler))
        assert same_handler is first_handler

        moved_logger = get_logger(logger.name, path=tmp_path / "second")
        second_handler = next(handler for handler in moved_logger.handlers if isinstance(handler, logging.FileHandler))
        assert second_handler is not first_handler
        assert first_handler.stream is None
    finally:
        _close_handlers(logger)
