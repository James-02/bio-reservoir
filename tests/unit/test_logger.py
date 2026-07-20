"""Unit tests for utils.logger.setup_logging."""

import logging

import pytest

from utils.logger import setup_logging


@pytest.fixture
def clean_root_logger():
    """Save and restore the root logger's handlers/level around a test."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    root.handlers = []
    try:
        yield root
    finally:
        for handler in root.handlers:
            if handler not in saved_handlers:
                handler.close()
        root.handlers = saved_handlers
        root.setLevel(saved_level)


class TestQuietLoggers:
    def test_quiet_loggers_silenced(self, clean_root_logger):
        setup_logging(level=logging.INFO, quiet_loggers=("numba", "matplotlib"))
        for name in ("numba", "matplotlib"):
            lg = logging.getLogger(name)
            assert lg.level == logging.ERROR
            assert lg.propagate is False

    def test_default_quiet_logger_is_numba(self, clean_root_logger):
        setup_logging()
        assert logging.getLogger("numba").level == logging.ERROR


class TestRootConfiguration:
    def test_configures_root_when_empty(self, clean_root_logger):
        # Ensure the root logger really has no handlers (pytest attaches its own
        # capture handler during a test).
        clean_root_logger.handlers = []
        setup_logging(level=logging.DEBUG)
        assert clean_root_logger.level == logging.DEBUG
        # basicConfig should have attached at least one handler.
        assert clean_root_logger.handlers

    def test_updates_level_when_already_configured(self, clean_root_logger):
        existing = logging.NullHandler()
        clean_root_logger.addHandler(existing)
        before = list(clean_root_logger.handlers)
        setup_logging(level=logging.WARNING)
        assert clean_root_logger.level == logging.WARNING
        # No new handler added when the root logger is already configured.
        assert clean_root_logger.handlers == before
        assert existing in clean_root_logger.handlers


class TestLogFile:
    def test_creates_log_file_and_directory(self, clean_root_logger, tmp_path):
        # Force the basicConfig path by clearing pytest's capture handler.
        clean_root_logger.handlers = []
        log_path = tmp_path / "logs" / "run.log"
        setup_logging(level=logging.INFO, log_file=str(log_path))
        assert log_path.parent.is_dir()
        file_handlers = [
            h for h in clean_root_logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert file_handlers, "expected a FileHandler when log_file is given"
