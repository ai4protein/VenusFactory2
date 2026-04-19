"""Tests for centralized logging."""
import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


class TestLogger:
    def test_get_logger(self):
        from logger import get_logger

        lg = get_logger("test_module")
        assert lg.name == "venus.test_module"
        assert isinstance(lg, logging.Logger)

    def test_logger_namespace(self):
        from logger import get_logger

        lg = get_logger("sub.module")
        assert lg.name == "venus.sub.module"

    def test_root_logger_has_handler(self):
        from logger import setup_logging

        setup_logging()
        root = logging.getLogger("venus")
        assert len(root.handlers) > 0

    def test_setup_idempotent(self):
        from logger import setup_logging

        setup_logging()
        count1 = len(logging.getLogger("venus").handlers)
        setup_logging()
        count2 = len(logging.getLogger("venus").handlers)
        assert count1 == count2

    def test_module_logger(self):
        from logger import logger

        assert logger.name == "venus.core"

    def test_env_bool(self):
        from logger import _env_bool

        os.environ["_TEST_BOOL_1"] = "true"
        os.environ["_TEST_BOOL_0"] = "0"
        assert _env_bool("_TEST_BOOL_1") is True
        assert _env_bool("_TEST_BOOL_0") is False
        assert _env_bool("_NONEXISTENT_KEY", default=True) is True
        del os.environ["_TEST_BOOL_1"]
        del os.environ["_TEST_BOOL_0"]
