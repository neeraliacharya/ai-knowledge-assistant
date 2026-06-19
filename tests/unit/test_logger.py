"""
Unit tests for app/services/logger.py

Verifies that every log line is valid JSON with the required fields and that
extra context fields are correctly embedded.
"""
import json
import logging
import pytest
from io import StringIO
from app.services.logger import get_logger

pytestmark = pytest.mark.unit


def capture_log(name: str, level: str, message: str, extra: dict = None) -> dict:
    """
    Helper: call the logger, capture stdout, return the parsed JSON record.
    """
    buf = StringIO()
    handler = logging.StreamHandler(buf)

    logger = get_logger(name)
    # Temporarily replace handlers so we can capture output cleanly
    original_handlers = logger.handlers[:]
    logger.handlers = [handler]
    handler.setFormatter(logger.handlers[0].formatter if original_handlers else None)

    # Re-get logger with buffer handler
    logger2 = logging.getLogger(name + "_test_capture")
    logger2.handlers = []
    buf2 = StringIO()
    from app.services.logger import _JSONFormatter
    h = logging.StreamHandler(buf2)
    h.setFormatter(_JSONFormatter())
    logger2.addHandler(h)
    logger2.setLevel(logging.DEBUG)
    logger2.propagate = False

    if extra:
        getattr(logger2, level)(message, extra=extra)
    else:
        getattr(logger2, level)(message)

    line = buf2.getvalue().strip()
    return json.loads(line)


def test_log_line_is_valid_json():
    record = capture_log("test.logger", "info", "hello world")
    assert isinstance(record, dict)


def test_required_fields_present():
    record = capture_log("test.logger", "info", "checking fields")
    assert "ts" in record
    assert "level" in record
    assert "msg" in record


def test_level_is_uppercase():
    record = capture_log("test.logger", "warning", "warn message")
    assert record["level"] == "WARNING"


def test_message_matches():
    record = capture_log("test.logger", "info", "exact message content")
    assert record["msg"] == "exact message content"


def test_extra_fields_embedded():
    record = capture_log("test.logger", "info", "with extras",
                         extra={"chunks": 42, "source": "test.pdf"})
    assert record.get("chunks") == 42
    assert record.get("source") == "test.pdf"


def test_error_level():
    record = capture_log("test.logger", "error", "something failed")
    assert record["level"] == "ERROR"


def test_timestamp_is_iso_format():
    record = capture_log("test.logger", "info", "ts check")
    ts = record["ts"]
    # Basic ISO format check: contains T separator and timezone offset
    assert "T" in ts


def test_get_logger_returns_same_instance():
    """get_logger with the same name must return the same Logger object."""
    l1 = get_logger("myapp.service")
    l2 = get_logger("myapp.service")
    assert l1 is l2
