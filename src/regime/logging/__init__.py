"""Structured logging helpers with redaction and warning capture."""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import sys
import warnings
from collections.abc import Mapping, MutableMapping
from typing import Any, TextIO

SECRET_FIELD_PATTERNS = (
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        "api[_-]?key",
        "secret",
        "token",
        "password",
        "passwd",
        "credential",
        "authorization",
    )
)
_SECRET_FIELD_PATTERNS = tuple(SECRET_FIELD_PATTERNS)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
)
REDACTION_TEXT = "[REDACTED]"

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def is_secret_field(name: str) -> bool:
    """Return whether a field name looks secret-like."""
    return any(pattern.search(name) for pattern in _SECRET_FIELD_PATTERNS)


def redact(value: Any, *, field_name: str | None = None) -> JsonValue:
    """Recursively redact secret-like fields and values into JSON-compatible data."""
    if field_name is not None and is_secret_field(field_name):
        return REDACTION_TEXT
    if isinstance(value, Mapping):
        return {str(key): redact(item, field_name=str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        redacted = value
        for pattern in _SECRET_VALUE_PATTERNS:
            if pattern.pattern.startswith("(?i)(bearer"):
                redacted = pattern.sub(r"\1" + REDACTION_TEXT, redacted)
            else:
                redacted = pattern.sub(REDACTION_TEXT, redacted)
        return redacted
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return repr(value)


class JsonLogFormatter(logging.Formatter):
    """Format log records as one-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: MutableMapping[str, JsonValue] = {
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        extra = getattr(record, "structured", None)
        if isinstance(extra, Mapping):
            payload["extra"] = redact(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(redact(payload), sort_keys=True, separators=(",", ":"))


def configure_logging(
    *,
    level: int | str = logging.INFO,
    stream: TextIO | None = None,
    logger_name: str | None = None,
    capture_warnings: bool = True,
) -> logging.Logger:
    """Configure a logger to emit structured JSON-compatible records."""
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    if capture_warnings:
        logging.captureWarnings(True)
        warnings.simplefilter("default")
        warning_logger = logging.getLogger("py.warnings")
        warning_logger.handlers.clear()
        warning_logger.addHandler(handler)
        warning_logger.setLevel(level)
        warning_logger.propagate = False
    return logger


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    """Log a structured event with redaction applied by the formatter."""
    logger.log(level, event, extra={"structured": fields})


__all__ = [
    "REDACTION_TEXT",
    "JsonLogFormatter",
    "JsonValue",
    "configure_logging",
    "is_secret_field",
    "log_event",
    "redact",
]
