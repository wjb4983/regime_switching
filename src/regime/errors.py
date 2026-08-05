"""Typed exception hierarchy for regime switching workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class RegimeError(Exception):
    """Base class for all package-specific exceptions."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        context: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.context = dict(context or {})
        if cause is not None:
            self.__cause__ = cause

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-compatible error record for structured logging."""
        return {
            "type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }


class RegimeConfigurationError(RegimeError):
    """Raised when configuration is missing, invalid, or inconsistent."""


class RegimeDataError(RegimeError):
    """Raised when data loading, validation, or hashing fails."""


class RegimeFeatureError(RegimeError):
    """Raised when feature generation or validation fails."""


class RegimeModelError(RegimeError):
    """Raised when model creation, training, inference, or hashing fails."""


class RegimeExperimentError(RegimeError):
    """Raised when experiment orchestration or provenance capture fails."""


class RegimeExternalServiceError(RegimeError):
    """Raised when an external service call fails."""
