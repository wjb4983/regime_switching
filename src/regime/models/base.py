"""Base interfaces and result containers for regime-switching models.

This module intentionally defines contracts rather than concrete algorithms.  Concrete
models should either implement each operation faithfully or inherit the explicit
``UnsupportedModelOperation`` failure from :class:`RegimeModel`.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from regime.config.base import RegimeBaseConfig
from regime.errors import RegimeModelError


class UnsupportedModelOperation(RegimeModelError):  # noqa: N818
    """Raised when a model cannot provide a requested operation exactly.

    Models should raise this exception instead of returning approximations for
    operations such as smoothing, transition estimation, or probabilistic
    prediction when those semantics are not supported by the concrete algorithm.
    """

    def __init__(self, operation: str, *, model_name: str | None = None) -> None:
        target = f" for {model_name}" if model_name else ""
        super().__init__(
            f"Operation {operation!r} is not supported{target}.",
            code="unsupported_model_operation",
            context={"operation": operation, "model_name": model_name},
        )
        self.operation = operation
        self.model_name = model_name


class RegimeModelConfig(RegimeBaseConfig):
    """Common configuration shared by regime model implementations."""

    model_name: str = Field(default="regime_model", min_length=1)
    n_states: PositiveInt = Field(default=2)
    random_seed: int | None = None
    model_version: str = Field(default="0.1.0", min_length=1)
    parameters: Mapping[str, Any] = Field(default_factory=dict)


class ModelMetadata(BaseModel):
    """Descriptive metadata exposed by fitted or loadable model instances."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str
    model_version: str
    n_states: PositiveInt
    fitted_at: datetime | None = None
    training_observations: int | None = Field(default=None, ge=0)
    feature_names: tuple[str, ...] = ()
    config_hash: str | None = None
    attributes: Mapping[str, Any] = Field(default_factory=dict)


class RegimeInferenceResult(BaseModel):
    """Full inference payload returned by regime model prediction APIs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    state: int = Field(ge=0, description="Hard regime/state assignment.")
    filtered_probabilities: tuple[float, ...]
    smoothed_probabilities: tuple[float, ...] | None = None
    change_probability: float = Field(ge=0.0, le=1.0)
    expected_regime_duration: float | None = Field(default=None, gt=0.0)
    transition_matrix: tuple[tuple[float, ...], ...] | None = None
    entropy: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    state_statistics: Mapping[str, Mapping[str, float]] = Field(default_factory=dict)
    model_version: str
    configuration_hash: str

    @model_validator(mode="after")
    def _validate_probability_shapes(self) -> Self:
        n_states = len(self.filtered_probabilities)
        if n_states == 0:
            raise ValueError("filtered_probabilities must contain at least one state probability")
        if self.smoothed_probabilities is not None and len(self.smoothed_probabilities) != n_states:
            raise ValueError("smoothed_probabilities must match filtered_probabilities length")
        if self.transition_matrix is not None:
            if len(self.transition_matrix) != n_states:
                raise ValueError("transition_matrix row count must match state probability length")
            if any(len(row) != n_states for row in self.transition_matrix):
                raise ValueError("transition_matrix must be square with one row per state")
        if self.state >= n_states:
            raise ValueError("state must index filtered_probabilities")
        return self


class SerializationInterface(ABC):
    """Persistence contract for regime models."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist this model to ``path``."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Self:
        """Load a model instance from ``path``."""


class ModelMetadataInterface(ABC):
    """Contract for models that expose stable metadata."""

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Return immutable descriptive metadata for this model."""


class RegimeModel(SerializationInterface, ModelMetadataInterface, ABC):
    """Abstract base class for regime model implementations.

    Default optional-operation implementations raise
    :class:`UnsupportedModelOperation` to prevent silent approximations.
    """

    @property
    def model_name(self) -> str:
        """Human-readable model name used in metadata and errors."""
        return self.metadata.model_name

    @abstractmethod
    def fit(self, dataset: Any, config: RegimeModelConfig) -> Self:
        """Fit model parameters using ``dataset`` and ``config``."""

    @abstractmethod
    def predict(self, dataset: Any) -> Sequence[int] | Sequence[RegimeInferenceResult]:
        """Return hard regime assignments or full inference records for ``dataset``."""

    def predict_proba(self, dataset: Any) -> Sequence[Sequence[float]]:
        """Return filtered state probabilities for ``dataset``."""
        raise UnsupportedModelOperation("predict_proba", model_name=self.model_name)

    def filter(self, observation: Any) -> RegimeInferenceResult:
        """Run online filtering for a single observation."""
        raise UnsupportedModelOperation("filter", model_name=self.model_name)

    def smooth(self, dataset: Any) -> Sequence[RegimeInferenceResult]:
        """Run fixed-interval smoothing for ``dataset``."""
        raise UnsupportedModelOperation("smooth", model_name=self.model_name)

    def transition_matrix(self) -> Sequence[Sequence[float]]:
        """Return the estimated Markov transition matrix."""
        raise UnsupportedModelOperation("transition_matrix", model_name=self.model_name)

    def state_statistics(self) -> Mapping[str, Mapping[str, float]]:
        """Return per-state summary statistics from the fitted model."""
        raise UnsupportedModelOperation("state_statistics", model_name=self.model_name)

    @staticmethod
    def configuration_hash(config: RegimeModelConfig | Mapping[str, Any]) -> str:
        """Return a stable SHA-256 hash for a model configuration."""
        if isinstance(config, RegimeModelConfig):
            return config.config_hash()
        payload = json.dumps(config, default=str, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
