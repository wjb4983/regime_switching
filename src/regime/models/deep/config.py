"""Configuration objects for optional deep regime models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from regime.models.base import RegimeModelConfig


class DeepModelConfig(RegimeModelConfig):
    """Training and runtime settings shared by all deep models.

    ``validation_window`` is always taken from the end of the supplied series.  It is
    never included in gradient updates, avoiding a common time-series leakage bug.
    """

    hidden_size: PositiveInt = 32
    embedding_dim: PositiveInt = 16
    num_layers: PositiveInt = 1
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    learning_rate: PositiveFloat = 1e-3
    weight_decay: float = Field(default=0.0, ge=0.0)
    batch_size: PositiveInt = 32
    max_epochs: PositiveInt = 100
    patience: PositiveInt = 10
    min_delta: float = Field(default=0.0, ge=0.0)
    validation_window: PositiveInt | float = Field(default=0.2)
    sequence_length: PositiveInt = 16
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    precision: Literal["float32", "float64", "16-mixed", "bf16-mixed"] = "float32"
    deterministic: bool = True
    checkpoint_path: Path | None = None
    calibration: Literal["none", "temperature"] = "temperature"
    gradient_clip_val: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_validation_window(self) -> DeepModelConfig:
        value = self.validation_window
        if isinstance(value, float) and not 0.0 < value < 1.0:
            raise ValueError("fractional validation_window must be between zero and one")
        return self
