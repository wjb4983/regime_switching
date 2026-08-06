"""Validated configuration for optional transformer representation models."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field, PositiveInt, model_validator

from regime.models.base import RegimeModelConfig


class TransformerConfig(RegimeModelConfig):
    """Architecture and windowing controls for transformer encoders."""

    model_name: str = "time_series_transformer"
    input_dim: PositiveInt = 1
    embedding_dim: PositiveInt = Field(
        default=64, validation_alias=AliasChoices("embedding_dim", "d_model")
    )
    num_heads: PositiveInt = Field(
        default=4, validation_alias=AliasChoices("num_heads", "n_heads")
    )
    num_layers: PositiveInt = Field(
        default=2, validation_alias=AliasChoices("num_layers", "n_layers")
    )
    feedforward_dim: PositiveInt = 128
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    sequence_length: PositiveInt = 32
    patch_length: PositiveInt = 8
    patch_stride: PositiveInt = 4
    pooling: Literal["last", "mean"] = "mean"

    @model_validator(mode="after")
    def _validate_heads_and_patches(self) -> TransformerConfig:
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.patch_length > self.sequence_length:
            raise ValueError("patch_length cannot exceed sequence_length")
        return self
