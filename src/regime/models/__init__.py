"""Model interfaces for regime switching workflows."""

from regime.models.base import (
    ModelMetadata,
    ModelMetadataInterface,
    RegimeInferenceResult,
    RegimeModel,
    RegimeModelConfig,
    SerializationInterface,
    UnsupportedModelOperation,
)
from regime.models.registry import (
    ModelRegistryError,
    ModelSpec,
    available_models,
    create_model,
    model_spec,
)

__all__ = [
    "ModelMetadata",
    "ModelMetadataInterface",
    "ModelRegistryError",
    "ModelSpec",
    "RegimeInferenceResult",
    "RegimeModel",
    "RegimeModelConfig",
    "SerializationInterface",
    "UnsupportedModelOperation",
    "available_models",
    "create_model",
    "model_spec",
]
