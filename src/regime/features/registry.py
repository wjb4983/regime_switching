"""Feature registry and leakage-aware feature building utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from regime.experiments.provenance import stable_hash


class MissingValuePolicy(StrEnum):
    """Supported missing-value handling policies for feature outputs."""

    ERROR = "error"
    DROP = "drop"
    FORWARD_FILL = "forward_fill"
    LEAVE = "leave"


class ScalingMethod(StrEnum):
    """Supported feature scaling methods."""

    NONE = "none"
    STANDARD = "standard"
    MIN_MAX = "min_max"


class FeatureSemantics(StrEnum):
    """Whether feature meaning is time-series or cross-sectional."""

    TIME_SERIES = "time_series"
    CROSS_SECTIONAL = "cross_sectional"


class FittingRequirement(StrEnum):
    """How a feature transform may be fit."""

    NONE = "none"
    TRAINING_WINDOW_ONLY = "training_window_only"


RawInputFrames = Mapping[str, pd.DataFrame]
FeatureTransform = Callable[[RawInputFrames], pd.DataFrame]


def _dataframe_hash(frame: pd.DataFrame) -> str:
    """Return a deterministic hash for a pandas frame including its index."""
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return stable_hash(hashed)


@dataclass(frozen=True, slots=True)
class OutputField:
    """A single column emitted by a feature definition."""

    name: str
    dtype: str
    nullable: bool = True
    description: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-compatible schema record."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "nullable": self.nullable,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """Declarative metadata and transformation logic for one feature set."""

    name: str
    version: str
    required_raw_inputs: tuple[str, ...]
    lookback: pd.Timedelta
    publication_lag: pd.Timedelta
    warm_up_period: pd.Timedelta
    missing_value_policy: MissingValuePolicy
    scaling_method: ScalingMethod
    semantics: FeatureSemantics
    fitting_requirement: FittingRequirement
    leakage_risks: tuple[str, ...]
    output_schema: tuple[OutputField, ...]
    cache_key_inputs: tuple[str, ...]
    transform: FeatureTransform

    def to_record(self) -> dict[str, Any]:
        """Return deterministic metadata suitable for hashes and provenance."""
        return {
            "name": self.name,
            "version": self.version,
            "required_raw_inputs": self.required_raw_inputs,
            "lookback_seconds": self.lookback.total_seconds(),
            "publication_lag_seconds": self.publication_lag.total_seconds(),
            "warm_up_period_seconds": self.warm_up_period.total_seconds(),
            "missing_value_policy": self.missing_value_policy.value,
            "scaling_method": self.scaling_method.value,
            "semantics": self.semantics.value,
            "fitting_requirement": self.fitting_requirement.value,
            "leakage_risks": self.leakage_risks,
            "output_schema": [field.to_record() for field in self.output_schema],
            "cache_key_inputs": self.cache_key_inputs,
        }


@dataclass(frozen=True, slots=True)
class FeatureBuildConfig:
    """Point-in-time configuration for a feature build."""

    as_of: pd.Timestamp
    training_start: pd.Timestamp | None = None
    training_end: pd.Timestamp | None = None
    entity_column: str | None = None
    timestamp_column: str = "timestamp"
    publication_timestamp_column: str = "published_at"

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration record."""
        return {
            "as_of": self.as_of.isoformat(),
            "training_start": self.training_start.isoformat()
            if self.training_start is not None
            else None,
            "training_end": self.training_end.isoformat()
            if self.training_end is not None
            else None,
            "entity_column": self.entity_column,
            "timestamp_column": self.timestamp_column,
            "publication_timestamp_column": self.publication_timestamp_column,
        }


@dataclass(frozen=True, slots=True)
class FeatureBuildResult:
    """Feature output plus reproducibility metadata."""

    features: pd.DataFrame
    provenance: dict[str, Any]
    feature_hash: str
    cache_key: str
    cache_hit: bool


@dataclass
class FeatureRegistry:
    """In-memory registry for feature definitions."""

    _definitions: dict[str, FeatureDefinition] = field(default_factory=dict)

    def register(self, definition: FeatureDefinition) -> None:
        """Register a feature definition by name and reject incompatible duplicates."""
        existing = self._definitions.get(definition.name)
        if existing is not None and existing.version != definition.version:
            raise ValueError(f"feature {definition.name!r} is already registered")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> FeatureDefinition:
        """Return a registered feature definition."""
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise KeyError(f"unknown feature {name!r}") from exc


@dataclass
class FeatureBuilder:
    """Build registered features with point-in-time filtering, fitting, and caching."""

    registry: FeatureRegistry
    _cache: dict[str, FeatureBuildResult] = field(default_factory=dict)

    def build(
        self,
        feature_name: str,
        raw_inputs: RawInputFrames,
        config: FeatureBuildConfig,
    ) -> FeatureBuildResult:
        """Build one feature set without using data unavailable at ``config.as_of``."""
        definition = self.registry.get(feature_name)
        filtered_inputs = self._point_in_time_inputs(definition, raw_inputs, config)
        cache_key = self._cache_key(definition, filtered_inputs, config)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return FeatureBuildResult(
                cached.features.copy(),
                cached.provenance,
                cached.feature_hash,
                cache_key,
                True,
            )

        features = definition.transform(filtered_inputs).copy()
        features = self._apply_warm_up(features, definition, config)
        features = self._apply_missing_policy(features, definition)
        features = self._apply_scaling(features, definition, config)
        self._validate_output_schema(features, definition)
        feature_hash = _dataframe_hash(features)
        provenance = {
            "definition": definition.to_record(),
            "config": config.to_record(),
            "raw_input_hashes": {
                name: _dataframe_hash(frame) for name, frame in filtered_inputs.items()
            },
            "feature_hash": feature_hash,
            "cache_key": cache_key,
        }
        result = FeatureBuildResult(features, provenance, feature_hash, cache_key, False)
        self._cache[cache_key] = FeatureBuildResult(
            features.copy(),
            provenance,
            feature_hash,
            cache_key,
            False,
        )
        return result

    def _point_in_time_inputs(
        self, definition: FeatureDefinition, raw_inputs: RawInputFrames, config: FeatureBuildConfig
    ) -> dict[str, pd.DataFrame]:
        missing = set(definition.required_raw_inputs) - set(raw_inputs)
        if missing:
            raise ValueError(f"missing raw inputs: {sorted(missing)}")
        earliest = config.as_of - definition.lookback - definition.warm_up_period
        filtered: dict[str, pd.DataFrame] = {}
        for name in definition.required_raw_inputs:
            frame = raw_inputs[name].copy()
            if (
                config.timestamp_column not in frame
                or config.publication_timestamp_column not in frame
            ):
                raise ValueError(
                    f"raw input {name!r} must include timestamp and publication columns"
                )
            timestamp = pd.to_datetime(frame[config.timestamp_column])
            published_at = pd.to_datetime(frame[config.publication_timestamp_column])
            available_at = published_at + definition.publication_lag
            if (timestamp > available_at).any():
                raise ValueError(f"raw input {name!r} has timestamps newer than their availability")
            frame = frame.loc[
                (timestamp >= earliest)
                & (timestamp <= config.as_of)
                & (available_at <= config.as_of)
            ]
            filtered[name] = frame
        return filtered

    def _apply_warm_up(
        self,
        features: pd.DataFrame,
        definition: FeatureDefinition,
        config: FeatureBuildConfig,
    ) -> pd.DataFrame:
        if config.timestamp_column in features and definition.warm_up_period > pd.Timedelta(0):
            start = config.as_of - definition.lookback
            return features.loc[pd.to_datetime(features[config.timestamp_column]) >= start].copy()
        return features

    def _apply_missing_policy(
        self, features: pd.DataFrame, definition: FeatureDefinition
    ) -> pd.DataFrame:
        if (
            definition.missing_value_policy is MissingValuePolicy.ERROR
            and features.isna().any().any()
        ):
            raise ValueError(f"feature {definition.name!r} produced missing values")
        if definition.missing_value_policy is MissingValuePolicy.DROP:
            return features.dropna().copy()
        if definition.missing_value_policy is MissingValuePolicy.FORWARD_FILL:
            return features.ffill()
        return features

    def _apply_scaling(
        self,
        features: pd.DataFrame,
        definition: FeatureDefinition,
        config: FeatureBuildConfig,
    ) -> pd.DataFrame:
        if definition.scaling_method is ScalingMethod.NONE:
            return features
        if definition.fitting_requirement is not FittingRequirement.TRAINING_WINDOW_ONLY:
            raise ValueError("scaling requires training-window-only fitting")
        if config.training_start is None or config.training_end is None:
            raise ValueError("training_start and training_end are required for fitted features")
        if config.training_end > config.as_of:
            raise ValueError("training window ends after as_of and would require future data")
        output_columns = [field.name for field in definition.output_schema]
        train_mask = pd.Series(True, index=features.index)
        if config.timestamp_column in features:
            ts = pd.to_datetime(features[config.timestamp_column])
            train_mask = (ts >= config.training_start) & (ts <= config.training_end)
        train = features.loc[train_mask, output_columns]
        if train.empty:
            raise ValueError("training window contains no feature rows")
        scaled = features.copy()
        if definition.scaling_method is ScalingMethod.STANDARD:
            std = train.std(ddof=0).replace(0, 1)
            scaled.loc[:, output_columns] = (scaled[output_columns] - train.mean()) / std
        elif definition.scaling_method is ScalingMethod.MIN_MAX:
            span = (train.max() - train.min()).replace(0, 1)
            scaled.loc[:, output_columns] = (scaled[output_columns] - train.min()) / span
        return scaled

    def _validate_output_schema(
        self, features: pd.DataFrame, definition: FeatureDefinition
    ) -> None:
        missing = [field.name for field in definition.output_schema if field.name not in features]
        if missing:
            raise ValueError(f"feature {definition.name!r} missing output columns {missing}")
        for field_def in definition.output_schema:
            if not field_def.nullable and features[field_def.name].isna().any():
                raise ValueError(f"feature output {field_def.name!r} is not nullable")

    def _cache_key(
        self, definition: FeatureDefinition, inputs: RawInputFrames, config: FeatureBuildConfig
    ) -> str:
        input_hashes = {
            name: _dataframe_hash(inputs[name])
            for name in definition.cache_key_inputs
            if name in inputs
        }
        return stable_hash(
            {
                "definition": definition.to_record(),
                "config": config.to_record(),
                "inputs": input_hashes,
            }
        )


__all__ = [
    "FeatureBuildConfig",
    "FeatureBuildResult",
    "FeatureBuilder",
    "FeatureDefinition",
    "FeatureRegistry",
    "FeatureSemantics",
    "FittingRequirement",
    "MissingValuePolicy",
    "OutputField",
    "ScalingMethod",
]
