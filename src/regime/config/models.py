"""Typed Pydantic configuration models for regime-switching workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from regime.config.base import RegimeBaseConfig


class DataIngestionConfig(RegimeBaseConfig):
    """Input data acquisition and storage settings."""

    _path_fields = frozenset({"source", "cache_dir"})

    name: str = Field(min_length=1)
    source: Path
    source_type: Literal["csv", "parquet", "duckdb", "api"]
    timestamp_column: str = "timestamp"
    symbol_column: str | None = None
    cache_dir: Path | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class FeatureConfig(RegimeBaseConfig):
    """Feature engineering settings."""

    name: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)
    transforms: list[str] = Field(default_factory=list)
    lookback: int = Field(default=1, ge=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class DatasetAssemblyConfig(RegimeBaseConfig):
    """Dataset construction settings."""

    _path_fields = frozenset({"output_path"})

    name: str = Field(min_length=1)
    data: DataIngestionConfig
    features: list[FeatureConfig] = Field(default_factory=list)
    target: str
    split_column: str | None = None
    output_path: Path | None = None

    @field_validator("features")
    @classmethod
    def _require_unique_feature_names(cls, value: list[FeatureConfig]) -> list[FeatureConfig]:
        names = [feature.name for feature in value]
        if len(names) != len(set(names)):
            raise ValueError("Feature names must be unique so downstream datasets are unambiguous.")
        return value


class ModelConfig(RegimeBaseConfig):
    """Model family and hyperparameter settings."""

    _path_fields = frozenset({"artifact_dir"})

    name: str = Field(min_length=1)
    model_type: Literal["hmm", "markov_switching", "changepoint", "classifier", "regressor"]
    n_regimes: int = Field(default=2, ge=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    random_state: int | None = Field(default=None, ge=0)
    artifact_dir: Path | None = None


class ValidationConfig(RegimeBaseConfig):
    """Model validation strategy settings."""

    strategy: Literal["holdout", "walk_forward", "blocked_cv", "purged_cv"] = "holdout"
    train_size: float = Field(default=0.7, gt=0, lt=1)
    test_size: float | None = Field(default=None, gt=0, lt=1)
    n_splits: int = Field(default=1, ge=1)
    embargo_periods: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_split_sizes(self) -> ValidationConfig:
        if self.test_size is not None and self.train_size + self.test_size > 1:
            raise ValueError("train_size + test_size must be <= 1.0; reduce one split fraction.")
        if self.strategy != "holdout" and self.n_splits < 2:
            raise ValueError("Cross-validation strategies require n_splits >= 2.")
        return self


class EvaluationConfig(RegimeBaseConfig):
    """Evaluation metric and output settings."""

    _path_fields = frozenset({"output_dir"})

    metrics: list[str] = Field(default_factory=lambda: ["accuracy"])
    baseline: str | None = None
    output_dir: Path | None = None
    save_predictions: bool = True

    @field_validator("metrics")
    @classmethod
    def _require_metrics(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("At least one evaluation metric is required.")
        return value


class BacktestConfig(RegimeBaseConfig):
    """Backtesting assumptions and cost settings."""

    _path_fields = frozenset({"output_dir"})

    initial_capital: float = Field(default=100_000.0, gt=0)
    commission_bps: float = Field(default=0.0, ge=0)
    slippage_bps: float = Field(default=0.0, ge=0)
    rebalance_frequency: str = "1D"
    risk_free_rate: float = 0.0
    output_dir: Path | None = None


class ReportConfig(RegimeBaseConfig):
    """Report rendering settings."""

    _path_fields = frozenset({"output_dir", "template_path"})

    title: str = Field(min_length=1)
    format: Literal["html", "markdown", "pdf", "json"] = "html"
    output_dir: Path
    template_path: Path | None = None
    include_sections: list[str] = Field(default_factory=lambda: ["summary", "metrics"])


class ExperimentConfig(RegimeBaseConfig):
    """Top-level experiment configuration composed from workflow config sections."""

    _path_fields = frozenset({"work_dir"})

    name: str = Field(min_length=1)
    work_dir: Path = Path(".")
    dataset: DatasetAssemblyConfig
    model: ModelConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    backtest: BacktestConfig | None = None
    report: ReportConfig | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
