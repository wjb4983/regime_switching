"""Strict, discriminated configuration schema for evaluation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationSource(StrictModel):
    """Exactly one model artifact or experiment group to evaluate."""

    model_artifact: Path | None = None
    experiment_group: str | None = None
    model: str | None = None
    model_parameters: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        choices = (self.model_artifact, self.experiment_group, self.model)
        if sum(value is not None for value in choices) != 1:
            raise ValueError(
                "source requires exactly one of model_artifact, experiment_group, or model"
            )


class SplitConfig(StrictModel):
    kind: Literal["expanding", "rolling", "purged"]
    initial_train_size: int = Field(default=20, ge=1)
    train_size: int = Field(default=20, ge=1)
    validation_size: int = Field(default=0, ge=0)
    test_size: int = Field(default=5, ge=1)
    step: int = Field(default=5, ge=1)
    embargo: int = Field(default=0, ge=0)


class EvaluationBase(StrictModel):
    source: EvaluationSource
    dataset: Path
    output_dir: Path
    splitter: SplitConfig
    run_id: str = "evaluation"
    features: list[str] = Field(min_length=1)
    timestamp_column: str | None = None
    metrics: list[str] = Field(default_factory=list)
    comparison_contract: dict[str, Any] = Field(default_factory=dict)
    retraining_schedule: str = "each_window"
    execution_delay: int = Field(default=0, ge=0)
    cost_assumptions: dict[str, Any] = Field(default_factory=dict)
    decision_rules: dict[str, Any] = Field(default_factory=dict)


class ValidationEvaluation(EvaluationBase):
    evaluation_type: Literal["validation"]


class RegimeQualityEvaluation(EvaluationBase):
    evaluation_type: Literal["regime_quality"]


class ForecastingEvaluation(EvaluationBase):
    evaluation_type: Literal["forecasting"]
    target_column: str


class EconomicEvaluation(EvaluationBase):
    evaluation_type: Literal["economic"]
    returns_column: str


class DownstreamPolicyEvaluation(EvaluationBase):
    evaluation_type: Literal["downstream_policy"]
    target_column: str
    decision_rules: dict[str, Any] = Field(min_length=1)


EvaluationWorkflowConfig: TypeAlias = Annotated[
    ValidationEvaluation
    | RegimeQualityEvaluation
    | ForecastingEvaluation
    | EconomicEvaluation
    | DownstreamPolicyEvaluation,
    Field(discriminator="evaluation_type"),
]
_ADAPTER = TypeAdapter(EvaluationWorkflowConfig)


def parse_evaluation_config(value: Any) -> EvaluationWorkflowConfig:
    """Parse an evaluation only through its required typed discriminator."""
    return _ADAPTER.validate_python(value)


__all__ = ["EvaluationWorkflowConfig", "parse_evaluation_config"]
