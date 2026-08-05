"""Typed rule-based baseline regime classifiers.

The classes in this module are intentionally simple, deterministic baselines for
later model evaluations.  They transform tabular feature observations into a
hard risk-on/risk-off label and, where defensible, a calibrated probability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from math import exp
from typing import Self

import pandas as pd
from pydantic import Field, PositiveInt, field_validator, model_validator

from regime.config.base import RegimeBaseConfig

HardLabel = int


class Direction(StrEnum):
    """Direction in which a larger feature value changes risk-off evidence."""

    ABOVE = "above"
    BELOW = "below"


class MissingDataPolicy(StrEnum):
    """How rules handle missing observations."""

    PROPAGATE = "propagate"
    RISK_OFF = "risk_off"
    RISK_ON = "risk_on"
    IGNORE = "ignore"


class CalibrationMethod(StrEnum):
    """Supported deterministic score-to-probability transforms."""

    NONE = "none"
    LINEAR = "linear"
    LOGISTIC = "logistic"


class RuleSignal(RegimeBaseConfig):
    """Rule output with labels and optional risk-off probabilities."""

    labels: tuple[HardLabel | None, ...]
    probabilities: tuple[float | None, ...] = ()
    scores: tuple[float | None, ...] = ()


class ThresholdRuleConfig(RegimeBaseConfig):
    """Typed configuration for one thresholded feature."""

    feature: str = Field(min_length=1)
    threshold: float
    direction: Direction = Direction.ABOVE
    risk_off_label: HardLabel = 1
    risk_on_label: HardLabel = 0
    weight: float = Field(default=1.0, ge=0.0)
    missing_data_policy: MissingDataPolicy = MissingDataPolicy.PROPAGATE


class ThresholdRuleSetConfig(RegimeBaseConfig):
    """Named group of static threshold rules for one risk domain."""

    name: str = Field(min_length=1)
    thresholds: tuple[ThresholdRuleConfig, ...]
    hard_label_mapping: Mapping[str, HardLabel] = Field(
        default_factory=lambda: {"risk_on": 0, "risk_off": 1}
    )
    calibration_method: CalibrationMethod = CalibrationMethod.LINEAR

    @field_validator("thresholds")
    @classmethod
    def _require_thresholds(
        cls, value: tuple[ThresholdRuleConfig, ...]
    ) -> tuple[ThresholdRuleConfig, ...]:
        if not value:
            raise ValueError("at least one threshold rule is required")
        return value


class StaticThresholdRuleConfig(ThresholdRuleSetConfig):
    """Configuration for static threshold rules."""


class PercentileRuleConfig(RegimeBaseConfig):
    """Configuration for rolling percentile threshold rules."""

    feature: str = Field(min_length=1)
    percentile: float = Field(ge=0.0, le=1.0)
    percentile_window: PositiveInt
    min_periods: PositiveInt = 1
    direction: Direction = Direction.ABOVE
    missing_data_policy: MissingDataPolicy = MissingDataPolicy.PROPAGATE
    hard_label_mapping: Mapping[str, HardLabel] = Field(
        default_factory=lambda: {"risk_on": 0, "risk_off": 1}
    )
    calibration_method: CalibrationMethod = CalibrationMethod.NONE

    @model_validator(mode="after")
    def _validate_min_periods(self) -> Self:
        if self.min_periods > self.percentile_window:
            raise ValueError("min_periods cannot exceed percentile_window")
        return self


class HysteresisRuleConfig(RegimeBaseConfig):
    """Configuration for stateful hysteresis rules."""

    feature: str = Field(min_length=1)
    enter_threshold: float
    exit_threshold: float
    direction: Direction = Direction.ABOVE
    initial_label: HardLabel = 0
    missing_data_policy: MissingDataPolicy = MissingDataPolicy.PROPAGATE
    hard_label_mapping: Mapping[str, HardLabel] = Field(
        default_factory=lambda: {"risk_on": 0, "risk_off": 1}
    )

    @model_validator(mode="after")
    def _validate_band(self) -> Self:
        if self.direction == Direction.ABOVE and self.exit_threshold > self.enter_threshold:
            raise ValueError("exit_threshold must be <= enter_threshold for above hysteresis")
        if self.direction == Direction.BELOW and self.exit_threshold < self.enter_threshold:
            raise ValueError("exit_threshold must be >= enter_threshold for below hysteresis")
        return self


class CompositeRiskRuleConfig(RegimeBaseConfig):
    """Configuration for a weighted composite risk-on/risk-off score."""

    thresholds: tuple[ThresholdRuleConfig, ...]
    lookbacks: Mapping[str, PositiveInt] = Field(default_factory=dict)
    hysteresis_bands: Mapping[str, tuple[float, float]] = Field(default_factory=dict)
    feature_weights: Mapping[str, float] = Field(default_factory=dict)
    missing_data_policy: MissingDataPolicy = MissingDataPolicy.IGNORE
    hard_label_mapping: Mapping[str, HardLabel] = Field(
        default_factory=lambda: {"risk_on": 0, "risk_off": 1}
    )
    decision_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    calibration_method: CalibrationMethod = CalibrationMethod.LOGISTIC


def _frame(data: pd.DataFrame | Mapping[str, Sequence[float]]) -> pd.DataFrame:
    return data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)


def _risk_off(value: float, threshold: float, direction: Direction) -> bool:
    return value >= threshold if direction == Direction.ABOVE else value <= threshold


def _label_missing(
    policy: MissingDataPolicy, previous: HardLabel | None = None
) -> HardLabel | None:
    if policy == MissingDataPolicy.RISK_OFF:
        return 1
    if policy == MissingDataPolicy.RISK_ON:
        return 0
    if policy == MissingDataPolicy.PROPAGATE:
        return previous
    return None


def _calibrate(score: float | None, method: CalibrationMethod) -> float | None:
    if score is None:
        return None
    if method == CalibrationMethod.NONE:
        return None
    if method == CalibrationMethod.LINEAR:
        return min(1.0, max(0.0, score))
    return 1.0 / (1.0 + exp(-8.0 * (score - 0.5)))


class StaticThresholdRule:
    """Evaluate one or more static threshold rules."""

    def __init__(self, config: StaticThresholdRuleConfig) -> None:
        self.config = config

    def predict(self, data: pd.DataFrame | Mapping[str, Sequence[float]]) -> RuleSignal:
        frame = _frame(data)
        labels: list[HardLabel | None] = []
        scores: list[float | None] = []
        probabilities: list[float | None] = []
        for _, row in frame.iterrows():
            score = 0.0
            total = 0.0
            for rule in self.config.thresholds:
                value = row.get(rule.feature)
                if pd.isna(value):
                    label = _label_missing(rule.missing_data_policy)
                    if label is None:
                        continue
                    score += rule.weight * label
                else:
                    risk_off = _risk_off(float(value), rule.threshold, rule.direction)
                    score += rule.weight * float(risk_off)
                total += rule.weight
            normalized = None if total == 0.0 else score / total
            scores.append(normalized)
            labels.append(None if normalized is None else int(normalized >= 0.5))
            probabilities.append(_calibrate(normalized, self.config.calibration_method))
        return RuleSignal(
            labels=tuple(labels), probabilities=tuple(probabilities), scores=tuple(scores)
        )


class PercentileRule:
    """Compare features with rolling percentile thresholds."""

    def __init__(self, config: PercentileRuleConfig) -> None:
        self.config = config

    def predict(self, data: pd.DataFrame | Mapping[str, Sequence[float]]) -> RuleSignal:
        series = _frame(data)[self.config.feature].astype(float)
        thresholds = series.rolling(
            self.config.percentile_window, min_periods=self.config.min_periods
        ).quantile(self.config.percentile)
        labels: list[HardLabel | None] = []
        for value, threshold in zip(series, thresholds, strict=True):
            if pd.isna(value) or pd.isna(threshold):
                labels.append(_label_missing(self.config.missing_data_policy))
            else:
                labels.append(int(_risk_off(float(value), float(threshold), self.config.direction)))
        return RuleSignal(labels=tuple(labels))


class HysteresisRule:
    """Stateful threshold rule with separate enter and exit bands."""

    def __init__(self, config: HysteresisRuleConfig) -> None:
        self.config = config

    def predict(self, data: pd.DataFrame | Mapping[str, Sequence[float]]) -> RuleSignal:
        state: HardLabel = self.config.initial_label
        labels: list[HardLabel | None] = []
        for value in _frame(data)[self.config.feature].astype(float):
            if pd.isna(value):
                label = _label_missing(self.config.missing_data_policy, state)
                labels.append(label)
                if label is not None:
                    state = label
                continue
            if _risk_off(float(value), self.config.enter_threshold, self.config.direction):
                state = 1
            elif not _risk_off(float(value), self.config.exit_threshold, self.config.direction):
                state = 0
            labels.append(state)
        return RuleSignal(labels=tuple(labels))


class CompositeRiskRule:
    """Weighted composite risk-on/risk-off baseline score."""

    def __init__(self, config: CompositeRiskRuleConfig) -> None:
        self.config = config

    def predict(self, data: pd.DataFrame | Mapping[str, Sequence[float]]) -> RuleSignal:
        base = StaticThresholdRule(
            StaticThresholdRuleConfig(
                name="composite",
                thresholds=tuple(
                    rule.model_copy(
                        update={
                            "weight": self.config.feature_weights.get(rule.feature, rule.weight)
                        }
                    )
                    for rule in self.config.thresholds
                ),
                calibration_method=CalibrationMethod.LINEAR,
            )
        ).predict(data)
        scores = base.scores
        labels = tuple(
            None if score is None else int(score >= self.config.decision_threshold)
            for score in scores
        )
        probabilities = tuple(_calibrate(score, self.config.calibration_method) for score in scores)
        return RuleSignal(labels=labels, probabilities=probabilities, scores=scores)


def _domain_config(
    name: str, feature: str, threshold: float, direction: Direction
) -> StaticThresholdRuleConfig:
    threshold_rule = ThresholdRuleConfig(feature=feature, threshold=threshold, direction=direction)
    return StaticThresholdRuleConfig(name=name, thresholds=(threshold_rule,))


def build_volatility_threshold_config(
    feature: str = "volatility", threshold: float = 0.2
) -> StaticThresholdRuleConfig:
    """Build the default volatility threshold baseline config."""
    return _domain_config("volatility", feature, threshold, Direction.ABOVE)


def build_correlation_threshold_config(
    feature: str = "correlation", threshold: float = 0.7
) -> StaticThresholdRuleConfig:
    """Build the default correlation threshold baseline config."""
    return _domain_config("correlation", feature, threshold, Direction.ABOVE)


def build_liquidity_threshold_config(
    feature: str = "liquidity", threshold: float = 0.0
) -> StaticThresholdRuleConfig:
    """Build the default liquidity threshold baseline config."""
    return _domain_config("liquidity", feature, threshold, Direction.BELOW)


def build_breadth_threshold_config(
    feature: str = "breadth", threshold: float = 0.5
) -> StaticThresholdRuleConfig:
    """Build the default breadth threshold baseline config."""
    return _domain_config("breadth", feature, threshold, Direction.BELOW)


def build_trend_threshold_config(
    feature: str = "trend", threshold: float = 0.0
) -> StaticThresholdRuleConfig:
    """Build the default trend threshold baseline config."""
    return _domain_config("trend", feature, threshold, Direction.BELOW)


def build_credit_threshold_config(
    feature: str = "credit", threshold: float = 0.0
) -> StaticThresholdRuleConfig:
    """Build the default credit threshold baseline config."""
    return _domain_config("credit", feature, threshold, Direction.ABOVE)


def build_skew_threshold_config(
    feature: str = "skew", threshold: float = 0.0
) -> StaticThresholdRuleConfig:
    """Build the default skew threshold baseline config."""
    return _domain_config("skew", feature, threshold, Direction.BELOW)


def build_term_structure_threshold_config(
    feature: str = "term_structure", threshold: float = 0.0
) -> StaticThresholdRuleConfig:
    """Build the default term-structure threshold baseline config."""
    return _domain_config("term_structure", feature, threshold, Direction.BELOW)
