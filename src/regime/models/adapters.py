"""Explicit contract adapters for predictors with non-model semantics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

from pydantic import Field

from regime.models.base import ModelMetadata, RegimeModel, RegimeModelConfig
from regime.models.rules.baselines import (
    Direction,
    StaticThresholdRule,
    StaticThresholdRuleConfig,
    ThresholdRuleConfig,
)


class VolatilityThresholdConfig(RegimeModelConfig):
    """Backward-compatible typed configuration for the rule baseline YAML."""

    model_name: str = "volatility-threshold"
    feature: str = Field(min_length=1)
    threshold: float
    direction: Direction = Direction.ABOVE
    labels: Mapping[str, int] = Field(default_factory=lambda: {"risk_on": 0, "risk_off": 1})


class RuleRegimeModelAdapter(RegimeModel):
    """Adapt a deterministic rule to the fit/predict contract without hiding its semantics."""

    def __init__(self, config: VolatilityThresholdConfig) -> None:
        self.config = config
        rule_config = StaticThresholdRuleConfig(
            name=config.model_name,
            thresholds=(
                ThresholdRuleConfig(
                    feature=config.feature,
                    threshold=config.threshold,
                    direction=config.direction,
                    risk_on_label=config.labels.get("risk_on", 0),
                    risk_off_label=config.labels.get("risk_off", 1),
                ),
            ),
            hard_label_mapping=config.labels,
        )
        self.rule = StaticThresholdRule(rule_config)
        self._metadata = ModelMetadata(
            model_name=config.model_name,
            model_version=config.model_version,
            n_states=config.n_states,
            config_hash=config.config_hash(),
            attributes={"semantics": "deterministic_rule_adapter"},
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def fit(self, dataset: Any, config: RegimeModelConfig | None = None) -> Self:
        """Rules have no learned parameters; retain data-independent configuration."""
        return self

    def predict(self, dataset: Any) -> Sequence[int]:
        signal = self.rule.predict(dataset)
        return [int(label) for label in signal.labels if label is not None]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.config.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(VolatilityThresholdConfig.model_validate(payload))


__all__ = ["RuleRegimeModelAdapter", "VolatilityThresholdConfig"]
