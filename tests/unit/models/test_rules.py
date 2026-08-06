import pandas as pd

from regime.models.adapters import RuleRegimeModelAdapter, VolatilityThresholdConfig
from regime.models.rules import (
    CompositeRiskRule,
    CompositeRiskRuleConfig,
    Direction,
    HysteresisRule,
    HysteresisRuleConfig,
    PercentileRule,
    PercentileRuleConfig,
    StaticThresholdRule,
    StaticThresholdRuleConfig,
    ThresholdRuleConfig,
    build_breadth_threshold_config,
    build_correlation_threshold_config,
    build_credit_threshold_config,
    build_liquidity_threshold_config,
    build_skew_threshold_config,
    build_term_structure_threshold_config,
    build_trend_threshold_config,
    build_volatility_threshold_config,
)


def test_static_threshold_rule_labels_and_probabilities() -> None:
    rule = StaticThresholdRule(
        StaticThresholdRuleConfig(
            name="volatility",
            thresholds=(ThresholdRuleConfig(feature="vol", threshold=0.2),),
        )
    )

    result = rule.predict(pd.DataFrame({"vol": [0.1, 0.25]}))

    assert result.labels == (0, 1)
    assert result.probabilities == (0.0, 1.0)


def test_rule_adapter_metadata_populates_fit_metadata() -> None:
    model = RuleRegimeModelAdapter(
        VolatilityThresholdConfig(model_name="volatility-threshold", feature="vol", threshold=0.2)
    )

    model.fit(pd.DataFrame({"vol": [0.1, 0.3]}))
    metadata = model.metadata.model_dump(mode="json")

    assert metadata["fitted_at"] is not None
    assert metadata["training_observations"] == 2


def test_percentile_rule_uses_rolling_window() -> None:
    rule = PercentileRule(
        PercentileRuleConfig(feature="x", percentile=0.5, percentile_window=3, min_periods=2)
    )

    result = rule.predict({"x": [1.0, 2.0, 3.0, 1.0]})

    assert result.labels == (None, 1, 1, 0)


def test_hysteresis_rule_keeps_state_inside_band() -> None:
    rule = HysteresisRule(
        HysteresisRuleConfig(feature="x", enter_threshold=10.0, exit_threshold=7.0)
    )

    result = rule.predict({"x": [8.0, 10.5, 8.0, 6.5]})

    assert result.labels == (0, 1, 1, 0)


def test_composite_rule_weights_and_calibrates_score() -> None:
    rule = CompositeRiskRule(
        CompositeRiskRuleConfig(
            thresholds=(
                ThresholdRuleConfig(feature="vol", threshold=0.2),
                ThresholdRuleConfig(feature="trend", threshold=0.0, direction=Direction.BELOW),
            ),
            feature_weights={"vol": 3.0, "trend": 1.0},
        )
    )

    result = rule.predict({"vol": [0.3, 0.1], "trend": [1.0, -1.0]})

    assert result.scores == (0.75, 0.25)
    assert result.labels == (1, 0)
    assert all(probability is not None for probability in result.probabilities)


def test_domain_threshold_config_builders_cover_requested_baselines() -> None:
    configs = [
        build_volatility_threshold_config(),
        build_correlation_threshold_config(),
        build_liquidity_threshold_config(),
        build_breadth_threshold_config(),
        build_trend_threshold_config(),
        build_credit_threshold_config(),
        build_skew_threshold_config(),
        build_term_structure_threshold_config(),
    ]

    assert [config.name for config in configs] == [
        "volatility",
        "correlation",
        "liquidity",
        "breadth",
        "trend",
        "credit",
        "skew",
        "term_structure",
    ]
