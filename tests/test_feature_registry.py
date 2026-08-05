"""Point-in-time feature registry tests."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd  # type: ignore[import-untyped]
import pytest

from regime.features import (
    FeatureBuildConfig,
    FeatureBuilder,
    FeatureDefinition,
    FeatureRegistry,
    FeatureSemantics,
    FittingRequirement,
    MissingValuePolicy,
    OutputField,
    ScalingMethod,
)


def _definition() -> FeatureDefinition:
    def transform(inputs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        prices = inputs["prices"].copy()
        prices["return"] = prices["close"].pct_change().fillna(0.0)
        return prices[["timestamp", "return"]]

    return FeatureDefinition(
        name="returns",
        version="1.0.0",
        required_raw_inputs=("prices",),
        lookback=pd.Timedelta(days=3),
        publication_lag=pd.Timedelta(hours=1),
        warm_up_period=pd.Timedelta(days=1),
        missing_value_policy=MissingValuePolicy.ERROR,
        scaling_method=ScalingMethod.STANDARD,
        semantics=FeatureSemantics.TIME_SERIES,
        fitting_requirement=FittingRequirement.TRAINING_WINDOW_ONLY,
        leakage_risks=("revisions published after as_of", "scaler fit on validation rows"),
        output_schema=(OutputField("return", "float64", nullable=False),),
        cache_key_inputs=("prices",),
        transform=transform,
    )


def test_feature_builder_filters_fits_and_caches_point_in_time_outputs() -> None:
    registry = FeatureRegistry()
    registry.register(_definition())
    builder = FeatureBuilder(registry)
    raw = {
        "prices": pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
                "published_at": pd.date_range("2024-01-01", periods=5, freq="D"),
                "close": [100.0, 101.0, 103.0, 102.0, 104.0],
            }
        )
    }
    config = FeatureBuildConfig(
        as_of=pd.Timestamp("2024-01-04 02:00:00"),
        training_start=pd.Timestamp("2024-01-02"),
        training_end=pd.Timestamp("2024-01-03"),
    )

    first = builder.build("returns", raw, config)
    second = builder.build("returns", raw, config)

    assert not first.cache_hit
    assert second.cache_hit
    assert first.cache_key == second.cache_key
    assert first.feature_hash == second.feature_hash
    assert first.features["timestamp"].max() <= config.as_of
    assert first.provenance["definition"]["name"] == "returns"
    assert first.provenance["raw_input_hashes"]["prices"]


def test_feature_builder_rejects_training_windows_after_as_of() -> None:
    registry = FeatureRegistry()
    registry.register(_definition())
    builder = FeatureBuilder(registry)
    raw = {
        "prices": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-03")],
                "published_at": [pd.Timestamp("2024-01-03")],
                "close": [100.0],
            }
        )
    }
    config = FeatureBuildConfig(
        as_of=pd.Timestamp("2024-01-04"),
        training_start=pd.Timestamp("2024-01-02"),
        training_end=pd.Timestamp("2024-01-05"),
    )

    with pytest.raises(ValueError, match="future data"):
        builder.build("returns", raw, config)
