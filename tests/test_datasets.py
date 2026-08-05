"""Dataset assembly tests."""

from __future__ import annotations

import pandas as pd
import pytest

from regime.datasets import (
    AssetUniverseSnapshot,
    DatasetBuilder,
    ExecutionDelay,
    MissingValuePolicy,
    ProbabilityMode,
    RegimeDataset,
    SplitMetadata,
    SplitName,
    TimeWindow,
)
from regime.errors import RegimeDataError


def _splits() -> SplitMetadata:
    return SplitMetadata(
        train=TimeWindow(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03")),
        validation=TimeWindow(pd.Timestamp("2020-01-04"), pd.Timestamp("2020-01-04")),
        test=TimeWindow(pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-05")),
    )


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.RangeIndex(5)
    features = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=5, freq="D"),
            "asset": ["A", "A", "B", "B", "C"],
            "value": [1.0, 2.0, 3.0, 100.0, 200.0],
            "sector": ["x", "x", "y", "z", "z"],
        },
        index=index,
    )
    targets = pd.DataFrame({"forward_return": [0.1, 0.2, -0.1, 0.4, 0.5]}, index=index)
    return features, targets


def test_dataset_builder_fits_preprocessing_on_training_split_only() -> None:
    """Validation/test rows are transformed by estimators fit on training rows."""
    features, targets = _frames()
    dataset = DatasetBuilder(
        scaler="standard",
        categorical_columns=("sector",),
        execution_delay=ExecutionDelay(decision_delay=pd.Timedelta(days=1)),
    ).build(
        features,
        targets,
        splits=_splits(),
        asset_universe=("A", "B", "C"),
        feature_columns=("value", "sector"),
    )

    train_x, train_y = dataset.split(SplitName.TRAIN)

    assert dataset.dataset_hash
    assert dataset.asset_universe.assets == ("A", "B", "C")
    assert dataset.preprocessing_metadata["fit_split"] == "train"
    assert train_y["forward_return"].tolist() == [0.1, 0.2, -0.1]
    assert pytest.approx(train_x["feature__0"].mean(), abs=1e-12) == 0.0
    assert "available_at" in dataset.X.columns


def test_live_equivalent_dataset_rejects_smoothed_probabilities() -> None:
    """Smoothed probabilities require explicit offline diagnostic mode."""
    features, targets = _frames()

    with pytest.raises(RegimeDataError, match="smoothed probabilities"):
        DatasetBuilder().build(
            features,
            targets,
            splits=_splits(),
            asset_universe=("A", "B", "C"),
            smoothed_probability_columns=("smoothed_p_bull",),
        )

    offline = DatasetBuilder(
        probability_mode=ProbabilityMode.OFFLINE_DIAGNOSTICS,
        allow_smoothed_probabilities=True,
    ).build(features, targets, splits=_splits(), asset_universe=("A", "B", "C"))

    assert offline.probability_mode is ProbabilityMode.OFFLINE_DIAGNOSTICS


def test_lazy_loading_round_trips_large_dataset(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Lazy output writes parquet files and loads copies on access."""
    features, targets = _frames()
    dataset = DatasetBuilder(missing_value_policy=MissingValuePolicy.LEAVE).build(
        features,
        targets,
        splits=_splits(),
        asset_universe=AssetUniverseSnapshot(pd.Timestamp("2020-01-05"), ("A", "B", "C")),
        lazy_output_dir=tmp_path,
    )

    assert isinstance(dataset, RegimeDataset)
    assert (tmp_path / "features.parquet").exists()
    assert dataset.X.shape[0] == 5
