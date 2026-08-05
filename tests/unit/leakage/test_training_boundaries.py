import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler

from regime.validation import (
    fit_on_training_window,
    require_adjustment_status,
    validate_probability_usage,
)


@pytest.mark.parametrize(
    "estimator",
    [
        pytest.param(StandardScaler(), id="scaler"),
        pytest.param(PCA(n_components=1), id="pca"),
        pytest.param(SelectKBest(k=1), id="feature-selector"),
    ],
)
def test_transformers_fit_only_on_training_window(estimator) -> None:
    features = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [1_000.0, 2_000.0]])

    fit_on_training_window(estimator, features, [0, 1, 2, 3], labels=np.array([0, 0, 1, 1, 1]))

    assert estimator.n_features_in_ == 2
    if hasattr(estimator, "mean_"):
        assert estimator.mean_.tolist() == [1.5, 2.5]
    if hasattr(estimator, "n_samples_seen_"):
        assert estimator.n_samples_seen_ == 4


class RecordingEstimator:
    def fit(self, features, labels=None):
        self.seen = list(features.index)
        return self


@pytest.mark.parametrize("component", ["surface-model", "calibrator"])
def test_fitted_models_never_see_validation_or_test_rows(component: str) -> None:
    estimator = RecordingEstimator()
    features = pd.DataFrame({"value": range(6)}, index=pd.date_range("2024-01-01", periods=6))

    fit_on_training_window(estimator, features, [0, 1, 2])

    assert estimator.seen == list(features.index[:3]), component


def test_corporate_action_adjustment_status_must_match_strategy_input() -> None:
    adjusted = pd.DataFrame({"close": [50.0], "adjustment_status": ["split_adjusted"]})
    raw = pd.DataFrame({"close": [100.0], "adjustment_status": ["raw"]})

    require_adjustment_status(adjusted, "split_adjusted")
    with pytest.raises(ValueError, match="expected adjustment_status"):
        require_adjustment_status(raw, "split_adjusted")


def test_smoothed_probabilities_are_blocked_from_live_backtests_by_default() -> None:
    validate_probability_usage("filtered")
    with pytest.raises(ValueError, match="hindsight-only"):
        validate_probability_usage("smoothed")
    validate_probability_usage("smoothed", live_equivalent=False)
