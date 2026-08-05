import numpy as np
import pytest

from regime.models.supervised import (
    SupervisedRegimeClassifier,
    SupervisedRegimeConfig,
    TransitionHazardConfig,
    TransitionHazardModel,
    transition_events,
)


def make_data(n=60):
    x0 = np.linspace(-2, 2, n)
    x1 = np.sin(np.arange(n) / 4)
    x = np.column_stack([x0, x1])
    y = (x0 + x1 > 0).astype(int)
    return x, y


@pytest.mark.parametrize(
    "classifier", ["logistic_regression", "random_forest", "gradient_boosted_trees"]
)
def test_supervised_classifier_outputs(classifier):
    x, y = make_data()
    model = SupervisedRegimeClassifier(
        SupervisedRegimeConfig(
            classifier=classifier,
            n_states=2,
            random_seed=7,
            cv=3,
            feature_names=("trend", "cycle"),
            label_source="economic_definition",
            label_definition="positive combined trend/cycle indicates expansion",
        )
    ).fit(x, labels=y)

    predictions = model.predict_full(x[:5])
    assert len(predictions) == 5
    assert all(result.predicted_class in {0, 1} for result in predictions)
    assert all(abs(sum(result.calibrated_probabilities) - 1.0) < 1e-9 for result in predictions)

    report = model.report(x, y)
    assert set(report.feature_importance) == {"trend", "cycle"}
    assert "log_loss" in report.calibration_diagnostics
    assert report.model_card["intended_outputs"] == [
        "predicted_class",
        "calibrated_probabilities",
        "feature_importance",
        "calibration_diagnostics",
    ]


def test_pseudo_label_replication_risk_required_and_reported():
    with pytest.raises(ValueError, match="pseudo-label training requires"):
        SupervisedRegimeConfig(label_source="unsupervised_pseudo_label")

    config = SupervisedRegimeConfig(
        label_source="unsupervised_pseudo_label",
        label_definition="KMeans cluster id",
        pseudo_label_source_model="kmeans:v1",
        pseudo_label_replication_risk=(
            "May reproduce KMeans label instability and clustering errors."
        ),
    )
    model = SupervisedRegimeClassifier(config)
    assert model.report().pseudo_label_replication_risk.startswith("May reproduce")
    assert model.report().model_card["pseudo_label_source_model"] == "kmeans:v1"


def test_transition_hazard_model_outputs_event_probability():
    x, y = make_data()
    events = transition_events(y, horizon=1)
    assert set(events) <= {0, 1}

    model = TransitionHazardModel(
        TransitionHazardConfig(
            classifier="logistic_regression",
            n_states=2,
            label_source="synthetic_known_state",
            label_definition="known simulated state",
            cv=3,
        )
    ).fit(x, labels=y)
    results = model.predict_full(x[:8])
    assert len(results) == 8
    assert all(result.transition_or_event_probability is not None for result in results)
    assert all(0.0 <= result.transition_or_event_probability <= 1.0 for result in results)
