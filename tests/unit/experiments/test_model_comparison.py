from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from regime.experiments.model_comparison import (
    discover_model_configs,
    evaluate_predictions,
    rank_models,
    render_comparison_report,
)


def test_discover_model_configs_returns_repo_relative_entries() -> None:
    entries = discover_model_configs(
        Path("configs/models"),
        relative_to=Path("configs/experiments"),
    )

    assert entries
    assert entries[0]["config"].startswith("../models/")
    assert all("label" in entry and "family" in entry for entry in entries)


def test_evaluate_predictions_uses_truth_and_probabilities() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, tz="UTC"),
            "true_state": [0, 0, 1, 1, 0, 0],
            "state": [1, 1, 0, 0, 1, 1],
            "prob_0": [0.1, 0.2, 0.9, 0.8, 0.3, 0.2],
            "prob_1": [0.9, 0.8, 0.1, 0.2, 0.7, 0.8],
        }
    )

    result = evaluate_predictions(
        frame,
        model_name="test-model",
        fit_seconds=1.25,
        metrics=["recovery_score", "probability_entropy", "fit_seconds"],
    )

    assert result["status"] == "completed"
    assert result["metrics"]["aligned_accuracy"] == 1.0
    assert "recovery_score" in result["metrics"]
    assert "probability_entropy" in result["metrics"]
    assert result["metrics"]["fit_seconds"] == 1.25


def test_rank_models_and_render_report(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, tz="UTC"),
            "close": [100.0, 101.0, 99.0, 102.0],
            "state": [0, 0, 1, 1],
            "prob_0": [0.8, 0.7, 0.2, 0.1],
            "prob_1": [0.2, 0.3, 0.8, 0.9],
        }
    )
    eval_path = tmp_path / "predictions.parquet"
    frame.to_parquet(eval_path, index=False)
    evaluation = {
        "models": [
            {
                "model": "good-model",
                "family": "probabilistic",
                "status": "completed",
                "metrics": {"recovery_score": 0.9, "fit_seconds": 1.0, "regime_persistence": 0.8},
                "evaluation_frame": str(eval_path),
            },
            {
                "model": "bad-model",
                "family": "rules",
                "status": "failed",
                "reason": "training failed",
            },
        ],
        "config": {"probability_kind": "filtered"},
        "data_period": "2024-01-01 to 2024-01-04",
    }

    comparison = rank_models(evaluation)
    payload = {**evaluation, **comparison}
    output = render_comparison_report(payload, title="Synthetic comparison", output=tmp_path / "report.html")

    assert comparison["ranking"]["recovery"][0]["model"] == "good-model"
    assert comparison["failures"][0]["model"] == "bad-model"
    html = output.read_text(encoding="utf-8")
    assert "Metric guide" in html
    assert "Recovery ranking" in html
    assert "Failures and unsupported models" in html

