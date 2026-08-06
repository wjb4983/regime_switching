"""Contract and reproducibility tests for artifact-driven comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from regime.evaluation.comparison_service import ComparisonConfiguration, ComparisonService
from regime.experiments.store import ExperimentStore


def _run(
    store: ExperimentStore,
    group_id: str,
    root: Path,
    name: str,
    losses: list[float],
    *,
    contract_cost: float = 0.0,
    include_metric: bool = True,
) -> str:
    run_id = store.create_run(group_id=group_id, name=name, hashes={"model_hash": name + "-hash"})
    directory = root / run_id
    directory.mkdir()
    predictions = directory / "predictions.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=len(losses)),
            "loss": losses,
            "fold": [0, 0, 1, 1][: len(losses)],
        }
    ).to_csv(predictions, index=False)
    metrics = directory / "metrics.json"
    metrics.write_text(
        json.dumps({"loss": sum(losses) / len(losses), "sharpe": 1.2} if include_metric else {}),
        encoding="utf-8",
    )
    contract = directory / "comparison_contract.json"
    contract.write_text(
        json.dumps({"information_set": "close", "costs": {"commission": contract_cost}}),
        encoding="utf-8",
    )
    store.add_artifact(run_id, "predictions", predictions)
    store.add_artifact(run_id, "metrics", metrics)
    store.add_artifact(run_id, "comparison", contract)
    store.update_run(run_id, "completed")
    return run_id


def test_alignment_bootstrap_ranking_and_parent_registration(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path)
    group_id = store.create_group("models")
    benchmark = _run(store, group_id, tmp_path, "benchmark", [2.0, 2.2, 1.9, 2.1])
    candidate = _run(store, group_id, tmp_path, "candidate", [1.0, 1.2, 0.9, 1.1])
    config = ComparisonConfiguration(
        primary_metric="loss", benchmark=benchmark, seed=42, n_bootstrap=50, block_length=2
    )

    first = ComparisonService(store).compare("models", config)
    second = ComparisonService(store).compare("models", config, persist=False)

    by_run = {row["run_id"]: row for row in first.table}
    assert by_run[candidate]["rank"] == 1
    assert by_run[benchmark]["rank"] == 2
    assert by_run[candidate]["uncertainty"] == next(
        row["uncertainty"] for row in second.table if row["run_id"] == candidate
    )
    assert first.statistical_tests[0]["n_obs"] == 4
    assert Path(first.artifact_path or "").exists()
    with store.connect() as connection:
        artifact = connection.execute(
            "SELECT a.artifact_id FROM artifacts a JOIN runs r ON r.run_id=a.run_id "
            "WHERE r.group_id=? AND a.kind='comparison' AND r.name LIKE 'comparison:%'",
            (group_id,),
        ).fetchone()
    assert artifact is not None


def test_incompatibility_is_diagnosed_and_never_ranked(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path)
    group_id = store.create_group("models")
    _run(store, group_id, tmp_path, "benchmark", [2.0, 2.1, 2.2, 2.3])
    incompatible = _run(
        store, group_id, tmp_path, "different-cost", [1.0, 1.1, 1.2, 1.3], contract_cost=0.01
    )

    result = ComparisonService(store).compare("models", persist=False)

    row = next(row for row in result.table if row["run_id"] == incompatible)
    diagnostic = next(item for item in result.diagnostics if item["run_id"] == incompatible)
    assert row["rank"] is None
    assert row["status"] == "incompatible"
    assert diagnostic["contract_differences"] == [
        {
            "field": "costs.commission",
            "reference": 0.0,
            "candidate": 0.01,
            "reference_missing": False,
            "candidate_missing": False,
        }
    ]


def test_missing_metric_remains_visible(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path)
    group_id = store.create_group("models")
    _run(store, group_id, tmp_path, "benchmark", [2.0, 2.1, 2.2, 2.3])
    missing = _run(store, group_id, tmp_path, "missing", [1.0, 1.1, 1.2, 1.3], include_metric=False)

    result = ComparisonService(store).compare("models", persist=False)

    row = next(row for row in result.table if row["run_id"] == missing)
    assert row["status"] == "missing_metric"
    assert row["rank"] is None
