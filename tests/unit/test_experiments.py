"""Tests for local experiment registries and runs."""

from __future__ import annotations

import json
import warnings

from regime.experiments import ExperimentStore, RunRegistry, config_hash, dataset_hash, model_hash


def test_hash_helpers_are_stable(tmp_path) -> None:
    """Config, dataset, and model hashes should be deterministic."""
    data_path = tmp_path / "data.csv"
    data_path.write_text("x,y\n1,2\n")

    assert config_hash({"b": 2, "a": 1}) == config_hash({"a": 1, "b": 2})
    assert dataset_hash(data_path) == dataset_hash(data_path)
    assert model_hash({"model": "baseline"}) == model_hash({"model": "baseline"})


def test_run_registry_records_artifacts_results_warnings_and_manifest(tmp_path) -> None:
    """A run should write local artifacts and SQLite registry rows."""
    store = ExperimentStore(tmp_path)
    registry = RunRegistry(store)

    def task(run):
        warnings.warn("captured", UserWarning, stacklevel=1)
        run.log_metrics({"accuracy": 1.0})
        run.log_json("predictions", "predictions.json", {"y_hat": [0, 1]})
        run.log_json("probabilities", "probabilities.json", {"p": [0.25, 0.75]})
        run.log_json("report", "report.json", {"summary": "ok"})
        run.log_json("plot", "plot.json", {"figure": "placeholder"})
        run.log_json("model", "model.json", {"coef": [1.0]})
        run.save_checkpoint({"step": 1})
        return run.load_checkpoint()

    assert registry.run("demo", task, group="examples", config={"alpha": 1}, resume=True) == {
        "step": 1
    }

    with store.connect() as con:
        run_row = con.execute("SELECT * FROM runs WHERE name='demo'").fetchone()
        artifact_kinds = {
            row["kind"]
            for row in con.execute(
                "SELECT kind FROM artifacts WHERE run_id=?", (run_row["run_id"],)
            )
        }
        metrics = con.execute("SELECT value_json FROM results WHERE kind='metrics'").fetchone()

    assert run_row["status"] == "completed"
    assert {
        "config",
        "metrics",
        "predictions",
        "probabilities",
        "model",
        "report",
        "plot",
        "log",
        "provenance",
        "manifest",
        "checkpoint",
    } <= artifact_kinds
    assert json.loads(metrics["value_json"]) == {"accuracy": 1.0}


def test_model_registry(tmp_path) -> None:
    """Model versions should be registered independently from runs."""
    store = ExperimentStore(tmp_path)
    model_id = store.register_model("hmm", "1", "abc123", metadata={"states": 2})

    with store.connect() as con:
        row = con.execute("SELECT * FROM models WHERE model_id=?", (model_id,)).fetchone()

    assert row["name"] == "hmm"
    assert row["model_hash"] == "abc123"
