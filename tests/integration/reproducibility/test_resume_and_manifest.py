"""Integration contracts for resumable and auditable experiment runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from regime.experiments import ExperimentStore, RunRegistry
from regime.experiments.hashes import file_hash


@pytest.mark.timeout(10)
@pytest.mark.integration
def test_checkpoint_resume_continues_the_same_incomplete_run(tmp_path: Path) -> None:
    registry = RunRegistry(ExperimentStore(tmp_path))
    interrupted = registry.start(name="resumable", config={"epochs": 4})
    checkpoint = {"completed_epochs": 2, "losses": [1.0, 0.5]}
    checkpoint_path = interrupted.save_checkpoint(checkpoint)
    original_run_id = interrupted.run_id

    resumed = registry.start(name="resumable", config={"epochs": 4}, resume=True)

    assert resumed.run_id == original_run_id
    assert resumed.load_checkpoint() == checkpoint
    assert checkpoint_path.exists()
    resumed.save_checkpoint({"completed_epochs": 4, "losses": [1.0, 0.5, 0.3, 0.2]})
    assert resumed.load_checkpoint()["completed_epochs"] == 4
    with resumed.store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


@pytest.mark.timeout(10)
@pytest.mark.integration
def test_artifact_manifest_matches_registered_files_and_hashes(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path)
    registry = RunRegistry(store)

    def produce(run):
        run.log_json("predictions", "predictions.json", {"states": [0, 1, 1]})
        run.log_metrics({"accuracy": 0.75})

    registry.run("manifest-audit", produce, config={"seed": 42})
    row = store.latest_run("manifest-audit")
    assert row is not None
    manifest_path = tmp_path / row["run_id"] / "manifest" / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["run_id"] == row["run_id"]
    assert manifest["artifacts"]
    for artifact in manifest["artifacts"]:
        artifact_path = Path(artifact["path"])
        assert artifact_path.exists()
        assert artifact["hash"] == file_hash(artifact_path)
        assert isinstance(json.loads(artifact["metadata_json"]), dict)

    with store.connect() as connection:
        registered_before_manifest = connection.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id=? AND kind != 'manifest'", (row["run_id"],)
        ).fetchone()[0]
    assert len(manifest["artifacts"]) == registered_before_manifest
