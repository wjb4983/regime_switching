"""Tests for reusable, read-only artifact browsing logic."""

from pathlib import Path

import pandas as pd
import pytest

from regime.artifacts import ArtifactBrowser, matching_artifacts
from regime.experiments import ExperimentStore


def test_browse_compare_load_and_download(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path)
    first = store.create_run(name="hmm", hashes={"model_hash": "model-a"})
    second = store.create_run(name="markov", hashes={"model_hash": "model-b"})
    probability_path = tmp_path / "probabilities.csv"
    pd.DataFrame({"date": ["2024-01-01"], "state_0_probability": [0.8]}).to_csv(
        probability_path, index=False
    )
    store.add_artifact(first, "probabilities", probability_path, metadata={"view": "regimes"})
    store.add_result(first, "downstream", {"sharpe": 1.2})
    store.add_result(second, "downstream", {"sharpe": 0.9})

    browser = ArtifactBrowser(tmp_path)
    assert {run.name for run in browser.runs()} == {"hmm", "markov"}
    artifacts = browser.artifacts((first, second))
    assert matching_artifacts(artifacts, "probability") == artifacts
    assert browser.load_table(artifacts[0]).iloc[0]["state_0_probability"] == pytest.approx(0.8)
    assert set(browser.results((first, second))["sharpe"]) == {0.9, 1.2}
    assert browser.download(artifacts[0])[0] == probability_path.read_bytes()


def test_missing_registry_does_not_create_one(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Experiment registry not found"):
        ArtifactBrowser(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
