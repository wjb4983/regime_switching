"""Serialization contracts for models and portable artifact metadata."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from regime.data.schemas import SchemaName
from regime.data.storage.manifest import DatasetManifest, DatasetPartition
from regime.models.probabilistic import GaussianHMM, ProbabilisticHMMConfig


@pytest.mark.timeout(10)
@pytest.mark.unit
def test_fitted_model_save_load_round_trip_preserves_inference(tmp_path: Path) -> None:
    """A restored model must produce the same probabilities as the fitted instance."""
    rng = np.random.default_rng(81)
    observations = np.r_[
        rng.normal(-2.0, 0.2, 30),
        rng.normal(2.0, 0.2, 30),
    ][:, None]
    model = GaussianHMM(
        ProbabilisticHMMConfig(
            model_name="round_trip_hmm",
            n_states=2,
            random_seed=19,
            n_init=2,
            max_iter=10,
        )
    ).fit(observations)

    destination = tmp_path / "nested" / "model.pkl"
    destination.parent.mkdir()
    model.save(destination)
    restored = GaussianHMM.load(destination)

    assert restored.metadata == model.metadata
    np.testing.assert_array_equal(restored.predict(observations), model.predict(observations))
    np.testing.assert_allclose(
        restored.predict_proba(observations), model.predict_proba(observations), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        restored.transition_matrix(), model.transition_matrix(), rtol=0, atol=0
    )


@pytest.mark.timeout(5)
@pytest.mark.unit
def test_manifest_path_serialization_is_platform_independent(tmp_path: Path) -> None:
    """Manifest paths use JSON strings with POSIX separators, independent of host paths."""
    portable_relative_path = PurePosixPath("parts", "2026", "observations.parquet").as_posix()
    manifest = DatasetManifest(
        partition=DatasetPartition(
            dataset=SchemaName.EQUITY_ETF_OHLCV,
            source="mock",
            asset_class="equity",
            date=date(2026, 8, 5),
            version="v1",
        ),
        schema_name=SchemaName.EQUITY_ETF_OHLCV,
        schema_fingerprint="schema-sha256",
        parquet_file=portable_relative_path,
        content_hash="content-sha256",
        row_count=3,
        columns=("timestamp", "close"),
        created_at=datetime(2026, 8, 5, tzinfo=UTC),
    )
    path = tmp_path / "manifest.json"

    manifest.write_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    restored = DatasetManifest.read_json(path)

    assert payload["parquet_file"] == "parts/2026/observations.parquet"
    assert "\\" not in payload["parquet_file"]
    assert restored == manifest
