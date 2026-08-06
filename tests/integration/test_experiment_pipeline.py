"""Tiny end-to-end experiment graph contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from regime.experiments.pipeline import ExperimentPipeline

pytestmark = [pytest.mark.integration, pytest.mark.synthetic, pytest.mark.timeout(20)]


def test_synthetic_pipeline_links_and_resumes_every_stage(tmp_path: Path) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "tiny-linked-experiment",
                "work_dir": str(tmp_path / "work"),
                "source": {"generator": "gaussian_hmm", "observations": 40, "seed": 7},
                "features": {"features": ["return_1d", "realized_volatility_21d"]},
                "baselines": [
                    {
                        "model": "volatility_threshold",
                        "features": ["realized_volatility_21d"],
                        "fit_parameters": {
                            "feature": "realized_volatility_21d",
                            "threshold": 0.2,
                        },
                    }
                ],
                "validation": {"strategy": "holdout", "train_size": 0.7},
                "evaluation": {"metrics": ["regime_persistence"]},
                "report": {"title": "Tiny report", "output_dir": str(tmp_path / "reports")},
            }
        ),
        encoding="utf-8",
    )
    pipeline = ExperimentPipeline.from_root(tmp_path / "registry")

    first = pipeline.run(config)
    second = pipeline.run(config, resume=True)

    assert second["run_id"] == first["run_id"]
    assert second["stages"] == first["stages"]
    assert Path(first["stages"]["evaluation"]["path"]).is_file()
    assert Path(first["stages"]["report"]["path"]).is_file()

    def contains_none(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, dict):
            return any(contains_none(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_none(item) for item in value)
        return False

    for stage in ("evaluation", "comparison"):
        payload = json.loads(Path(first["stages"][stage]["path"]).read_text(encoding="utf-8"))
        assert not contains_none(payload), f"{stage} json still contains null values"
    with pipeline.registry.store.connect() as connection:
        children = connection.execute(
            "SELECT * FROM runs WHERE group_id=? AND run_id<>?",
            (first["group_id"], first["run_id"]),
        ).fetchall()
        linked = connection.execute(
            "SELECT metadata_json FROM artifacts WHERE run_id<>?", (first["run_id"],)
        ).fetchall()
    assert len(children) == len(first["stages"])
    assert all(child["status"] == "completed" for child in children)
    assert any(json.loads(row["metadata_json"]).get("upstream_run_ids") for row in linked)
