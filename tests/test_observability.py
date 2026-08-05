"""Tests for logging, errors, and run provenance helpers."""

from __future__ import annotations

import io
import json
import logging
import warnings

from regime.errors import RegimeConfigurationError
from regime.experiments.provenance import RunMetadataRecorder, TimePeriod, stable_hash
from regime.logging import REDACTION_TEXT, configure_logging, log_event, redact


def test_redact_masks_secret_fields_and_values() -> None:
    """Secret-looking field names and token-like values should be masked."""
    payload = {
        "api_key": "sk-testsecret123456789",
        "nested": {"Authorization": "Bearer abcdefghijklmnop"},
        "safe": "visible",
    }

    assert redact(payload) == {
        "api_key": REDACTION_TEXT,
        "nested": {"Authorization": REDACTION_TEXT},
        "safe": "visible",
    }


def test_structured_logging_outputs_json_and_captures_warnings() -> None:
    """Structured logs and Python warnings should share JSON formatting."""
    stream = io.StringIO()
    logger = configure_logging(stream=stream, logger_name="regime.test")

    log_event(logger, logging.INFO, "training_started", api_key="sk-testsecret123456789", rows=10)
    warnings.warn("deprecated path", UserWarning, stacklevel=1)

    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert lines[0]["message"] == "training_started"
    assert lines[0]["extra"]["api_key"] == REDACTION_TEXT
    assert lines[0]["extra"]["rows"] == 10
    assert lines[1]["logger"] == "py.warnings"
    assert "deprecated path" in lines[1]["message"]


def test_regime_error_to_record_includes_typed_context() -> None:
    """Package errors should expose structured typed context."""
    error = RegimeConfigurationError("bad config", code="CONFIG_BAD", context={"field": "x"})

    assert error.to_record() == {
        "type": "RegimeConfigurationError",
        "code": "CONFIG_BAD",
        "message": "bad config",
        "context": {"field": "x"},
    }


def test_run_metadata_recorder_captures_required_fields() -> None:
    """Run metadata should include reproducibility fields and redact assumptions."""
    recorder = RunMetadataRecorder(package_names=["pytest"])
    recorder.add_artifact("reports/run.json")

    metadata = recorder.capture(
        config_hash=stable_hash({"alpha": 1}),
        dataset_hash=stable_hash(b"dataset"),
        feature_hash=stable_hash({"features": ["x"]}),
        model_hash=stable_hash({"model": "baseline"}),
        training_period=TimePeriod("2020-01-01", "2020-12-31"),
        validation_period=TimePeriod("2021-01-01", "2021-03-31"),
        execution_assumptions={"api_token": "secret", "retries": 2},
        cost_assumptions={"usd_per_hour": 1.0},
        random_seeds={"numpy": 7},
    )
    record = metadata.to_record()

    assert record["git_commit"]
    assert "python_version" in record
    assert record["package_versions"]["pytest"] != "not-installed"
    assert record["hardware_summary"]["cpu_count"] is not None
    assert record["random_seeds"]["numpy"] == 7
    assert record["training_period"] == {"start": "2020-01-01", "end": "2020-12-31"}
    assert record["validation_period"] == {"start": "2021-01-01", "end": "2021-03-31"}
    assert record["execution_assumptions"]["api_token"] == REDACTION_TEXT
    assert record["runtime_seconds"] >= 0
    assert record["generated_artifacts"] == ["reports/run.json"]
