"""Configuration-driven training service and stable artifact contract."""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from regime.experiments.hashes import file_hash
from regime.experiments.runner import ExperimentRun, capture_run_warnings
from regime.models.base import UnsupportedModelOperation
from regime.models.registry import create_model, model_configuration, model_spec


def _required(config: Mapping[str, Any], name: str) -> Any:
    value = config.get(name)
    if value is None or value == "":
        raise ValueError(f"Training configuration requires {name!r}")
    return value


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    if suffix in {".feather", ".arrow"}:
        return pd.read_feather(path)
    raise ValueError(f"Unsupported feature dataset format {suffix!r}")


def _validated_data(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, Any]]:
    features = config.get("features")
    if not isinstance(features, Sequence) or isinstance(features, str) or not features:
        raise ValueError("Training configuration requires a non-empty 'features' list")
    names = [str(item) for item in features]
    timestamp = str(config.get("timestamp_column", "timestamp"))
    missing = [name for name in [timestamp, *names] if name not in frame.columns]
    if missing:
        raise ValueError(f"Feature dataset is missing required columns: {', '.join(missing)}")
    work = frame.loc[:, [timestamp, *names]].copy()
    work[timestamp] = pd.to_datetime(work[timestamp], utc=True, errors="raise")
    if work[timestamp].duplicated().any() or not work[timestamp].is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and strictly increasing")
    input_rows = len(work)
    cutoff = config.get("fit_cutoff")
    if cutoff is not None:
        parsed = pd.to_datetime(cutoff, utc=True, errors="raise")
        work = work.loc[work[timestamp] <= parsed].copy()
    policy = str(config.get("missing_value_policy", config.get("missing_values", "error")))
    invalid = work[names].isna() | ~np.isfinite(work[names].astype(float))
    missing_rows = int(invalid.any(axis=1).sum())
    if missing_rows:
        if policy == "error":
            raise ValueError(f"Feature dataset contains {missing_rows} rows with missing values")
        if policy == "drop":
            work = work.loc[~invalid.any(axis=1)].copy()
        elif policy in {"forward_fill", "ffill"}:
            work[names] = work[names].replace([np.inf, -np.inf], np.nan).ffill()
            work = work.dropna(subset=names)
        else:
            raise ValueError("missing-value policy must be 'error', 'drop', or 'forward_fill'")
    minimum = int(config.get("minimum_observations", config.get("min_observations", 2)))
    if minimum < 1:
        raise ValueError("minimum_observations must be positive")
    if len(work) < minimum:
        raise ValueError(f"Training requires at least {minimum} observations; found {len(work)}")
    numeric = work.loc[:, names].apply(pd.to_numeric, errors="raise")
    diagnostics = {
        "input_observations": input_rows,
        "training_observations": len(work),
        "excluded_observations": input_rows - len(work),
        "missing_observations": missing_rows,
        "missing_value_policy": policy,
        "fit_cutoff": None if cutoff is None else str(cutoff),
        "timestamp_start": work[timestamp].iloc[0].isoformat(),
        "timestamp_end": work[timestamp].iloc[-1].isoformat(),
    }
    return work, numeric, timestamp, diagnostics


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _register(run: ExperimentRun, kind: str, path: Path, **metadata: Any) -> str:
    digest = file_hash(path)
    run.store.add_artifact(run.run_id, kind, path, artifact_hash=digest, metadata=metadata)
    run.metadata_recorder.add_artifact(path)
    if run.tracker:
        run.tracker.log_artifact(path, artifact_path=kind)
    return digest


def train_model(run: ExperimentRun, config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate data, fit one registered model, and register its complete artifact set."""
    started = time.perf_counter()
    model_name = str(_required(config, "model"))
    input_path = Path(str(_required(config, "input"))).expanduser().resolve()
    output = Path(str(_required(config, "output"))).expanduser().resolve()
    frame = _load_frame(input_path)
    training, values, timestamp, diagnostics = _validated_data(frame, config)

    # Construction intentionally occurs only after all inexpensive data validation.
    typed_config = model_configuration(model_name, config)
    spec = model_spec(model_name)
    model = create_model(model_name, config)
    output.mkdir(parents=True, exist_ok=True)
    resolved = {
        "model": spec.name,
        "input": str(input_path),
        "output": str(output),
        "timestamp_column": timestamp,
        "features": list(values.columns),
        "model_configuration": typed_config.model_dump(mode="json"),
    }
    resolved_path = output / "resolved_configuration.json"
    _write_json(resolved_path, resolved)

    with capture_run_warnings(run):
        model.fit(values, typed_config)
    model_path = output / ("model.json" if spec.name == "volatility-threshold" else "model.pkl")
    model.save(model_path)
    metadata = model.metadata.model_dump(mode="json")
    metadata_path = output / "metadata.json"
    _write_json(metadata_path, metadata)

    diagnostics["fit_seconds"] = time.perf_counter() - started
    diagnostics_path = output / "training_diagnostics.json"
    _write_json(diagnostics_path, diagnostics)
    try:
        statistics: Any = model.state_statistics()
        statistics_record = {"status": "supported", "statistics": statistics}
    except (UnsupportedModelOperation, NotImplementedError):
        statistics_record = {"status": "unsupported", "capability": "state_statistics"}
    statistics_path = output / "state_statistics.json"
    _write_json(statistics_path, statistics_record)

    try:
        states = list(model.predict(values))
        if len(states) != len(training):
            raise ValueError("Model returned a different number of predictions than observations")
        predictions = pd.DataFrame({timestamp: training[timestamp].to_numpy(), "state": states})
        predictions_path = output / "in_sample_predictions.parquet"
        predictions.to_parquet(predictions_path, index=False)
        prediction_status = {"status": "supported", "observations": len(predictions)}
        prediction_kind = "predictions"
    except UnsupportedModelOperation as error:
        predictions_path = output / "in_sample_predictions.unsupported.json"
        prediction_status = {"status": "unsupported", "capability": "predict", "reason": str(error)}
        _write_json(predictions_path, prediction_status)
        prediction_kind = "log"

    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "duration_seconds": time.perf_counter() - started,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    runtime_path = output / "runtime.json"
    _write_json(runtime_path, runtime)
    paths = [
        ("config", resolved_path),
        ("model", model_path),
        ("model", metadata_path),
        ("metrics", diagnostics_path),
        ("model", statistics_path),
        (prediction_kind, predictions_path),
        ("provenance", runtime_path),
    ]
    hashes = {path.name: _register(run, kind, path) for kind, path in paths}
    model_hash = hashes[model_path.name]
    run.store.register_model(
        metadata["model_name"],
        metadata["model_version"],
        model_hash,
        path=model_path,
        metadata=metadata,
    )
    metrics = {
        "training_observations": float(len(training)),
        "fit_seconds": float(diagnostics["fit_seconds"]),
    }
    run.log_metrics(metrics)
    run.store.add_result(run.run_id, "training_diagnostics", diagnostics)
    run.store.update_hashes(
        run.run_id,
        dataset_hash=file_hash(input_path),
        feature_hash=file_hash(input_path),
        model_hash=model_hash,
    )
    return {
        "model": spec.name,
        "output": str(output),
        "artifacts": {path.name: str(path) for _, path in paths},
        "hashes": hashes,
        "prediction_capability": prediction_status,
    }
