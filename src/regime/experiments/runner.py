"""Experiment run orchestration, checkpointing, manifests, and warning capture."""

from __future__ import annotations

import json
import pickle
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from regime.experiments.hashes import config_hash as compute_config_hash
from regime.experiments.hashes import file_hash, stable_hash
from regime.experiments.provenance import RunMetadataRecorder
from regime.experiments.store import ExperimentStore
from regime.logging import JsonValue, redact

T = TypeVar("T")
RunCallable = Callable[["ExperimentRun"], T]


class TrackingAdapter(Protocol):
    """Optional external tracking adapter interface."""

    def start_run(self, run_id: str, name: str | None = None) -> None: ...
    def log_params(self, params: Mapping[str, Any]) -> None: ...
    def log_metrics(self, metrics: Mapping[str, float]) -> None: ...
    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None: ...
    def end_run(self, status: str) -> None: ...


@dataclass
class MLflowAdapter:
    """Thin optional MLflow adapter imported only when instantiated."""

    experiment_name: str | None = None

    def __post_init__(self) -> None:
        import mlflow

        self._mlflow = mlflow
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_id: str, name: str | None = None) -> None:
        self._mlflow.start_run(run_name=name or run_id)
        self._mlflow.set_tag("regime_run_id", run_id)

    def log_params(self, params: Mapping[str, Any]) -> None:
        safe_params = cast(dict[str, JsonValue], redact(params))
        self._mlflow.log_params({key: str(value) for key, value in safe_params.items()})

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self._mlflow.log_metrics(metrics)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def end_run(self, status: str) -> None:
        self._mlflow.end_run(status="FINISHED" if status == "completed" else "FAILED")


@dataclass
class ExperimentRun:
    """Mutable handle for a local experiment run."""

    store: ExperimentStore
    run_id: str
    root: Path
    tracker: TrackingAdapter | None = None
    metadata_recorder: RunMetadataRecorder = field(default_factory=RunMetadataRecorder)
    warnings: list[dict[str, JsonValue]] = field(default_factory=list)

    def artifact_path(self, kind: str, filename: str) -> Path:
        """Return and create a namespaced artifact path for this run."""
        path = self.root / self.run_id / kind / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def log_json(self, kind: str, filename: str, value: Mapping[str, Any]) -> Path:
        """Write a JSON artifact and register it."""
        path = self.artifact_path(kind, filename)
        path.write_text(json.dumps(redact(value), indent=2, sort_keys=True, default=repr))
        digest = file_hash(path)
        self.store.add_artifact(self.run_id, kind, path, artifact_hash=digest)
        self.metadata_recorder.add_artifact(path)
        if self.tracker:
            self.tracker.log_artifact(path, artifact_path=kind)
        return path

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        """Record metric results locally and through the optional tracker."""
        self.store.add_result(self.run_id, "metrics", dict(metrics))
        self.log_json("metrics", "metrics.json", dict(metrics))
        if self.tracker:
            self.tracker.log_metrics(metrics)

    def save_checkpoint(self, state: Any, name: str = "checkpoint.pkl") -> Path:
        """Persist checkpoint state for resumable runs."""
        path = self.artifact_path("checkpoint", name)
        with path.open("wb") as file_obj:
            pickle.dump(state, file_obj)
        self.store.add_artifact(self.run_id, "checkpoint", path, artifact_hash=file_hash(path))
        self.store.update_run(self.run_id, "running", checkpoint_path=str(path))
        return path

    def load_checkpoint(self) -> Any | None:
        """Load the latest checkpoint for this run if one exists."""
        with self.store.connect() as con:
            row = con.execute(
                "SELECT checkpoint_path FROM runs WHERE run_id=?", (self.run_id,)
            ).fetchone()
        if row is None or row["checkpoint_path"] is None:
            return None
        with Path(row["checkpoint_path"]).open("rb") as file_obj:
            return pickle.load(file_obj)

    def write_manifest(self) -> Path:
        """Write an artifact manifest containing paths, kinds, hashes, and metadata."""
        with self.store.connect() as con:
            rows = con.execute(
                "SELECT kind, path, hash, metadata_json FROM artifacts WHERE run_id=?",
                (self.run_id,),
            ).fetchall()
        manifest = {"run_id": self.run_id, "artifacts": [dict(row) for row in rows]}
        return self.log_json("manifest", "artifact-manifest.json", manifest)


@contextmanager
def capture_run_warnings(run: ExperimentRun) -> Iterator[None]:
    """Capture warnings emitted inside a run as log artifacts."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        yield
    for item in records:
        run.warnings.append(
            {
                "message": str(item.message),
                "category": item.category.__name__,
                "filename": item.filename,
                "lineno": item.lineno,
            }
        )
    if run.warnings:
        run.log_json("log", "warnings.json", {"warnings": run.warnings})


@dataclass
class RunRegistry:
    """High-level run registry with resumable execution."""

    store: ExperimentStore = field(default_factory=ExperimentStore)
    tracker: TrackingAdapter | None = None

    def start(
        self,
        *,
        name: str,
        group: str | None = None,
        config: Mapping[str, Any] | None = None,
        resume: bool = False,
        hashes: Mapping[str, str | None] | None = None,
    ) -> ExperimentRun:
        """Start a new run or resume the latest non-completed run with the same name."""
        if resume:
            previous = self.store.latest_run(name)
            if previous is not None and previous["status"] != "completed":
                return ExperimentRun(
                    self.store, str(previous["run_id"]), Path(self.store.root), self.tracker
                )
        group_id = self.store.create_group(group) if group else None
        run_hashes = dict(hashes or {})
        if config is not None:
            run_hashes.setdefault("config_hash", compute_config_hash(config))
        run_id = self.store.create_run(
            group_id=group_id,
            name=name,
            hashes=run_hashes,
            metadata={"config": config or {}},
        )
        if self.tracker:
            self.tracker.start_run(run_id, name)
            if config:
                self.tracker.log_params(config)
        run = ExperimentRun(self.store, run_id, Path(self.store.root), self.tracker)
        if config is not None:
            run.log_json("config", "config.json", config)
        return run

    def run(
        self,
        name: str,
        function: RunCallable[T],
        *,
        group: str | None = None,
        config: Mapping[str, Any] | None = None,
        resume: bool = False,
    ) -> T:
        """Execute a function with warning capture, provenance, manifest, and status updates."""
        run = self.start(name=name, group=group, config=config, resume=resume)
        try:
            with capture_run_warnings(run):
                result = function(run)
            metadata = run.metadata_recorder.capture(
                config_hash=stable_hash(config or {})
            ).to_record()
            run.log_json("provenance", "provenance.json", metadata)
            run.write_manifest()
            self.store.update_run(run.run_id, "completed")
            if self.tracker:
                self.tracker.end_run("completed")
            return result
        except Exception:
            self.store.update_run(run.run_id, "failed")
            if self.tracker:
                self.tracker.end_run("failed")
            raise


__all__ = [
    "ExperimentRun",
    "MLflowAdapter",
    "RunRegistry",
    "TrackingAdapter",
    "capture_run_warnings",
]
