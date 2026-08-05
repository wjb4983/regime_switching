"""Run provenance and metadata capture utilities."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from regime.logging import JsonValue, redact


@dataclass(frozen=True)
class TimePeriod:
    """Named time period used for training or validation windows."""

    start: str | None = None
    end: str | None = None

    def to_record(self) -> dict[str, str | None]:
        """Return a JSON-compatible period record."""
        return {"start": self.start, "end": self.end}


@dataclass
class RunMetadata:
    """JSON-compatible metadata captured for a single experiment run."""

    git_commit: str | None
    working_tree_status: str
    python_version: str
    package_versions: dict[str, str]
    os: dict[str, str]
    hardware_summary: dict[str, JsonValue]
    random_seeds: dict[str, int | None]
    config_hash: str | None
    dataset_hash: str | None
    feature_hash: str | None
    model_hash: str | None
    training_period: TimePeriod | None
    validation_period: TimePeriod | None
    execution_assumptions: dict[str, JsonValue]
    cost_assumptions: dict[str, JsonValue]
    runtime_seconds: float
    generated_artifacts: list[str]
    started_at_unix: float
    completed_at_unix: float

    def to_record(self) -> dict[str, JsonValue]:
        """Return JSON-compatible metadata with secret-like values redacted."""
        return redact(
            {
                "git_commit": self.git_commit,
                "working_tree_status": self.working_tree_status,
                "python_version": self.python_version,
                "package_versions": self.package_versions,
                "os": self.os,
                "hardware_summary": self.hardware_summary,
                "random_seeds": self.random_seeds,
                "config_hash": self.config_hash,
                "dataset_hash": self.dataset_hash,
                "feature_hash": self.feature_hash,
                "model_hash": self.model_hash,
                "training_period": self.training_period.to_record()
                if self.training_period is not None
                else None,
                "validation_period": self.validation_period.to_record()
                if self.validation_period is not None
                else None,
                "execution_assumptions": self.execution_assumptions,
                "cost_assumptions": self.cost_assumptions,
                "runtime_seconds": self.runtime_seconds,
                "generated_artifacts": self.generated_artifacts,
                "started_at_unix": self.started_at_unix,
                "completed_at_unix": self.completed_at_unix,
            }
        )  # type: ignore[return-value]


def stable_hash(value: Any) -> str:
    """Return a deterministic SHA-256 hash for JSON-like values or bytes."""
    if isinstance(value, bytes):
        payload = value
    else:
        payload = json.dumps(
            redact(value), sort_keys=True, default=repr, separators=(",", ":")
        ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: str | Path) -> str:
    """Return a SHA-256 hash for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(args: Sequence[str], repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo, stderr=subprocess.DEVNULL, text=True, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def capture_git_commit(repo: str | Path = ".") -> str | None:
    """Capture the current Git commit if available."""
    return _run_git(["rev-parse", "HEAD"], Path(repo))


def capture_working_tree_status(repo: str | Path = ".") -> str:
    """Capture porcelain working-tree status, or a descriptive fallback."""
    return _run_git(["status", "--porcelain=v1"], Path(repo)) or "clean"


def capture_package_versions(package_names: Sequence[str] | None = None) -> dict[str, str]:
    """Capture installed package versions for selected packages or all distributions."""
    if package_names is None:
        return dict(
            sorted((dist.metadata["Name"], dist.version) for dist in metadata.distributions())
        )
    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
    return versions


def capture_hardware_summary() -> dict[str, JsonValue]:
    """Capture a lightweight hardware summary without optional dependencies."""
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def capture_random_seeds(extra: Mapping[str, int | None] | None = None) -> dict[str, int | None]:
    """Capture known random seed values supplied by the caller."""
    seeds = {"python_random_state_marker": random.getstate()[1][0]}
    seeds.update(extra or {})
    return seeds


@dataclass
class RunMetadataRecorder:
    """Helper that tracks runtime and artifacts before producing run metadata."""

    repo: Path = Path(".")
    package_names: Sequence[str] | None = None
    started_at_unix: float = field(default_factory=time.time)
    generated_artifacts: list[str] = field(default_factory=list)

    def add_artifact(self, artifact: str | Path) -> None:
        """Record a generated artifact path or URI."""
        self.generated_artifacts.append(str(artifact))

    def capture(
        self,
        *,
        config_hash: str | None = None,
        dataset_hash: str | None = None,
        feature_hash: str | None = None,
        model_hash: str | None = None,
        training_period: TimePeriod | None = None,
        validation_period: TimePeriod | None = None,
        execution_assumptions: Mapping[str, Any] | None = None,
        cost_assumptions: Mapping[str, Any] | None = None,
        random_seeds: Mapping[str, int | None] | None = None,
    ) -> RunMetadata:
        """Capture metadata required for reproducible experiment runs."""
        completed_at = time.time()
        return RunMetadata(
            git_commit=capture_git_commit(self.repo),
            working_tree_status=capture_working_tree_status(self.repo),
            python_version=sys.version,
            package_versions=capture_package_versions(self.package_names),
            os={
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
            },
            hardware_summary=capture_hardware_summary(),
            random_seeds=capture_random_seeds(random_seeds),
            config_hash=config_hash,
            dataset_hash=dataset_hash,
            feature_hash=feature_hash,
            model_hash=model_hash,
            training_period=training_period,
            validation_period=validation_period,
            execution_assumptions=redact(dict(execution_assumptions or {})),  # type: ignore[arg-type]
            cost_assumptions=redact(dict(cost_assumptions or {})),  # type: ignore[arg-type]
            runtime_seconds=completed_at - self.started_at_unix,
            generated_artifacts=list(self.generated_artifacts),
            started_at_unix=self.started_at_unix,
            completed_at_unix=completed_at,
        )


__all__ = [
    "RunMetadata",
    "RunMetadataRecorder",
    "TimePeriod",
    "capture_git_commit",
    "capture_hardware_summary",
    "capture_package_versions",
    "capture_random_seeds",
    "capture_working_tree_status",
    "file_hash",
    "stable_hash",
]
