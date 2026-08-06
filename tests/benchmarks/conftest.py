"""Benchmark timing and local experiment-registry persistence."""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest

from regime.experiments.store import ExperimentStore


@pytest.fixture(scope="session")
def benchmark_recorder() -> Generator[Callable[[str, Callable[[], Any], int], Any], None, None]:
    """Time benchmark callables and persist one portable artifact for the session."""
    root = Path(os.environ.get("REGIME_BENCHMARK_REGISTRY", "experiments/benchmarks"))
    store = ExperimentStore(root)
    run_id = store.create_run(
        name="pytest-benchmarks",
        metadata={"python": platform.python_version(), "platform": platform.platform()},
    )
    results: list[dict[str, Any]] = []

    def record(name: str, operation: Callable[[], Any], rounds: int = 1) -> Any:
        samples: list[float] = []
        result: Any = None
        for _ in range(rounds):
            started = time.perf_counter()
            result = operation()
            samples.append(time.perf_counter() - started)
        entry = {
            "name": name,
            "rounds": rounds,
            "samples_seconds": samples,
            "min_seconds": min(samples),
            "mean_seconds": sum(samples) / len(samples),
        }
        results.append(entry)
        store.add_result(run_id, "benchmark", entry)
        return result

    try:
        yield record
    except BaseException:
        store.update_run(run_id, "failed")
        raise
    else:
        artifact_dir = root / "artifacts" / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact = artifact_dir / "benchmarks.json"
        artifact.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        store.add_artifact(
            run_id,
            "metrics",
            artifact,
            metadata={"format": "benchmark-v1", "benchmark_count": len(results)},
        )
        store.update_run(run_id, "completed")
