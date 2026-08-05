"""Local-first Optuna study orchestration and persistence."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Literal

from regime.experiments import ExperimentStore

from .config import SearchSpace

Algorithm = Literal["bayesian", "random"]


def _optuna() -> Any:
    try:
        import optuna
    except ImportError as error:
        raise ImportError("Tuning requires `pip install regime-switching[optimization]`") from error
    return optuna


def _trial_constraints(trial: Any) -> tuple[float, ...]:
    """Read signed constraint violations written by ``MetricObjective``."""
    return tuple(trial.user_attrs.get("constraint_violations", ()))


@dataclass(frozen=True)
class StudyConfig:
    """Settings for a resumable local study."""

    name: str
    storage: str | Path = Path("experiments/optuna.sqlite3")
    algorithm: Algorithm = "bayesian"
    directions: tuple[str, ...] = ("maximize",)
    seed: int | None = None
    n_trials: int | None = 100
    timeout: float | None = None
    n_jobs: int = 1
    patience: int | None = None
    min_trials: int = 1


@dataclass
class EarlyStopping:
    """Stop after a configurable number of completed trials without improvement."""

    patience: int
    min_trials: int = 1
    _best: tuple[float, ...] | None = field(default=None, init=False)
    _stale: int = field(default=0, init=False)

    def __call__(self, study: Any, trial: Any) -> None:
        if trial.state.name != "COMPLETE":
            return
        values = tuple(trial.values)
        signs = tuple(1 if direction.name == "MAXIMIZE" else -1 for direction in study.directions)
        ranked = tuple(value * sign for value, sign in zip(values, signs, strict=True))
        if self._best is None or ranked > self._best:
            self._best, self._stale = ranked, 0
        else:
            self._stale += 1
        if len(study.trials) >= self.min_trials and self._stale >= self.patience:
            study.stop()


def create_study(config: StudyConfig) -> Any:
    """Create or resume a SQLite-backed Optuna study."""
    optuna = _optuna()
    path = Path(config.storage)
    path.parent.mkdir(parents=True, exist_ok=True)
    sampler = (
        optuna.samplers.RandomSampler(seed=config.seed)
        if config.algorithm == "random"
        else optuna.samplers.TPESampler(
            seed=config.seed, multivariate=True, constraints_func=_trial_constraints
        )
    )
    return optuna.create_study(
        study_name=config.name,
        storage=f"sqlite:///{path.resolve()}",
        load_if_exists=True,
        sampler=sampler,
        directions=list(config.directions),
        pruner=optuna.pruners.MedianPruner(),
    )


def optimize(
    config: StudyConfig,
    space: SearchSpace,
    objective: Callable[[Mapping[str, Any], Any], float | Sequence[float]],
    *,
    registry: ExperimentStore | None = None,
) -> Any:
    """Run trials in parallel, resume them safely, and persist a registry summary."""
    study = create_study(config)

    def wrapped(trial: Any) -> float | Sequence[float]:
        return objective(space.suggest(trial), trial)

    callbacks = (
        [EarlyStopping(config.patience, config.min_trials)] if config.patience is not None else []
    )
    run_id = None
    if registry is not None:
        run_id = registry.create_run(name=config.name, metadata={"optimizer": config.algorithm})
    try:
        study.optimize(
            wrapped,
            n_trials=config.n_trials,
            timeout=config.timeout,
            n_jobs=config.n_jobs,
            callbacks=callbacks,
            gc_after_trial=True,
        )
    except BaseException:
        if registry is not None and run_id is not None:
            registry.update_run(run_id, "failed", checkpoint_path=str(config.storage))
        raise
    if registry is not None and run_id is not None:
        summary = {
            "study_name": study.study_name,
            "storage": str(config.storage),
            "trials": len(study.trials),
            "best_trials": [trial.number for trial in study.best_trials],
        }
        registry.add_result(run_id, "tuning", summary)
        registry.update_run(run_id, "completed", checkpoint_path=str(config.storage))
    return study


def stability_analysis(
    run_seed: Callable[[int], float | Sequence[float]], seeds: Sequence[int]
) -> dict[str, Any]:
    """Summarize objective stability across independent sampler/model seeds."""
    if not seeds:
        raise ValueError("At least one seed is required")
    raw = [run_seed(seed) for seed in seeds]
    rows = [tuple(value) if isinstance(value, Sequence) else (value,) for value in raw]
    columns = list(zip(*rows, strict=True))
    return {
        "seeds": list(seeds),
        "values": [list(row) for row in rows],
        "mean": [mean(column) for column in columns],
        "std": [pstdev(column) for column in columns],
    }


def save_stability(result: Mapping[str, Any], path: str | Path) -> Path:
    """Persist a stability report as portable JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return destination


__all__ = [
    "Algorithm",
    "EarlyStopping",
    "StudyConfig",
    "create_study",
    "optimize",
    "save_stability",
    "stability_analysis",
]
