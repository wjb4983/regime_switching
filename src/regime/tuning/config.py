"""Validated, YAML-backed tuning search spaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]

ParameterKind = Literal["float", "int", "categorical"]


@dataclass(frozen=True)
class Parameter:
    """One parameter distribution, optionally enabled by parent values."""

    kind: ParameterKind
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[Any, ...] = ()
    step: float | int | None = None
    log: bool = False
    when: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Parameter:
        """Build and validate a parameter from decoded YAML."""
        parameter = cls(
            kind=value["type"],
            low=value.get("low"),
            high=value.get("high"),
            choices=tuple(value.get("choices", ())),
            step=value.get("step"),
            log=bool(value.get("log", False)),
            when=dict(value.get("when", {})),
        )
        if parameter.kind not in {"float", "int", "categorical"}:
            raise ValueError(f"Unsupported parameter type: {parameter.kind}")
        if parameter.kind == "categorical" and not parameter.choices:
            raise ValueError("Categorical parameters require non-empty choices")
        if parameter.kind != "categorical" and (parameter.low is None or parameter.high is None):
            raise ValueError(f"{parameter.kind} parameters require low and high")
        if parameter.log and parameter.step is not None:
            raise ValueError("Optuna does not support step together with log")
        return parameter

    def enabled(self, selected: Mapping[str, Any]) -> bool:
        """Return whether all parent conditions match selected values."""
        return all(selected.get(name) == expected for name, expected in self.when.items())

    def suggest(self, trial: Any, name: str) -> Any:
        """Ask an Optuna trial for a value from this distribution."""
        if self.kind == "categorical":
            return trial.suggest_categorical(name, list(self.choices))
        if self.kind == "int":
            assert self.low is not None
            assert self.high is not None
            return trial.suggest_int(
                name, int(self.low), int(self.high), step=int(self.step or 1), log=self.log
            )
        assert self.low is not None
        assert self.high is not None
        return trial.suggest_float(
            name,
            float(self.low),
            float(self.high),
            step=float(self.step) if self.step is not None else None,
            log=self.log,
        )


@dataclass(frozen=True)
class SearchSpace:
    """Ordered parameter space; parents must appear before conditional children."""

    parameters: Mapping[str, Parameter]

    @classmethod
    def from_yaml(cls, path: str | Path) -> SearchSpace:
        """Load a search space from a local YAML file."""
        with Path(path).open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        if not isinstance(document, Mapping):
            raise ValueError("Search-space YAML must be a mapping")
        raw_parameters = document.get("parameters", document.get("search_space"))
        if not isinstance(raw_parameters, Mapping):
            raise ValueError("Search-space YAML must contain 'parameters' or 'search_space'")
        parameters = {str(name): Parameter.from_dict(spec) for name, spec in raw_parameters.items()}
        seen: set[str] = set()
        for name, parameter in parameters.items():
            missing = set(parameter.when) - seen
            if missing:
                raise ValueError(f"Conditional parents must precede {name}: {sorted(missing)}")
            seen.add(name)
        return cls(parameters)

    def suggest(self, trial: Any) -> dict[str, Any]:
        """Materialize active parameters for a trial."""
        selected: dict[str, Any] = {}
        for name, parameter in self.parameters.items():
            if parameter.enabled(selected):
                selected[name] = parameter.suggest(trial, name)
        return selected

    def validate_tunable(self, model: str) -> None:
        """Reject parameters which the selected registry descriptor does not expose."""
        from regime.models.registry import model_spec

        allowed = model_spec(model).tunable_parameters
        unknown = set(self.parameters) - allowed
        if unknown:
            raise ValueError(
                f"Parameters are not tunable for {model!r}: {', '.join(sorted(unknown))}. "
                f"Declared tunable parameters: {', '.join(sorted(allowed)) or '(none)'}"
            )


@dataclass(frozen=True)
class ObjectiveSpec:
    """A named evaluation metric and its unambiguous optimization direction."""

    metric: str
    direction: Literal["maximize", "minimize"]


@dataclass(frozen=True)
class SeedPolicy:
    """Reproducibility and stability seeds used by a study."""

    sampler: int | None = None
    model: int | None = None
    stability: tuple[int, ...] = ()


@dataclass(frozen=True)
class TuningConfig:
    """Complete, validated tuning workflow document."""

    name: str
    base_model: Mapping[str, Any]
    search_space: SearchSpace
    validation: Mapping[str, Any]
    objectives: tuple[ObjectiveSpec, ...]
    constraints: Mapping[str, float]
    trials: int
    timeout: float | None
    parallelism: int
    seed_policy: SeedPolicy
    storage: Path
    algorithm: Literal["bayesian", "random"] = "bayesian"

    @classmethod
    def from_yaml(cls, path: str | Path) -> TuningConfig:
        """Decode the expanded tuning schema and validate it before optimization."""
        source = Path(path)
        with source.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, Mapping):
            raise ValueError("Tuning YAML must be a mapping")
        base = raw.get("base_model")
        validation = raw.get("validation")
        objective_rows = raw.get("objectives")
        if not isinstance(base, Mapping) or not base.get("model"):
            raise ValueError("Tuning YAML requires base_model.model")
        if not isinstance(validation, Mapping):
            raise ValueError("Tuning YAML requires a validation mapping")
        if not isinstance(objective_rows, list) or not objective_rows:
            raise ValueError("Tuning YAML requires one or more objectives")
        objectives = tuple(
            ObjectiveSpec(str(item["metric"]), item["direction"]) for item in objective_rows
        )
        if any(item.direction not in {"maximize", "minimize"} for item in objectives):
            raise ValueError("Objective directions must be 'maximize' or 'minimize'")
        parameters = raw.get("search_space")
        if not isinstance(parameters, Mapping):
            raise ValueError("Tuning YAML requires a search_space mapping")
        space = SearchSpace(
            {str(name): Parameter.from_dict(spec) for name, spec in parameters.items()}
        )
        # Reuse ordering validation, including conditional parent spelling.
        seen: set[str] = set()
        for name, parameter in space.parameters.items():
            missing = set(parameter.when) - seen
            if missing:
                raise ValueError(f"Conditional parents must precede {name}: {sorted(missing)}")
            seen.add(name)
        space.validate_tunable(str(base["model"]))
        seeds = raw.get("seed_policy", {})
        if not isinstance(seeds, Mapping):
            raise ValueError("seed_policy must be a mapping")
        trials = int(raw.get("trial_count", 100))
        parallelism = int(raw.get("parallelism", 1))
        if trials < 1 or parallelism < 1:
            raise ValueError("trial_count and parallelism must be positive")
        constraints = raw.get("constraints", {})
        if not isinstance(constraints, Mapping):
            raise ValueError("constraints must map metric names to upper limits")
        return cls(
            name=str(raw.get("name", source.stem)),
            base_model=dict(base),
            search_space=space,
            validation=dict(validation),
            objectives=objectives,
            constraints={str(k): float(v) for k, v in constraints.items()},
            trials=trials,
            timeout=None if raw.get("study_timeout") is None else float(raw["study_timeout"]),
            parallelism=parallelism,
            seed_policy=SeedPolicy(
                sampler=seeds.get("sampler"),
                model=seeds.get("model"),
                stability=tuple(int(value) for value in seeds.get("stability", ())),
            ),
            storage=Path(str(raw.get("storage", "experiments/optuna.sqlite3"))),
            algorithm=raw.get("algorithm", "bayesian"),
        )


__all__ = [
    "ObjectiveSpec",
    "Parameter",
    "ParameterKind",
    "SearchSpace",
    "SeedPolicy",
    "TuningConfig",
]
