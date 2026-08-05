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
        if not isinstance(document, Mapping) or not isinstance(document.get("parameters"), Mapping):
            raise ValueError("Search-space YAML must contain a 'parameters' mapping")
        parameters = {
            str(name): Parameter.from_dict(spec) for name, spec in document["parameters"].items()
        }
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


__all__ = ["Parameter", "ParameterKind", "SearchSpace"]
