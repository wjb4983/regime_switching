"""Tests for optimizer-independent tuning primitives."""

from pathlib import Path

import pytest

from regime.tuning import MetricObjective, SearchSpace, nested_validation_objective
from regime.tuning.runner import save_stability, stability_analysis


class FakeTrial:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.reports: list[tuple[float, int]] = []

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        return choices[0]

    def suggest_int(self, name: str, low: int, high: int, **kwargs: object) -> int:
        return low

    def suggest_float(self, name: str, low: float, high: float, **kwargs: object) -> float:
        return low

    def set_user_attr(self, name: str, value: object) -> None:
        self.attributes[name] = value

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        return False


def test_yaml_space_supports_conditional_parameters(tmp_path: Path) -> None:
    path = tmp_path / "space.yaml"
    path.write_text(
        "parameters:\n  model:\n    type: categorical\n    choices: [markov, hmm]\n"
        "  covariance:\n    type: categorical\n    choices: [full]\n    when:\n      model: hmm\n",
        encoding="utf-8",
    )
    assert SearchSpace.from_yaml(path).suggest(FakeTrial()) == {"model": "markov"}


def test_metric_and_nested_objectives() -> None:
    trial = FakeTrial()
    objective = MetricObjective(
        lambda params, context: params["return"],
        {"economic_return": float, "statistical_score": lambda value: value / 2},
        {"drawdown": (lambda value: value, 0.1)},
    )
    assert objective({"return": 0.2}, trial) == (0.2, 0.1)
    assert trial.attributes["constraint_violations"] == pytest.approx((0.1,))

    nested = nested_validation_objective(lambda params, split: params["x"] + split, [1, 3])
    assert nested({"x": 2}, trial) == 4
    assert trial.reports == [(3.0, 0), (4.0, 1)]


def test_stability_analysis_and_persistence(tmp_path: Path) -> None:
    result = stability_analysis(lambda seed: (float(seed), float(seed * 2)), [1, 2, 3])
    assert result["mean"] == [2.0, 4.0]
    assert save_stability(result, tmp_path / "reports" / "stability.json").is_file()
