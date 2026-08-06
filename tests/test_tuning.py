"""Tests for optimizer-independent tuning primitives."""

from pathlib import Path

import pytest

from regime.tuning import MetricObjective, SearchSpace, TuningConfig, nested_validation_objective
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


def test_expanded_schema_and_registry_tunable_validation(tmp_path: Path) -> None:
    path = tmp_path / "tune.yaml"
    path.write_text(
        "name: deterministic\nbase_model: {model: gaussian-hmm}\n"
        "validation: {dataset: sample.csv}\n"
        "objectives: [{metric: score, direction: maximize}]\n"
        "trial_count: 3\nstudy_timeout: 2\nparallelism: 1\n"
        "seed_policy: {sampler: 7, model: 9, stability: [9, 10]}\n"
        "search_space:\n  n_states: {type: int, low: 2, high: 3}\n",
        encoding="utf-8",
    )
    config = TuningConfig.from_yaml(path)
    assert config.trials == 3
    assert config.seed_policy.stability == (9, 10)
    assert config.objectives[0].direction == "maximize"


def test_irrelevant_or_misspelled_parameters_fail_before_study(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "base_model: {model: gaussian-hmm}\nvalidation: {}\n"
        "objectives: [{metric: score, direction: maximize}]\n"
        "search_space:\n  student_t_dof: {type: float, low: 3, high: 4}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not tunable"):
        TuningConfig.from_yaml(path)


@pytest.mark.optuna
@pytest.mark.timeout(10)
def test_small_optuna_multi_objective_study(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    from regime.tuning.runner import StudyConfig, optimize

    space_path = tmp_path / "space.yaml"
    space_path.write_text(
        "parameters:\n  x: {type: float, low: 0.0, high: 1.0}\n", encoding="utf-8"
    )
    study = optimize(
        StudyConfig(
            "small",
            storage=tmp_path / "study.sqlite3",
            algorithm="random",
            directions=("maximize", "minimize"),
            seed=3,
            n_trials=3,
            timeout=5,
        ),
        SearchSpace.from_yaml(space_path),
        lambda params, trial: (params["x"], params["x"]),
    )
    assert len(study.trials) == 3
    assert study.best_trials
