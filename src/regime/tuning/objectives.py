"""Composable statistical, economic, constrained, and nested objectives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Metric = Callable[[Any], float]
Evaluator = Callable[[Mapping[str, Any], Any], Any]


@dataclass(frozen=True)
class MetricObjective:
    """Convert model output into one or several optimization metrics."""

    evaluate: Evaluator
    metrics: Mapping[str, Metric]
    constraints: Mapping[str, tuple[Metric, float]] | None = None

    def __call__(self, params: Mapping[str, Any], context: Any = None) -> float | tuple[float, ...]:
        output = self.evaluate(params, context)
        violations = {
            name: metric(output) - limit
            for name, (metric, limit) in (self.constraints or {}).items()
        }
        if violations:
            # Optuna's constrained samplers consume non-positive feasible values.
            context_trial = getattr(context, "trial", context)
            if hasattr(context_trial, "set_user_attr"):
                context_trial.set_user_attr("constraint_violations", tuple(violations.values()))
        values = tuple(float(metric(output)) for metric in self.metrics.values())
        return values[0] if len(values) == 1 else values


def nested_validation_objective(
    evaluate_fold: Callable[[Mapping[str, Any], Any], float],
    inner_splits: Sequence[Any],
    *,
    aggregate: Callable[[Sequence[float]], float] = lambda values: sum(values) / len(values),
) -> Callable[[Mapping[str, Any], Any], float]:
    """Aggregate inner-fold scores while reporting them for trial pruning."""

    def objective(params: Mapping[str, Any], trial: Any = None) -> float:
        scores: list[float] = []
        for step, split in enumerate(inner_splits):
            scores.append(float(evaluate_fold(params, split)))
            if trial is not None and hasattr(trial, "report"):
                trial.report(aggregate(scores), step)
                if trial.should_prune():
                    import optuna

                    raise optuna.TrialPruned
        return float(aggregate(scores))

    return objective


__all__ = ["Evaluator", "Metric", "MetricObjective", "nested_validation_objective"]
