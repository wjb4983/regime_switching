"""Typed metadata and compatibility checks for evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class ProbabilityKind(StrEnum):
    """Probability representation accepted by a metric."""

    NONE = "none"
    FILTERED = "filtered"
    SMOOTHED = "smoothed"
    PREDICTIVE = "predictive"
    ANY = "any"


@dataclass(frozen=True)
class MetricDescriptor:
    """Complete, machine-readable contract for one metric."""

    name: str
    required_inputs: frozenset[str]
    direction: Literal["minimize", "maximize", "diagnostic"]
    probability_kind: ProbabilityKind
    aggregation: Literal["mean", "sum", "pooled", "none"]
    display_format: str

    def compatible(self, available_inputs: set[str], probability_kind: str = "none") -> bool:
        """Return whether inputs and probability semantics satisfy this metric."""
        probability_ok = self.probability_kind in {ProbabilityKind.ANY, ProbabilityKind.NONE}
        if self.probability_kind is not ProbabilityKind.NONE:
            probability_ok = self.probability_kind.value == probability_kind
        return self.required_inputs <= available_inputs and probability_ok


def descriptor(
    name: str,
    inputs: tuple[str, ...],
    direction: Literal["minimize", "maximize", "diagnostic"],
    probability: ProbabilityKind = ProbabilityKind.NONE,
    aggregation: Literal["mean", "sum", "pooled", "none"] = "mean",
    display: str = ".4f",
) -> MetricDescriptor:
    """Build a normalized descriptor without repeating conversion boilerplate."""
    return MetricDescriptor(name, frozenset(inputs), direction, probability, aggregation, display)


__all__ = ["MetricDescriptor", "ProbabilityKind", "descriptor"]
