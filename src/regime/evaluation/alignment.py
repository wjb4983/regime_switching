"""State alignment utilities for independently fitted regime models.

Regime labels are nominal identifiers: state ``0`` from one fit is not comparable to
state ``0`` from another fit unless their fitted state summaries have first been
aligned.  This module builds a cost matrix from configurable state descriptors and
uses Hungarian matching to map candidate-state labels onto reference-state labels.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wasserstein_distance

DistanceName = Literal[
    "state_means",
    "covariances",
    "return_risk",
    "durations",
    "transitions",
    "distributions",
]

_DEFAULT_DISTANCES: tuple[DistanceName, ...] = (
    "state_means",
    "covariances",
    "return_risk",
    "durations",
    "transitions",
    "distributions",
)

_KEY_ALIASES: dict[DistanceName, tuple[str, ...]] = {
    "state_means": ("state_means", "means", "centroids"),
    "covariances": ("covariances", "covariance_matrices", "covariance"),
    "return_risk": ("return_risk", "return_risk_summaries", "risk", "summaries"),
    "durations": ("durations", "duration_distributions", "duration"),
    "transitions": ("transitions", "transition_probabilities", "transition_matrix"),
    "distributions": ("distributions", "distributional", "samples"),
}


@dataclass(frozen=True)
class AlignmentMethod:
    """Configuration for Hungarian state matching.

    Distances are computed only when the corresponding descriptor is present for
    both models and both compared states.  Weights allow mixed units to be
    emphasized or disabled without encouraging raw-label comparisons.
    """

    distances: tuple[DistanceName, ...] = _DEFAULT_DISTANCES
    weights: Mapping[DistanceName, float] = field(default_factory=dict)
    ambiguity_ratio: float = 0.05
    ambiguity_epsilon: float = 1e-12


@dataclass(frozen=True)
class AlignmentDiagnostics:
    """Diagnostics describing the fitted candidate-to-reference assignment."""

    total_cost: float
    matched_costs: Mapping[int, float]
    component_costs: Mapping[tuple[int, int], Mapping[str, float]]
    candidate_to_reference: Mapping[int, int]
    reference_labels: tuple[int, ...]
    candidate_labels: tuple[int, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AlignmentResult:
    """Hungarian alignment result for nominal regime-state labels."""

    alignment_matrix: tuple[tuple[float, ...], ...]
    candidate_to_reference: Mapping[int, int]
    reference_to_candidate: Mapping[int, int]
    diagnostics: AlignmentDiagnostics

    def aligned_labels(self, labels: Sequence[int]) -> tuple[int, ...]:
        """Map candidate labels into the aligned reference-label space."""
        return aligned_labels(labels, self.candidate_to_reference)


def _mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "model_dump"):
        dumped = obj.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return {name: getattr(obj, name) for name in dir(obj) if not name.startswith("_")}


def _first_present(payload: Mapping[str, Any], name: DistanceName) -> Any | None:
    for key in _KEY_ALIASES[name]:
        if key in payload:
            return payload[key]
    return None


def _state_value(block: Any, label: int, *, transition_row: bool = False) -> Any | None:
    if block is None:
        return None
    if isinstance(block, Mapping):
        for key in (label, str(label)):
            if key in block:
                return block[key]
        if transition_row and "matrix" in block and "labels" in block:
            labels = [int(x) for x in block["labels"]]
            if label in labels:
                return np.asarray(block["matrix"], dtype=float)[labels.index(label)]
        return None
    arr = np.asarray(block, dtype=float)
    if transition_row and arr.ndim == 2 and 0 <= label < arr.shape[0]:
        return arr[label]
    if arr.ndim >= 1 and 0 <= label < arr.shape[0]:
        return arr[label]
    return None


def _labels(payload: Mapping[str, Any]) -> tuple[int, ...]:
    if "labels" in payload:
        return tuple(int(x) for x in payload["labels"])
    found: set[int] = set()
    for name in _DEFAULT_DISTANCES:
        block = _first_present(payload, name)
        if isinstance(block, Mapping):
            if "labels" in block and "matrix" in block:
                found.update(int(x) for x in block["labels"])
            else:
                for key in block:
                    try:
                        found.add(int(key))
                    except (TypeError, ValueError):
                        pass
        elif block is not None:
            arr = np.asarray(block)
            if arr.ndim >= 1:
                found.update(range(arr.shape[0]))
    if not found:
        raise ValueError("state summaries must expose labels or per-state descriptors")
    return tuple(sorted(found))


def _as_vector(value: Any) -> np.ndarray:
    if isinstance(value, Mapping):
        return np.asarray([float(value[k]) for k in sorted(value)], dtype=float)
    return np.ravel(np.asarray(value, dtype=float))


def _distance(name: DistanceName, left: Any, right: Any) -> float:
    if name in {"durations", "distributions"}:
        return float(wasserstein_distance(_as_vector(left), _as_vector(right)))
    left_vector = _as_vector(left)
    right_vector = _as_vector(right)
    if name == "transitions":
        left_vector = np.sort(left_vector)
        right_vector = np.sort(right_vector)
    delta = left_vector - right_vector
    return float(np.linalg.norm(delta))


def _coerce_method(
    method: str | AlignmentMethod | Sequence[DistanceName] | None,
) -> AlignmentMethod:
    if method is None or method == "auto":
        return AlignmentMethod()
    if isinstance(method, AlignmentMethod):
        return method
    if isinstance(method, str):
        return AlignmentMethod(distances=(method,))  # type: ignore[arg-type]
    return AlignmentMethod(distances=tuple(method))


def alignment_matrix(
    reference: Any,
    candidate: Any,
    method: str | AlignmentMethod | Sequence[DistanceName] | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Return the candidate/reference cost matrix without comparing raw labels."""
    matrix, _, _, _ = _build_costs(reference, candidate, _coerce_method(method))
    return tuple(tuple(float(x) for x in row) for row in matrix)


def align_states(
    reference: Any,
    candidate: Any,
    method: str | AlignmentMethod | Sequence[DistanceName] | None = None,
) -> AlignmentResult:
    """Align candidate states to reference states with Hungarian matching."""
    config = _coerce_method(method)
    costs, components, ref_labels, cand_labels = _build_costs(reference, candidate, config)
    rows, cols = linear_sum_assignment(costs)
    candidate_to_reference = {
        cand_labels[c]: ref_labels[r] for r, c in zip(rows, cols, strict=True)
    }
    reference_to_candidate = {ref: cand for cand, ref in candidate_to_reference.items()}
    notes = _ambiguity_warnings(costs, ref_labels, cand_labels, rows, cols, config)
    for note in notes:
        warnings.warn(note, RuntimeWarning, stacklevel=2)
    matched = {cand_labels[c]: float(costs[r, c]) for r, c in zip(rows, cols, strict=True)}
    diagnostics = AlignmentDiagnostics(
        total_cost=float(sum(matched.values())),
        matched_costs=matched,
        component_costs=components,
        candidate_to_reference=candidate_to_reference,
        reference_labels=ref_labels,
        candidate_labels=cand_labels,
        warnings=tuple(notes),
    )
    return AlignmentResult(
        alignment_matrix=tuple(tuple(float(x) for x in row) for row in costs),
        candidate_to_reference=candidate_to_reference,
        reference_to_candidate=reference_to_candidate,
        diagnostics=diagnostics,
    )


def aligned_labels(
    labels: Sequence[int], mapping: Mapping[int, int] | AlignmentResult
) -> tuple[int, ...]:
    """Apply a candidate-to-reference alignment mapping to candidate labels."""
    label_mapping = (
        mapping.candidate_to_reference if isinstance(mapping, AlignmentResult) else mapping
    )
    return tuple(int(label_mapping.get(int(label), int(label))) for label in labels)


def _build_costs(
    reference: Any, candidate: Any, method: AlignmentMethod
) -> tuple[np.ndarray, dict[tuple[int, int], dict[str, float]], tuple[int, ...], tuple[int, ...]]:
    ref_payload, cand_payload = _mapping(reference), _mapping(candidate)
    ref_labels, cand_labels = _labels(ref_payload), _labels(cand_payload)
    costs = np.zeros((len(ref_labels), len(cand_labels)), dtype=float)
    components: dict[tuple[int, int], dict[str, float]] = {}
    for i, ref_label in enumerate(ref_labels):
        for j, cand_label in enumerate(cand_labels):
            pair: dict[str, float] = {}
            for name in method.distances:
                ref_block = _first_present(ref_payload, name)
                cand_block = _first_present(cand_payload, name)
                ref_value = _state_value(ref_block, ref_label, transition_row=name == "transitions")
                cand_value = _state_value(
                    cand_block, cand_label, transition_row=name == "transitions"
                )
                if ref_value is None or cand_value is None:
                    continue
                pair[name] = _distance(name, ref_value, cand_value) * float(
                    method.weights.get(name, 1.0)
                )
            if not pair:
                raise ValueError(
                    "no shared descriptors for reference state "
                    f"{ref_label} and candidate state {cand_label}"
                )
            components[(ref_label, cand_label)] = pair
            costs[i, j] = sum(pair.values())
    return costs, components, ref_labels, cand_labels


def _ambiguity_warnings(
    costs: np.ndarray,
    ref_labels: tuple[int, ...],
    cand_labels: tuple[int, ...],
    rows: np.ndarray,
    cols: np.ndarray,
    method: AlignmentMethod,
) -> list[str]:
    notes: list[str] = []
    for r, c in zip(rows, cols, strict=True):
        row = np.sort(costs[r, :])
        col = np.sort(costs[:, c])
        best = float(costs[r, c])
        alternatives = [float(x[1]) for x in (row, col) if len(x) > 1]
        if alternatives and min(alternatives) - best <= max(
            method.ambiguity_epsilon, abs(best) * method.ambiguity_ratio
        ):
            notes.append(
                "Ambiguous alignment for candidate state "
                f"{cand_labels[c]} to reference state {ref_labels[r]}: "
                "competing costs are too close."
            )
    return notes


__all__ = [
    "AlignmentDiagnostics",
    "AlignmentMethod",
    "AlignmentResult",
    "align_states",
    "aligned_labels",
    "alignment_matrix",
]
