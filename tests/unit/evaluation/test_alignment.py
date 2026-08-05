"""Tests for regime state alignment utilities."""

from __future__ import annotations

import pytest

from regime.evaluation import AlignmentMethod, align_states, aligned_labels, alignment_matrix


def test_align_states_uses_descriptors_instead_of_raw_labels() -> None:
    reference = {
        "means": {0: [0.0, 0.0], 1: [10.0, 1.0]},
        "covariances": {0: [[1.0, 0.0], [0.0, 1.0]], 1: [[2.0, 0.1], [0.1, 3.0]]},
        "return_risk": {0: {"return": 0.01, "risk": 0.10}, 1: {"return": -0.02, "risk": 0.30}},
        "durations": {0: [2, 3, 3], 1: [8, 9, 10]},
        "transitions": {"labels": (0, 1), "matrix": [[0.8, 0.2], [0.1, 0.9]]},
        "distributions": {0: [-1.0, 0.0, 1.0], 1: [9.0, 10.0, 11.0]},
    }
    candidate = {
        "means": {7: [10.1, 1.1], 3: [0.1, -0.1]},
        "covariances": {7: [[2.1, 0.1], [0.1, 2.9]], 3: [[1.1, 0.0], [0.0, 1.0]]},
        "return_risk": {7: {"return": -0.021, "risk": 0.31}, 3: {"return": 0.011, "risk": 0.11}},
        "durations": {7: [8, 9, 11], 3: [2, 3, 4]},
        "transitions": {"labels": (3, 7), "matrix": [[0.79, 0.21], [0.11, 0.89]]},
        "distributions": {7: [9.2, 10.1, 10.9], 3: [-0.9, 0.1, 1.1]},
    }

    result = align_states(reference, candidate, AlignmentMethod())

    assert result.candidate_to_reference == {3: 0, 7: 1}
    assert result.aligned_labels([7, 7, 3]) == (1, 1, 0)
    assert aligned_labels([3, 7], result) == (0, 1)
    assert len(result.alignment_matrix) == 2
    assert result.diagnostics.warnings == ()


def test_alignment_matrix_can_use_single_distance_method() -> None:
    matrix = alignment_matrix(
        {"means": {0: [0.0], 1: [5.0]}},
        {"means": {10: [4.9], 20: [0.2]}},
        "state_means",
    )

    assert matrix[0][1] < matrix[0][0]
    assert matrix[1][0] < matrix[1][1]


def test_ambiguous_alignment_warns_and_records_diagnostic() -> None:
    reference = {"means": {0: [0.0], 1: [0.01]}}
    candidate = {"means": {5: [0.004], 6: [0.006]}}

    with pytest.warns(RuntimeWarning, match="Ambiguous alignment"):
        result = align_states(reference, candidate, AlignmentMethod(ambiguity_ratio=1.0))

    assert result.diagnostics.warnings
