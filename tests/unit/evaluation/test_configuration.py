"""Evaluation configuration and metric-contract tests."""

import pytest
from pydantic import ValidationError

from regime.evaluation.config import RegimeQualityEvaluation, parse_evaluation_config
from regime.evaluation.metrics import ProbabilityKind
from regime.evaluation.regime_quality import METRICS as QUALITY_METRICS
from regime.evaluation.statistical import METRICS as STATISTICAL_METRICS


def _base() -> dict[str, object]:
    return {
        "source": {"model": "kmeans"},
        "dataset": "features.parquet",
        "output_dir": "evaluation",
        "splitter": {"kind": "rolling"},
        "features": ["return", "volatility"],
    }


def test_parser_requires_discriminator_instead_of_inferring_from_keys() -> None:
    with pytest.raises(ValidationError, match="evaluation_type"):
        parse_evaluation_config({**_base(), "returns_column": "return"})

    parsed = parse_evaluation_config({**_base(), "evaluation_type": "regime_quality"})
    assert isinstance(parsed, RegimeQualityEvaluation)


def test_metric_compatibility_checks_inputs_and_probability_kind() -> None:
    brier = STATISTICAL_METRICS["brier_score"]
    assert brier.compatible({"truth", "probabilities"}, ProbabilityKind.FILTERED)
    assert not brier.compatible({"probabilities"}, ProbabilityKind.FILTERED)
    assert not brier.compatible({"truth", "probabilities"}, ProbabilityKind.SMOOTHED)
    persistence = QUALITY_METRICS["regime_persistence"]
    assert persistence.compatible({"states"})
