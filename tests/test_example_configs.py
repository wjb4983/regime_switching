"""Contract tests for the ordered, executable example configuration set."""

from __future__ import annotations

from pathlib import Path

import pytest

from regime.config.base import load_yaml_mapping
from regime.tuning.config import SearchSpace

pytestmark = pytest.mark.timeout(5)

CONFIGS = (
    "synthetic/gaussian_hmm.yaml",
    "data/mock_provider.yaml",
    "features/core_market.yaml",
    "models/rule_volatility_threshold.yaml",
    "models/kmeans_regime.yaml",
    "models/gaussian_hmm.yaml",
    "models/student_t_hmm.yaml",
    "validation/walk_forward_daily.yaml",
    "evaluation/statistical_regime_quality.yaml",
    "evaluation/downstream_vol_targeting.yaml",
    "backtesting/equity_vol_targeting.yaml",
    "backtesting/options_delta_hedged.yaml",
    "tuning/hmm_search.yaml",
    "report/research_report.yaml",
)


@pytest.mark.parametrize("relative_path", CONFIGS)
def test_example_config_is_a_nonempty_mapping(relative_path: str) -> None:
    config = load_yaml_mapping(Path("configs") / relative_path)

    assert config


def test_hmm_search_is_an_executable_search_space() -> None:
    search = SearchSpace.from_yaml("configs/tuning/hmm_search.yaml")

    assert tuple(search.parameters) == (
        "n_states",
        "covariance_regularization",
        "sticky_strength",
        "student_t_dof",
    )
