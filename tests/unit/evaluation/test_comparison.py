from __future__ import annotations

import numpy as np
import pandas as pd

from regime.evaluation.comparison import (
    align_loss_matrix,
    block_bootstrap_indices,
    deflated_sharpe_ratio,
    diebold_mariano_test,
    false_discovery_control,
    model_confidence_set,
    probabilistic_sharpe_ratio,
    reality_check,
    stationary_bootstrap_indices,
    superior_predictive_ability_test,
)


def test_pairwise_comparison_aligns_out_of_sample_indexes() -> None:
    idx = pd.date_range("2024-01-01", periods=5)
    model = pd.Series([1.0, 0.9, np.nan, 0.8, 0.7], index=idx)
    benchmark = pd.Series([1.1, 1.0, 0.95, 0.9, 0.85], index=idx[0:5])

    result = diebold_mariano_test(model, benchmark, bandwidth=0)

    assert result.n_obs == 4
    assert 0.0 <= result.p_value <= 1.0
    assert result.documentation.null_hypothesis


def test_bootstrap_indices_have_expected_shape_and_bounds() -> None:
    block = block_bootstrap_indices(20, block_length=4, n_bootstrap=3, random_state=1)
    stationary = stationary_bootstrap_indices(
        20, average_block_length=4, n_bootstrap=3, random_state=1
    )

    assert block.shape == (3, 20)
    assert stationary.shape == (3, 20)
    assert np.all((0 <= block) & (block < 20))
    assert np.all((0 <= stationary) & (stationary < 20))


def test_multiple_model_bootstrap_tests_return_documented_probabilities() -> None:
    rng = np.random.default_rng(2)
    losses = pd.DataFrame(
        {
            "candidate_a": rng.normal(0.9, 0.05, 40),
            "candidate_b": rng.normal(1.1, 0.05, 40),
            "candidate_c": rng.normal(1.0, 0.05, 40),
        },
        index=pd.date_range("2024-01-01", periods=40),
    )
    benchmark = pd.Series(rng.normal(1.0, 0.05, 40), index=losses.index)

    aligned = align_loss_matrix(losses)
    mcs = model_confidence_set(aligned, n_bootstrap=20, random_state=3)
    rc = reality_check(aligned, benchmark, n_bootstrap=20, random_state=3)
    spa = superior_predictive_ability_test(aligned, benchmark, n_bootstrap=20, random_state=3)

    assert mcs.included_models
    assert rc.documentation.valid_use_cases
    assert spa.documentation.invalid_use_cases
    assert 0.0 <= rc.p_value <= 1.0
    assert 0.0 <= spa.p_value <= 1.0


def test_sharpe_and_false_discovery_helpers_are_bounded() -> None:
    rng = np.random.default_rng(4)
    returns = rng.normal(0.01, 0.05, 80)

    psr = probabilistic_sharpe_ratio(returns, periods_per_year=252)
    dsr = deflated_sharpe_ratio(returns, n_trials=25, periods_per_year=252)
    fdr = false_discovery_control([0.01, 0.04, 0.50], alpha=0.05)

    assert 0.0 <= psr.p_value <= 1.0
    assert 0.0 <= dsr.p_value <= 1.0
    assert fdr.adjusted_p_values == (0.03, 0.06, 0.5)
    assert fdr.rejected == (True, False, False)
