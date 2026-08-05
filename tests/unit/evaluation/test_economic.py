from __future__ import annotations

import numpy as np
import pandas as pd

from regime.evaluation.economic import (
    conditional_economic_metrics,
    economic_metrics,
    regime_baselines,
)


def test_economic_metrics_include_cost_exposure_capacity_and_factor_outputs() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01, 0.02])
    positions = pd.DataFrame({"asset_a": [1.0, 0.5, 0.0, 1.0, -0.5, 0.25]})
    factors = pd.DataFrame({"value": [0.005, -0.01, 0.01, 0.0, -0.005, 0.015]})

    metrics = economic_metrics(
        returns,
        positions=positions,
        benchmark_returns=pd.Series([0.005, -0.01, 0.01, 0.002, -0.003, 0.01]),
        factor_returns=factors,
        transaction_costs=np.full(6, 0.0001),
        slippage=np.full(6, 0.0002),
        borrow_costs=np.full(6, 0.00005),
        option_bid_ask_costs=np.full(6, 0.00015),
        margin_or_capital_usage=np.full(6, 0.6),
        volume=np.arange(1.0, 7.0),
        market_impact=np.full(6, 0.0003),
    )

    assert metrics.n_obs == 6
    assert metrics.max_drawdown < 0.0
    assert metrics.turnover > 0.0
    assert metrics.gross_exposure > 0.0
    assert metrics.total_costs == 0.0005
    assert metrics.margin_or_capital_usage == 0.6
    assert "value" in metrics.factor_exposures
    assert "average_volume" in metrics.capacity_proxies


def test_conditional_metrics_and_required_baselines() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.005])
    slices = {
        "inferred_regime": [0, 0, 1, 1, 1, 0],
        "asset": ["a", "a", "b", "b", "a", "b"],
    }

    conditional = conditional_economic_metrics(returns, slices, periods_per_year=12)
    baselines = regime_baselines(returns, regimes=slices["inferred_regime"], periods_per_year=12)

    assert set(conditional) == {"inferred_regime", "asset"}
    assert set(conditional["inferred_regime"]) == {"0", "1"}
    assert set(baselines) == {
        "no_regime",
        "simple_rule",
        "oracle_or_synthetic_upper_bound",
    }
    assert (
        baselines["oracle_or_synthetic_upper_bound"]["annualized_return"]
        >= baselines["no_regime"]["annualized_return"]
    )
