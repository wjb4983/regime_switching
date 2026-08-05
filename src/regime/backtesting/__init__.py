"""Backtesting package for regime switching workflows."""

from regime.backtesting.equity import (
    EquityBacktestConfig,
    EquityBacktestResult,
    PositionConstraints,
    run_equity_backtest,
)
from regime.backtesting.options import (
    OptionBacktestConfig,
    OptionBacktestResult,
    run_options_backtest,
)

__all__ = [
    "EquityBacktestConfig",
    "EquityBacktestResult",
    "OptionBacktestConfig",
    "OptionBacktestResult",
    "PositionConstraints",
    "run_equity_backtest",
    "run_options_backtest",
]
