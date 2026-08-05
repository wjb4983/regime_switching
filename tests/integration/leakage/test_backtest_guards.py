import pandas as pd

from regime.backtesting.equity import EquityBacktestConfig, run_equity_backtest
from regime.backtesting.options import OptionBacktestConfig, run_options_backtest
from regime.validation import PurgedTimeSeriesSplitter


def test_option_chain_excludes_contract_not_available_at_decision_time() -> None:
    decision = pd.Timestamp("2024-01-02 10:00")
    chain = pd.DataFrame(
        {
            "timestamp": [decision, decision],
            "quote_time": [decision - pd.Timedelta(minutes=1), decision + pd.Timedelta(minutes=1)],
            "expiration": [decision + pd.Timedelta(days=30)] * 2,
            "strike": [100.0, 105.0],
            "option_type": ["call", "call"],
            "bid": [1.0, 1.0],
            "ask": [1.1, 1.1],
            "underlying_price": [100.0, 100.0],
            "delta": [0.25, 0.25],
            "implied_volatility": [0.2, 0.2],
            "open_interest": [1_000.0, 1_000.0],
        }
    )

    result = run_options_backtest(
        chain,
        pd.Series([100.0], index=[decision]),
        config=OptionBacktestConfig(execution_delay=0),
    )

    assert result.filtered_chain["strike"].tolist() == [100.0]
    assert result.rejected_chain["strike"].tolist() == [105.0]


def test_execution_delay_is_applied_before_returns() -> None:
    dates = pd.date_range("2024-01-01", periods=4)
    returns = pd.DataFrame({"asset": [0.10, 0.20, 0.30, 0.40]}, index=dates)
    targets = pd.DataFrame({"asset": [1.0, 0.0, 0.0, 0.0]}, index=dates)

    result = run_equity_backtest(
        returns,
        targets,
        config=EquityBacktestConfig(execution_delay=1, transaction_cost_bps=0.0),
    )

    assert result.weights["asset"].tolist()[:2] == [0.0, 1.0]
    assert result.returns.iloc[0] == 0.0
    assert result.returns.iloc[1] == 0.20


def test_embargo_and_purge_remove_training_labels_overlapping_evaluation() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8),
            "label_start": pd.date_range("2024-01-01", periods=8),
            "label_end": pd.date_range("2024-01-03", periods=8),
        }
    )
    split = next(
        PurgedTimeSeriesSplitter(
            initial_train_size=4, validation_size=1, test_size=1, embargo=1
        ).split(frame)
    )

    evaluation_start = frame.loc[min(split.validation + split.test), "label_start"]
    assert all(frame.loc[index, "label_end"] <= evaluation_start for index in split.train)
    assert set(split.embargo).isdisjoint(split.train)
