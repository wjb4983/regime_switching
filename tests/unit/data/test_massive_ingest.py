from __future__ import annotations

from datetime import date

import pandas as pd

from regime.data.ingest import run_data_ingest
from regime.data.providers.massive import MassiveClient


class _FakeMassiveClient:
    def fetch_stock_aggregates(self, **_: object) -> list[dict[str, object]]:
        return [
            {"t": 1704153600000, "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1000, "vw": 100.2, "n": 10},
            {"t": 1704240000000, "o": 100.5, "h": 102.0, "l": 100.0, "c": 101.5, "v": 1100, "vw": 101.1, "n": 11},
        ]

    def fetch_option_contracts(self, **_: object) -> list[dict[str, object]]:
        return [
            {
                "ticker": "O:SPY240201C00500000",
                "underlying_ticker": "SPY",
                "expiration_date": "2024-02-01",
                "strike_price": 500.0,
                "contract_type": "call",
                "shares_per_contract": 100,
            }
        ]

    def fetch_option_aggregates(self, **_: object) -> list[dict[str, object]]:
        return [
            {"t": 1704153600000, "o": 1.0, "h": 1.1, "l": 0.9, "c": 1.0, "v": 25, "vw": 1.02, "n": 4},
            {"t": 1704240000000, "o": 1.1, "h": 1.2, "l": 1.0, "c": 1.15, "v": 30, "vw": 1.10, "n": 5},
        ]


def test_massive_ingest_writes_stock_and_option_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(MassiveClient, "from_api_key", classmethod(lambda cls, *args, **kwargs: _FakeMassiveClient()))  # type: ignore[method-assign]
    config = {
        "provider": "massive",
        "plan": "starter",
        "symbol": "SPY",
        "output": str(tmp_path / "stocks.parquet"),
        "options_output": str(tmp_path / "options.parquet"),
        "options_contracts_output": str(tmp_path / "contracts.parquet"),
        "start_date": date(2024, 1, 2).isoformat(),
        "end_date": date(2024, 1, 3).isoformat(),
        "api_key": "test-key",
        "options": {"enabled": True, "max_contracts": 5},
    }

    result = run_data_ingest(config)

    stocks = pd.read_parquet(tmp_path / "stocks.parquet")
    options = pd.read_parquet(tmp_path / "options.parquet")
    contracts = pd.read_parquet(tmp_path / "contracts.parquet")

    assert result["rows"] == 2
    assert list(stocks["symbol"].unique()) == ["SPY"]
    assert {"bid", "ask", "delta", "implied_volatility", "underlying_price"}.issubset(options.columns)
    assert len(options) == 2
    assert len(contracts) == 1
