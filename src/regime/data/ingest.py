"""Data-ingestion workflow implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from regime.data.options import BlackScholesInputs, implied_volatility, option_greeks
from regime.data.providers.massive import (
    OPTIONS_ALL_HISTORY_START,
    PLANS,
    STOCKS_ALL_HISTORY_START,
    MassiveAPIError,
    MassiveClient,
)


def run_data_ingest(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Execute provider-specific ingestion and return command metadata."""
    provider = str(config.get("provider", "mock")).lower()
    if provider == "massive":
        return _run_massive_ingest(config)
    if provider == "mock":
        return _run_mock_ingest(config)
    raise ValueError(f"unsupported data provider: {provider}")


def _run_mock_ingest(config: Mapping[str, Any]) -> Mapping[str, Any]:
    source = Path(str(config["source"]))
    output = Path(str(config["output"]))
    timestamp_column = str(config.get("timestamp_column", "timestamp"))
    symbol = str(config.get("symbol", "MOCK"))
    frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    if timestamp_column not in frame.columns:
        raise ValueError(f"timestamp column {timestamp_column!r} not found in {source}")
    out = frame.copy()
    out = out.rename(columns={timestamp_column: "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert(None)
    out["symbol"] = symbol
    for required, default in (
        ("open", out.get("close", out.iloc[:, 1] if len(out.columns) > 1 else 0.0)),
        ("high", out.get("close", out.iloc[:, 1] if len(out.columns) > 1 else 0.0)),
        ("low", out.get("close", out.iloc[:, 1] if len(out.columns) > 1 else 0.0)),
        ("close", out.iloc[:, 1] if len(out.columns) > 1 else 0.0),
        ("volume", 0.0),
        ("vwap", out.get("close", out.iloc[:, 1] if len(out.columns) > 1 else 0.0)),
    ):
        if required not in out.columns:
            out[required] = default
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    result: dict[str, Any] = {"provider": "mock", "output": str(output), "rows": int(len(out))}
    options_config = config.get("options", {})
    if isinstance(options_config, Mapping) and options_config.get("enabled"):
        options_output = Path(str(config["options_output"]))
        options_frame = _build_mock_options_chain(out)
        options_output.parent.mkdir(parents=True, exist_ok=True)
        options_frame.to_parquet(options_output, index=False)
        result["options_output"] = str(options_output)
        result["options_rows"] = int(len(options_frame))
    return result


def _run_massive_ingest(config: Mapping[str, Any]) -> Mapping[str, Any]:
    output = Path(str(config["output"]))
    symbols = _symbols_from_config(config)
    end_date = _coerce_date(config.get("end_date")) or date.today()
    adjusted = bool(config.get("adjusted", True))
    timespan = str(config.get("timespan", "day"))
    multiplier = int(config.get("multiplier", 1))
    plan_name = str(config.get("plan", "starter")).lower()
    if plan_name not in PLANS:
        raise ValueError(f"unsupported MASSIVE plan {plan_name!r}; choose one of {sorted(PLANS)}")
    client = MassiveClient.from_api_key(
        config.get("api_key"),
        base_url=str(config.get("base_url", "https://api.massive.com")),
        secret_file=str(config["secret_file"]) if "secret_file" in config else None,
    )

    stock_start = _history_start(
        requested_start=_coerce_date(config.get("start_date")),
        history_years=PLANS[plan_name].stocks_history_years,
        full_history_start=STOCKS_ALL_HISTORY_START,
        end_date=end_date,
        max_history=bool(config.get("max_history", True)),
    )
    stock_start = _resolve_accessible_stock_start(
        client,
        symbols=symbols,
        start_date=stock_start,
        end_date=end_date,
        adjusted=adjusted,
        multiplier=multiplier,
        timespan=timespan,
    )
    stock_frame = _download_stock_bars(
        client,
        symbols=symbols,
        start_date=stock_start,
        end_date=end_date,
        adjusted=adjusted,
        multiplier=multiplier,
        timespan=timespan,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    stock_frame.to_parquet(output, index=False)
    result: dict[str, Any] = {
        "provider": "massive",
        "output": str(output),
        "rows": int(len(stock_frame)),
        "symbols": symbols,
        "start_date": stock_start.isoformat(),
        "end_date": end_date.isoformat(),
        "plan": plan_name,
    }

    options_config = config.get("options", {})
    if isinstance(options_config, Mapping) and options_config.get("enabled"):
        options_output = Path(str(config["options_output"]))
        contracts_output_value = config.get("options_contracts_output")
        contracts_output = (
            Path(str(contracts_output_value)) if contracts_output_value is not None else None
        )
        options_frame, contracts_frame = _download_option_history(
            client,
            symbols=symbols,
            stock_frame=stock_frame,
            end_date=end_date,
            adjusted=adjusted,
            config=options_config,
            plan_name=plan_name,
        )
        options_output.parent.mkdir(parents=True, exist_ok=True)
        options_frame.to_parquet(options_output, index=False)
        result["options_output"] = str(options_output)
        result["options_rows"] = int(len(options_frame))
        if contracts_output is not None:
            contracts_output.parent.mkdir(parents=True, exist_ok=True)
            contracts_frame.to_parquet(contracts_output, index=False)
            result["options_contracts_output"] = str(contracts_output)
            result["contracts_rows"] = int(len(contracts_frame))
    return result


def _download_stock_bars(
    client: MassiveClient,
    *,
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    adjusted: bool,
    multiplier: int,
    timespan: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        records = client.fetch_stock_aggregates(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjusted=adjusted,
            multiplier=multiplier,
            timespan=timespan,
        )
        if not records:
            continue
        frame = pd.DataFrame.from_records(records)
        frame["symbol"] = symbol
        frames.append(_normalize_stock_aggregates(frame))
    if not frames:
        raise ValueError("MASSIVE returned no stock aggregate rows for the configured request")
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(
        drop=True
    )


def _download_option_history(
    client: MassiveClient,
    *,
    symbols: Sequence[str],
    stock_frame: pd.DataFrame,
    end_date: date,
    adjusted: bool,
    config: Mapping[str, Any],
    plan_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    option_start = _history_start(
        requested_start=_coerce_date(config.get("start_date")),
        history_years=PLANS[plan_name].options_history_years,
        full_history_start=OPTIONS_ALL_HISTORY_START,
        end_date=end_date,
        max_history=bool(config.get("max_history", True)),
    )
    max_contracts = int(config.get("max_contracts", 250))
    contracts_frames: list[pd.DataFrame] = []
    option_frames: list[pd.DataFrame] = []
    option_timespan = str(config.get("timespan", "day"))
    option_multiplier = int(config.get("multiplier", 1))
    quote_spread_pct = float(config.get("derived_quote_spread_pct", 0.02))
    contract_type = str(config["contract_type"]) if config.get("contract_type") else None
    expiration_date_gte = _coerce_date(config.get("expiration_date_gte"))
    expiration_date_lte = _coerce_date(config.get("expiration_date_lte"))
    risk_free_rate = float(config.get("risk_free_rate", 0.0))
    dividend_yield = float(config.get("dividend_yield", 0.0))

    stock_lookup = stock_frame.copy()
    stock_lookup["trade_date"] = pd.to_datetime(stock_lookup["timestamp"]).dt.date
    stock_lookup = stock_lookup.set_index(["symbol", "trade_date"])
    for symbol in symbols:
        contract_records = client.fetch_option_contracts(
            underlying_symbol=symbol,
            as_of_date=end_date,
            expired=bool(config.get("expired", True)),
            limit=min(max_contracts, 1_000),
            contract_type=contract_type,
            expiration_date_gte=expiration_date_gte,
            expiration_date_lte=expiration_date_lte,
        )
        if not contract_records:
            continue
        if len(contract_records) > max_contracts:
            contract_records = contract_records[:max_contracts]
        contracts_frame = pd.DataFrame.from_records(contract_records)
        contracts_frames.append(contracts_frame.assign(underlying_symbol=symbol))
        for contract in contract_records:
            contract_start = _resolve_accessible_option_start(
                client,
                option_ticker=str(contract["ticker"]),
                start_date=option_start,
                end_date=end_date,
                adjusted=adjusted,
                multiplier=option_multiplier,
                timespan=option_timespan,
            )
            option_records = client.fetch_option_aggregates(
                option_ticker=str(contract["ticker"]),
                start_date=contract_start,
                end_date=end_date,
                adjusted=adjusted,
                multiplier=option_multiplier,
                timespan=option_timespan,
            )
            if not option_records:
                continue
            option_frame = _normalize_option_aggregates(
                pd.DataFrame.from_records(option_records),
                contract=contract,
                stock_lookup=stock_lookup,
                quote_spread_pct=quote_spread_pct,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
            )
            if not option_frame.empty:
                option_frames.append(option_frame)
    if not option_frames:
        raise ValueError("MASSIVE returned no option aggregate rows for the configured request")
    option_result = pd.concat(option_frames, ignore_index=True).sort_values(
        ["underlying_symbol", "timestamp", "expiration", "strike", "option_type"]
    )
    contracts_result = (
        pd.concat(contracts_frames, ignore_index=True).sort_values(["underlying_ticker", "ticker"])
        if contracts_frames
        else pd.DataFrame()
    )
    return option_result.reset_index(drop=True), contracts_result.reset_index(drop=True)


def _normalize_stock_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_convert(None),
            "symbol": frame["symbol"].astype(str),
            "open": frame["o"].astype(float),
            "high": frame["h"].astype(float),
            "low": frame["l"].astype(float),
            "close": frame["c"].astype(float),
            "volume": frame["v"].astype(float),
            "vwap": frame.get("vw", pd.Series(index=frame.index, dtype=float)).astype(float),
            "transactions": frame.get("n", pd.Series(index=frame.index, dtype=float)).astype(float),
        }
    )


def _normalize_option_aggregates(
    frame: pd.DataFrame,
    *,
    contract: Mapping[str, Any],
    stock_lookup: pd.DataFrame,
    quote_spread_pct: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> pd.DataFrame:
    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_convert(None),
            "quote_time": pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_convert(None),
            "option_symbol": str(contract["ticker"]),
            "underlying_symbol": str(contract["underlying_ticker"]),
            "expiration": pd.to_datetime(contract["expiration_date"]),
            "strike": float(contract["strike_price"]),
            "option_type": str(contract["contract_type"]),
            "open": frame["o"].astype(float),
            "high": frame["h"].astype(float),
            "low": frame["l"].astype(float),
            "close": frame["c"].astype(float),
            "volume": frame["v"].astype(float),
            "vwap": frame.get("vw", pd.Series(index=frame.index, dtype=float)).astype(float),
            "transactions": frame.get("n", pd.Series(index=frame.index, dtype=float)).astype(float),
            "multiplier": float(contract.get("shares_per_contract", 100.0)),
            "data_source": "massive_daily_aggregates",
            "field_derivation": "bid_ask_iv_delta_derived_from_daily_ohlc",
        }
    )
    if normalized.empty:
        return normalized
    normalized["trade_date"] = pd.to_datetime(normalized["timestamp"]).dt.date
    underlying_prices: list[float | None] = []
    for row in normalized.itertuples(index=False):
        key = (row.underlying_symbol, row.trade_date)
        try:
            underlying_prices.append(float(stock_lookup.loc[key, "close"]))
        except KeyError:
            underlying_prices.append(None)
    normalized["underlying_price"] = underlying_prices
    normalized = normalized.dropna(subset=["underlying_price"]).copy()
    if normalized.empty:
        return normalized
    mid = normalized["close"].astype(float).clip(lower=0.01)
    half_spread = (mid * (quote_spread_pct / 2.0)).clip(lower=0.01)
    normalized["bid"] = (mid - half_spread).clip(lower=0.01)
    normalized["ask"] = mid + half_spread
    normalized["bid_size"] = 1.0
    normalized["ask_size"] = 1.0
    normalized["open_interest"] = normalized["volume"].rolling(5, min_periods=1).mean()
    iv_values: list[float | None] = []
    delta_values: list[float | None] = []
    gamma_values: list[float | None] = []
    theta_values: list[float | None] = []
    vega_values: list[float | None] = []
    rho_values: list[float | None] = []
    for row in normalized.itertuples(index=False):
        tenor_days = max((pd.Timestamp(row.expiration) - pd.Timestamp(row.timestamp)).days, 1)
        inputs = BlackScholesInputs(
            spot=float(row.underlying_price),
            strike=float(row.strike),
            tenor=tenor_days / 365.0,
            rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        option_type = "call" if str(row.option_type).lower() == "call" else "put"
        iv = implied_volatility(float(row.close), inputs, option_type)
        if iv is None:
            iv = 0.25
        greeks = option_greeks(inputs, iv, option_type)
        iv_values.append(iv)
        delta_values.append(greeks.delta)
        gamma_values.append(greeks.gamma)
        theta_values.append(greeks.theta)
        vega_values.append(greeks.vega)
        rho_values.append(greeks.rho)
    normalized["implied_volatility"] = iv_values
    normalized["delta"] = delta_values
    normalized["gamma"] = gamma_values
    normalized["theta"] = theta_values
    normalized["vega"] = vega_values
    normalized["rho"] = rho_values
    normalized["symbol"] = normalized["option_symbol"]
    return normalized.drop(columns=["trade_date"]).reset_index(drop=True)


def _build_mock_options_chain(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        spot = float(row.close)
        for option_type, strike in (("put", spot * 0.95), ("call", spot * 1.05)):
            rows.append(
                {
                    "timestamp": timestamp,
                    "quote_time": timestamp - pd.Timedelta(minutes=1),
                    "expiration": timestamp + pd.Timedelta(days=30),
                    "strike": strike,
                    "option_type": option_type,
                    "bid": 1.0,
                    "ask": 1.1,
                    "underlying_price": spot,
                    "delta": -0.25 if option_type == "put" else 0.25,
                    "implied_volatility": 0.25,
                    "volume": 100.0,
                    "open_interest": 500.0,
                    "bid_size": 10.0,
                    "ask_size": 10.0,
                    "multiplier": 100.0,
                    "symbol": f"{row.symbol}_{option_type.upper()}",
                    "underlying_symbol": row.symbol,
                }
            )
    return pd.DataFrame(rows)


def _history_start(
    *,
    requested_start: date | None,
    history_years: int | None,
    full_history_start: date,
    end_date: date,
    max_history: bool,
) -> date:
    if requested_start is not None:
        return requested_start
    if not max_history:
        return end_date - timedelta(days=365)
    if history_years is None:
        return full_history_start
    return end_date - timedelta(days=365 * history_years)


def _resolve_accessible_stock_start(
    client: MassiveClient,
    *,
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    adjusted: bool,
    multiplier: int,
    timespan: str,
) -> date:
    resolved = start_date
    for symbol in symbols:
        resolved = max(
            resolved,
            _resolve_accessible_start_date(
                lambda probe_start, probe_end: client.fetch_stock_aggregates(
                    symbol=symbol,
                    start_date=probe_start,
                    end_date=probe_end,
                    adjusted=adjusted,
                    multiplier=multiplier,
                    timespan=timespan,
                    limit=10,
                ),
                start_date=start_date,
                end_date=end_date,
            ),
        )
    return resolved


def _resolve_accessible_option_start(
    client: MassiveClient,
    *,
    option_ticker: str,
    start_date: date,
    end_date: date,
    adjusted: bool,
    multiplier: int,
    timespan: str,
) -> date:
    return _resolve_accessible_start_date(
        lambda probe_start, probe_end: client.fetch_option_aggregates(
            option_ticker=option_ticker,
            start_date=probe_start,
            end_date=probe_end,
            adjusted=adjusted,
            multiplier=multiplier,
            timespan=timespan,
            limit=10,
        ),
        start_date=start_date,
        end_date=end_date,
    )


def _resolve_accessible_start_date(
    fetch_probe: Any,
    *,
    start_date: date,
    end_date: date,
) -> date:
    if start_date >= end_date:
        return start_date
    probe_end = end_date
    try:
        fetch_probe(start_date, probe_end)
        return start_date
    except MassiveAPIError as exc:
        if "HTTP 401" not in str(exc):
            raise

    low = start_date
    high = end_date
    while (high - low).days > 1:
        mid = low + timedelta(days=(high - low).days // 2)
        try:
            fetch_probe(mid, probe_end)
            high = mid
        except MassiveAPIError as exc:
            if "HTTP 401" not in str(exc):
                raise
            low = mid
    return high


def _symbols_from_config(config: Mapping[str, Any]) -> list[str]:
    symbols_value = config.get("symbols")
    if symbols_value is not None:
        if not isinstance(symbols_value, Sequence) or isinstance(symbols_value, str):
            raise ValueError("'symbols' must be a list of ticker strings")
        symbols = [str(symbol).upper() for symbol in symbols_value]
    elif "symbol" in config:
        symbols = [str(config["symbol"]).upper()]
    else:
        raise ValueError("MASSIVE ingestion requires 'symbol' or 'symbols'")
    if not symbols:
        raise ValueError("at least one ticker symbol is required")
    return symbols


def _coerce_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.astimezone(UTC).date()
    return datetime.fromisoformat(str(value)).date()
